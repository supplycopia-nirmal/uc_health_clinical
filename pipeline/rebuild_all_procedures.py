"""
Complete Pipeline to Rebuild All 1,400+ Procedures for UC Health.
1. Extracts all records from Snowflake SC_MULTI_TENANT_CONSUMPTION_SAVINGS (all procedures).
2. Computes case-level metrics and Isolation Forest + compositional anomalies.
3. Generates Standard BOMs for all procedures (Core Pack, Hold Cart, Specialty PRN).
4. Mines Master and Physician Substitution Construct Matrices across all procedures.
5. Saves local CSV artifacts for instant high-speed dashboard consumption.
"""

import os
import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MultiTenantConfig, get_snowflake_connection

def extract_all_procedures() -> pd.DataFrame:
    print(f"\n=======================================================")
    print(f" STEP 1: EXTRACTING ALL PROCEDURES FROM SNOWFLAKE")
    print(f"=======================================================")
    t0 = time.time()
    conn = get_snowflake_connection()
    try:
        query = f"""
        SELECT 
            COALESCE(NULLIF(TRIM(STANDARD_FACILITY), ''), NULLIF(TRIM(FACILITY), ''), 'Main Hospital') AS "hospital_name",
            TRIM(CASE_ID) AS "case_id",
            COALESCE(NULLIF(TRIM(STANDARDIZED_PROCEDURE), ''), NULLIF(TRIM(PRIMARY_PROCEDURE), ''), 'UNSPECIFIED') AS "unified_procedure",
            COALESCE(TOTAL_QUANTITY, 1.0) AS "quantity",
            COALESCE(NULLIF(TRIM(PRIMARY_PROCEDURE_GROUP), ''), NULLIF(TRIM(SURGICAL_HIERARCHY), ''), 'General Surgery') AS "contract_vertical",
            COALESCE(NULLIF(TRIM(PRODUCT_SUBCLASS), ''), NULLIF(TRIM(CONTRACT_CATEGORY), ''), 'Medical-Surgical Supplies') AS "contract_category",
            CASE 
                WHEN MAPPED_UNSPSC IS NOT NULL AND UNSPSC_DESCRIPTION IS NOT NULL 
                THEN TRIM(MAPPED_UNSPSC) || '-' || UPPER(TRIM(UNSPSC_DESCRIPTION))
                WHEN UNSPSC_CODE IS NOT NULL AND UNSPSC_DESCRIPTION IS NOT NULL
                THEN REPLACE(TRIM(UNSPSC_CODE), 'UNSPC:', '') || '-' || UPPER(TRIM(UNSPSC_DESCRIPTION))
                WHEN MAPPED_UNSPSC IS NOT NULL THEN TRIM(MAPPED_UNSPSC)
                ELSE COALESCE(NULLIF(TRIM(PRODUCT_SUBCLASS), ''), 'GENERAL MEDICAL')
            END AS "contract_sub_category",
            COALESCE(NULLIF(TRIM(MANUFACTURER_CATALOG_NUMBER), ''), NULLIF(TRIM(ITEM_NUMBER), ''), 'UNKNOWN') AS "part_number",
            COALESCE(NULLIF(TRIM(MANUFACTURER_NAME), ''), NULLIF(TRIM(VENDOR_NAME_STANDARD), ''), 'UNKNOWN') AS "manufacturer",
            COALESCE(NULLIF(TRIM(VENDOR_NAME_STANDARD), ''), 'UNKNOWN') AS "supplier_name",
            COALESCE(NULLIF(TRIM(MANUFACTURER_NAME), ''), NULLIF(TRIM(VENDOR_NAME_STANDARD), ''), 'UNKNOWN') AS "brand",
            COALESCE(NULLIF(TRIM(ITEM_DESCRIPTION), ''), 'UNKNOWN') AS "item_description",
            COALESCE(SUPPLY_UNIT_PRICE, 0.0) AS "unit_cost",
            COALESCE(LINE_SPEND, TOTAL_ACQUISITION_COST, 0.0) AS "line_spend",
            COALESCE(TOTAL_CHARGES, 0.0) AS "total_charges",
            COALESCE(NULLIF(TRIM(LEAD_SURGEON), ''), 'UNASSIGNED') AS "lead_surgeon",
            COALESCE(NULLIF(TRIM(PATIENT_AGE_BUCKET), ''), 'Not Specified') AS "patient_age_bucket",
            COALESCE(NULLIF(TRIM(ASA_RATING), ''), 'Not Specified') AS "asa_rating",
            COALESCE(NULLIF(TRIM(BMI_BUCKET), ''), 'Not Specified') AS "bmi_bucket",
            COALESCE(NULLIF(TRIM(PATIENT_TYPE), ''), 'Inpatient') AS "patient_type_bucket",
            COALESCE(NULLIF(TRIM(SPEND_CATEGORY), ''), 'SUPPLIES') AS "spend_category",
            CASE 
                WHEN IS_CONTRACT_COMPLIANT = TRUE OR CONTRACT_STATUS ILIKE '%on%' THEN 'ON CONTRACT'
                ELSE 'OFF CONTRACT'
            END AS "contract_flag",
            COALESCE(NULLIF(TRIM(SMOKING_STATUS), ''), 'Non-Smoker') AS "smoking_status",
            COALESCE(NULLIF(TRIM(DIABETIC_STATUS), ''), '0') AS "diabetic_status",
            COALESCE(NULLIF(TRIM(SERVICE_LINE), ''), 'General') AS "service_line",
            COALESCE(CONSUMPTION_YEAR_MONTH, '2025-01') AS "discharge_date",
            COALESCE(SAVINGS_OPPORTUNITY, 0.0) AS "savings_opportunity",
            COALESCE(PRICE_VARIANCE2, 0.0) AS "price_variance"
        FROM {MultiTenantConfig.SOURCE_TABLE}
        WHERE CLIENT_ID = '{MultiTenantConfig.TENANT_ID}'
          AND CASE_ID IS NOT NULL
          AND STANDARDIZED_PROCEDURE IS NOT NULL
        """
        print("Executing Snowflake query for ALL procedures...")
        cur = conn.cursor()
        cur.execute(query)
        df = cur.fetch_pandas_all()
        cur.close()
        df.columns = df.columns.str.lower()
        print(f"Extracted {len(df):,} records, {df['case_id'].nunique():,} cases, and {df['unified_procedure'].nunique():,} procedures in {round(time.time() - t0, 1)}s")
        return df
    finally:
        conn.close()

def compute_anomalies_and_boms(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=======================================================")
    print(f" STEP 2: CASE-LEVEL METRICS, ANOMALY DETECTION & BOMS")
    print(f"=======================================================")
    t0 = time.time()
    
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(1.0)
    df['unit_cost'] = pd.to_numeric(df['unit_cost'], errors='coerce').fillna(0.0)
    df['line_spend'] = pd.to_numeric(df['line_spend'], errors='coerce').fillna(0.0)
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0.0)
    else:
        df['total_charges'] = 0.0

    print("Computing case-level aggregations...")
    case_summary = df.groupby(['unified_procedure', 'case_id']).agg(
        total_qty=('quantity', 'sum'),
        total_cost=('line_spend', 'sum'),
        total_charges=('total_charges', 'first')
    ).reset_index()

    case_summary['cost_volume_outlier'] = 'No'
    case_summary['composition_outlier'] = 'No'
    case_summary['if_anomaly_score'] = 0.0
    case_summary['anomaly_type_detail'] = 'Normal Case'
    case_summary['outlier_items'] = ''

    procedures = case_summary['unified_procedure'].unique()
    print(f"Processing models across {len(procedures):,} procedures...")

    bom_records = []
    case_outlier_map = {}

    # Pre-group df by procedure for fast slicing
    proc_groups = df.groupby('unified_procedure')

    for proc in procedures:
        try:
            proc_cases = case_summary[case_summary['unified_procedure'] == proc]
            n_cases = len(proc_cases)
            if n_cases == 0:
                continue
                
            proc_lines = proc_groups.get_group(proc) if proc in proc_groups.groups else pd.DataFrame()
            if proc_lines.empty:
                continue

            # Standard BOM computation
            item_stats = proc_lines.groupby('item_description').agg(
                case_count=('case_id', 'nunique'),
                total_spend=('line_spend', 'sum'),
                avg_unit_cost=('unit_cost', 'mean'),
                sub_cat=('contract_sub_category', lambda s: s.iloc[0] if len(s) > 0 else 'GENERAL MEDICAL'),
                brand=('brand', lambda s: s.value_counts().index[0] if len(s) > 0 else 'STANDARD')
            ).reset_index()

            item_stats['freq_pct'] = (item_stats['case_count'] / n_cases) * 100.0

            for _, row in item_stats.iterrows():
                freq = row['freq_pct']
                if freq >= 80.0:
                    status = "CORE_PACK"
                    recom = "Standardize in Routine Custom Pack"
                elif freq >= 20.0:
                    status = "HOLD_CART"
                    recom = "Move to Sterile Hold Cart (Open PRN)"
                else:
                    status = "SPECIALTY_PRN"
                    recom = "Physician Preference Exception Only"

                bom_records.append({
                    "procedure": str(proc).strip(),
                    "item_description": str(row['item_description']).strip(),
                    "contract_sub_category": str(row['sub_cat']).strip(),
                    "brand": str(row['brand']).strip(),
                    "case_count": int(row['case_count']),
                    "total_procedure_cases": int(n_cases),
                    "utilization_frequency_pct": round(float(freq), 1),
                    "average_unit_cost": round(float(row['avg_unit_cost']), 2),
                    "total_spend": round(float(row['total_spend']), 2),
                    "pack_status": status,
                    "standardization_action": recom
                })

            # Isolation Forest Cost/Volume Outlier detection
            if n_cases >= 10:
                X = proc_cases[['total_qty', 'total_cost']].values
                iso = IsolationForest(contamination=0.05, random_state=42)
                preds = iso.fit_predict(X)
                scores = iso.decision_function(X)
                mask = case_summary['unified_procedure'] == proc
                case_summary.loc[mask, 'cost_volume_outlier'] = np.where(preds == -1, 'Yes', 'No')
                case_summary.loc[mask, 'if_anomaly_score'] = scores

                # Compositional items
                subcat_case_counts = proc_lines.groupby('contract_sub_category')['case_id'].nunique()
                standard_subcats = set(subcat_case_counts[subcat_case_counts / n_cases >= 0.10].index)

                case_subcats = proc_lines.groupby('case_id')['contract_sub_category'].apply(set)
                for cid, c_items in case_subcats.items():
                    rogue = c_items - standard_subcats
                    if len(rogue) > 0:
                        case_outlier_map[cid] = {
                            'comp_outlier': 'Yes',
                            'rogue_items': '; '.join(list(rogue)[:4])
                        }
        except Exception as e:
            continue

    # Map composition flags back
    case_summary['composition_outlier'] = case_summary['case_id'].map(lambda x: case_outlier_map.get(x, {}).get('comp_outlier', 'No'))
    case_summary['outlier_items'] = case_summary['case_id'].map(lambda x: case_outlier_map.get(x, {}).get('rogue_items', ''))
    
    case_summary['is_master_outlier'] = np.where(
        (case_summary['cost_volume_outlier'] == 'Yes') | (case_summary['composition_outlier'] == 'Yes'),
        'Yes', 'No'
    )
    
    case_summary['anomaly_type_detail'] = np.where(
        (case_summary['cost_volume_outlier'] == 'Yes') & (case_summary['composition_outlier'] == 'Yes'),
        'Cost_Volume & Compositional Outlier',
        np.where(
            case_summary['cost_volume_outlier'] == 'Yes',
            'Cost_Volume_Outlier',
            np.where(case_summary['composition_outlier'] == 'Yes', 'Composition_Outlier', 'Normal Case')
        )
    )

    case_summary['total_case_quantity'] = case_summary['total_qty']
    print("Merging metrics back onto line items...")
    merge_cols = [
        'case_id', 'total_case_quantity', 'total_qty', 'total_cost', 'total_charges',
        'cost_volume_outlier', 'composition_outlier', 'is_master_outlier',
        'if_anomaly_score', 'anomaly_type_detail', 'outlier_items'
    ]
    df_merged = pd.merge(df, case_summary[merge_cols], on='case_id', how='left')
    df_merged['cost_per_case'] = df_merged['total_cost']
    df_merged['total_charges'] = df_merged['total_charges_y'].fillna(df_merged['total_charges_x'])
    df_merged['supply_intensity_pct'] = np.where(
        df_merged['total_charges'] > 0,
        np.round((df_merged['cost_per_case'] / df_merged['total_charges']) * 100, 2),
        0.0
    )

    bom_df = pd.DataFrame(bom_records)
    print(f"Generated {len(bom_df):,} BOM items across {len(procedures):,} procedures in {round(time.time() - t0, 1)}s")
    return df_merged, bom_df

def generate_substitution_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=======================================================")
    print(f" STEP 3: MINING MASTER & PHYSICIAN SUBSTITUTION MATRICES")
    print(f"=======================================================")
    t0 = time.time()
    
    # Filter valid lines
    valid = df.dropna(subset=['case_id', 'unified_procedure', 'contract_sub_category', 'brand']).copy()
    valid['sub_cat'] = valid['contract_sub_category'].astype(str).str.strip().str.upper()
    valid['brand_clean'] = valid['brand'].astype(str).str.strip().str.upper()
    valid['mfr'] = valid['brand_clean'].apply(lambda b: b.split('-')[0].split()[0] if b else 'UNKNOWN')

    master_records = []
    physician_records = []

    proc_groups = valid.groupby('unified_procedure')
    total_procs = len(proc_groups)
    print(f"Mining substitution construct patterns across {total_procs:,} procedures...")

    for idx, (proc, p_df) in enumerate(proc_groups):
        tot_cases = p_df['case_id'].nunique()
        if tot_cases < 3:
            continue

        # Commodity co-occurrence across encounters
        baskets = p_df.groupby('case_id')['sub_cat'].unique().tolist()
        cat_counts = p_df.groupby('sub_cat')['case_id'].nunique()
        frequent_cats = cat_counts[cat_counts / tot_cases >= 0.35].index.tolist()

        # Top co-utilized pairs
        for i in range(len(frequent_cats)):
            for j in range(i + 1, min(len(frequent_cats), i + 4)):
                cA = frequent_cats[i]
                cB = frequent_cats[j]
                pair_cnt = sum(1 for b in baskets if cA in b and cB in b)
                supp = (pair_cnt / tot_cases) * 100.0
                if supp >= 30.0:
                    master_records.append({
                        "scope": "MASTER",
                        "procedure": str(proc).strip().upper(),
                        "surgeon": "ALL SURGEONS",
                        "manufacturer": "MULTI-VENDOR",
                        "construct_items": f"{cA} + {cB}",
                        "support_pct": round(supp, 1)
                    })

        # Physician constructs
        if 'lead_surgeon' in p_df.columns:
            surg_groups = p_df.groupby('lead_surgeon')
            for surg, s_df in surg_groups:
                surg_cases = s_df['case_id'].nunique()
                if surg_cases < 3:
                    continue
                s_baskets = s_df.groupby('case_id').apply(
                    lambda g: list(set(g['sub_cat'] + ' | ' + g['brand_clean']))
                ).tolist()
                s_counts = s_df.groupby('sub_cat')['case_id'].nunique()
                s_freq = s_counts[s_counts / surg_cases >= 0.40].index.tolist()

                for i in range(len(s_freq)):
                    for j in range(i + 1, min(len(s_freq), i + 3)):
                        cA = s_freq[i]
                        cB = s_freq[j]
                        # Top brand for each
                        bA = s_df[s_df['sub_cat'] == cA]['brand_clean'].mode()
                        bB = s_df[s_df['sub_cat'] == cB]['brand_clean'].mode()
                        brandA = bA[0] if not bA.empty else 'STANDARD'
                        brandB = bB[0] if not bB.empty else 'STANDARD'
                        
                        mfrA = brandA.split('-')[0].split()[0]
                        
                        itemA = f"{cA} | {brandA}"
                        itemB = f"{cB} | {brandB}"
                        pair_cnt = sum(1 for b in s_baskets if itemA in b and itemB in b)
                        supp = (pair_cnt / surg_cases) * 100.0
                        if supp >= 35.0:
                            physician_records.append({
                                "scope": "PHYSICIAN",
                                "procedure": str(proc).strip().upper(),
                                "surgeon": str(surg).strip(),
                                "manufacturer": mfrA,
                                "construct_items": f"{itemA} + {itemB}",
                                "support_pct": round(supp, 1)
                            })

    master_df = pd.DataFrame(master_records)
    physician_df = pd.DataFrame(physician_records)

    print(f"Generated {len(master_df):,} Master construct rules and {len(physician_df):,} Physician construct rules in {round(time.time() - t0, 1)}s")
    return master_df, physician_df

def run_pipeline():
    total_start = time.time()
    print("=" * 65)
    print(f" STARTING FULL SYSTEM-WIDE REBUILD FOR ALL PROCEDURES")
    print("=" * 65)
    
    # 1. Extract all procedures
    raw_df = extract_all_procedures()
    
    # Save raw utilization dataset
    raw_path = MultiTenantConfig.raw_data_path()
    print(f"Saving full raw utilization data to {raw_path}...")
    raw_df.to_csv(raw_path, index=False)
    
    # 2. Case anomalies and BOMs
    anomaly_df, bom_df = compute_anomalies_and_boms(raw_df)
    
    anomaly_path = MultiTenantConfig.anomaly_data_path()
    bom_path = MultiTenantConfig.bom_data_path()
    print(f"Saving anomaly dataset ({len(anomaly_df):,} rows) to {anomaly_path}...")
    anomaly_df.to_csv(anomaly_path, index=False)
    
    print(f"Saving standard BOMs ({len(bom_df):,} records) to {bom_path}...")
    bom_df.to_csv(bom_path, index=False)
    
    # 3. Master & Physician substitution matrices
    master_df, physician_df = generate_substitution_matrices(anomaly_df)
    
    master_path = MultiTenantConfig.master_matrix_path()
    physician_path = MultiTenantConfig.physician_matrix_path()
    
    print(f"Saving master substitution matrix ({len(master_df):,} rows) to {master_path}...")
    master_df.to_csv(master_path, index=False)
    
    print(f"Saving physician substitution matrix ({len(physician_df):,} rows) to {physician_path}...")
    physician_df.to_csv(physician_path, index=False)
    
    print("=" * 65)
    print(f" ALL PROCEDURES DATASETS REBUILT SUCCESSFULLY in {round(time.time() - total_start, 1)}s!")
    print(f" Procedures Count: {anomaly_df['unified_procedure'].nunique():,}")
    print(f" Encounters Count: {anomaly_df['case_id'].nunique():,}")
    print(f" Total Rows:       {len(anomaly_df):,}")
    print(f" Standard BOMs:    {len(bom_df):,} items")
    print(f" Master Rules:     {len(master_df):,} rules")
    print(f" Physician Rules:  {len(physician_df):,} rules")
    print("=" * 65)

if __name__ == "__main__":
    run_pipeline()
