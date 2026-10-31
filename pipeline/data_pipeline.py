"""
Multi-Tenant Clinical Data Pipeline for UC Health and Future Clients.
Performs:
1. Extraction of utilization data from Snowflake SC_MULTI_TENANT_CONSUMPTION_SAVINGS.
2. Case-level aggregation of Cost and Volume.
3. Machine Learning Anomaly Detection (Isolation Forest) & Compositional Rogue Item Detection.
4. Standard Bill of Materials (BOM) Report generation.
5. Cloud sync to Snowflake tenant tables.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

# Add parent directory to sys.path for config import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MultiTenantConfig, get_snowflake_connection

def extract_from_snowflake(limit_top_procs: int = 100) -> pd.DataFrame:
    """Extracts client utilization data from Snowflake multi-tenant table."""
    tenant_id = MultiTenantConfig.TENANT_ID
    tenant_name = MultiTenantConfig.TENANT_NAME
    raw_csv_path = MultiTenantConfig.raw_data_path()
    
    print(f"\n=======================================================")
    print(f" STEP 1: EXTRACTING CONSUMPTION FOR {tenant_name} ({tenant_id})")
    print(f"=======================================================")
    
    conn = get_snowflake_connection()
    try:
        query = f"""
        WITH BaseData AS (
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
            WHERE CLIENT_ID = '{tenant_id}'
              AND CASE_ID IS NOT NULL
              AND STANDARDIZED_PROCEDURE IS NOT NULL
        ),
        TopProcedures AS (
            SELECT "unified_procedure"
            FROM BaseData
            GROUP BY "unified_procedure"
            ORDER BY COUNT(DISTINCT "case_id") DESC
            LIMIT {limit_top_procs}
        )
        SELECT b.*
        FROM BaseData b
        INNER JOIN TopProcedures t ON b."unified_procedure" = t."unified_procedure";
        """
        
        print("Executing Snowflake extraction query...")
        cursor = conn.cursor()
        cursor.execute(query)
        df = cursor.fetch_pandas_all()
        cursor.close()
        
        # Clean column names
        df.columns = df.columns.str.lower()
        print(f"Extracted {len(df)} rows across {df['case_id'].nunique()} cases and {df['unified_procedure'].nunique()} procedures.")
        
        # Save raw data cache locally
        df.to_csv(raw_csv_path, index=False)
        print(f"Saved raw data to {raw_csv_path}")
        return df
        
    finally:
        conn.close()


def detect_anomalies_and_build_boms(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs case-level Isolation Forest anomaly detection,
    compositional rogue item detection, and computes Standard BOMs.
    """
    print(f"\n=======================================================")
    print(f" STEP 2: ML ANOMALY DETECTION & STANDARD BOM GENERATION ")
    print(f"=======================================================")
    
    # Numeric sanitization
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(1.0)
    df['unit_cost'] = pd.to_numeric(df['unit_cost'], errors='coerce').fillna(0.0)
    df['line_spend'] = pd.to_numeric(df['line_spend'], errors='coerce').fillna(0.0)
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce').fillna(0.0)
    else:
        df['total_charges'] = 0.0
    
    # Calculate case-level total quantity, total supply cost, and deduplicated total procedure charge
    print("Aggregating case-level metrics...")
    agg_kwargs = {
        'Total_Case_Quantity': ('quantity', 'sum'),
        'Total_Case_Cost': ('line_spend', 'sum'),
        'Total_Procedure_Charges': ('total_charges', 'first')
    }
    case_metrics = df.groupby(['unified_procedure', 'case_id']).agg(**agg_kwargs).reset_index()
    
    case_metrics['Cost_Volume_Outlier'] = 'No'
    case_metrics['Composition_Outlier'] = 'No'
    case_metrics['IF_Anomaly_Score'] = 0.0
    case_metrics['Anomaly_Type_Detail'] = 'Normal Case'
    case_metrics['Outlier_Items'] = ''
    
    procedures = case_metrics['unified_procedure'].unique()
    print(f"Running Isolation Forest and Compositional Outlier models on {len(procedures)} procedures...")
    
    bom_records = []
    case_outlier_map = {}
    
    for proc in procedures:
        proc_case_mask = case_metrics['unified_procedure'] == proc
        proc_case_df = case_metrics[proc_case_mask]
        total_proc_cases = len(proc_case_df)
        
        proc_lines = df[df['unified_procedure'] == proc]
        
        # --- A. STANDARD BOM GENERATION (Support >= 10%) ---
        item_case_counts = proc_lines.groupby('contract_sub_category')['case_id'].nunique()
        item_support = item_case_counts / total_proc_cases if total_proc_cases > 0 else 0
        
        standard_items_series = item_support[item_support >= 0.10]
        standard_items = set(standard_items_series.index)
        
        for item, supp in standard_items_series.items():
            item_data = proc_lines[proc_lines['contract_sub_category'] == item]
            avg_qty = round(float(item_data['quantity'].mean()), 2)
            avg_cost = round(float(item_data['unit_cost'].mean()), 2)
            cases_used = int(item_case_counts.get(item, 0))
            
            bom_records.append({
                'Procedure': proc,
                'Total_Historical_Cases': total_proc_cases,
                'Number_of_Standard_Items': len(standard_items),
                'Standard_Item': item,
                'Cases_Item_Used': cases_used,
                'Average_Quantity_Used': avg_qty,
                'Average_Unit_Cost': avg_cost
            })
            
        # --- B. ISOLATION FOREST COST & VOLUME ANOMALIES ---
        if total_proc_cases >= 10:
            X = proc_case_df[['Total_Case_Quantity', 'Total_Case_Cost']].values
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(X)
            scores = iso.decision_function(X)
            
            case_metrics.loc[proc_case_mask, 'Cost_Volume_Outlier'] = np.where(preds == -1, 'Yes', 'No')
            case_metrics.loc[proc_case_mask, 'IF_Anomaly_Score'] = scores
        else:
            case_metrics.loc[proc_case_mask, 'Cost_Volume_Outlier'] = 'Not Enough Data'
            
        # --- C. COMPOSITIONAL ROGUE ITEM ANOMALIES ---
        if total_proc_cases >= 10:
            case_items_grouped = proc_lines.groupby('case_id')['contract_sub_category'].apply(set)
            for c_id, items_set in case_items_grouped.items():
                rogue = items_set - standard_items
                has_rogue = len(rogue) > 0
                case_outlier_map[c_id] = {
                    'comp_outlier': 'Yes' if has_rogue else 'No',
                    'rogue_items': '; '.join(list(rogue)[:5])
                }
        else:
            for c_id in proc_case_df['case_id']:
                case_outlier_map[c_id] = {'comp_outlier': 'Not Enough Data', 'rogue_items': ''}

    # Map composition flags back
    case_metrics['Composition_Outlier'] = case_metrics['case_id'].map(lambda x: case_outlier_map.get(x, {}).get('comp_outlier', 'No'))
    case_metrics['Outlier_Items'] = case_metrics['case_id'].map(lambda x: case_outlier_map.get(x, {}).get('rogue_items', ''))
    
    # Master outlier flag
    case_metrics['Is_Master_Outlier'] = np.where(
        (case_metrics['Cost_Volume_Outlier'] == 'Yes') | (case_metrics['Composition_Outlier'] == 'Yes'),
        'Yes', 'No'
    )
    
    # Anomaly detail description
    def get_anomaly_detail(row):
        cv = row['Cost_Volume_Outlier']
        comp = row['Composition_Outlier']
        if cv == 'Yes' and comp == 'Yes': return 'Cost_Volume & Compositional Outlier'
        if cv == 'Yes': return 'Cost_Volume_Outlier'
        if comp == 'Yes': return 'Composition_Outlier'
        return 'Normal Case'
        
    case_metrics['Anomaly_Type_Detail'] = case_metrics.apply(get_anomaly_detail, axis=1)
    
    # Merge case level metrics back onto full line-item dataset
    print("Merging anomaly metrics to line-item dataset...")
    case_cols_to_merge = [
        'case_id', 'Total_Case_Quantity', 'Total_Case_Cost', 'Total_Procedure_Charges',
        'Cost_Volume_Outlier', 'Composition_Outlier', 'Is_Master_Outlier',
        'IF_Anomaly_Score', 'Anomaly_Type_Detail', 'Outlier_Items'
    ]
    df_merged = pd.merge(
        df,
        case_metrics[[c for c in case_cols_to_merge if c in case_metrics.columns]],
        on='case_id',
        how='left'
    )
    
    # Map cost_per_case and total_charges
    df_merged['cost_per_case'] = df_merged['Total_Case_Cost']
    df_merged['total_charges'] = df_merged.get('Total_Procedure_Charges', 0.0)
    df_merged['supply_intensity_pct'] = np.where(
        df_merged['total_charges'] > 0,
        np.round((df_merged['cost_per_case'] / df_merged['total_charges']) * 100, 2),
        0.0
    )
    
    # Convert BOM records to DataFrame
    bom_df = pd.DataFrame(bom_records)
    
    # Save local CSV artifacts
    anomaly_csv_path = MultiTenantConfig.anomaly_data_path()
    bom_csv_path = MultiTenantConfig.bom_data_path()
    
    df_merged.to_csv(anomaly_csv_path, index=False)
    bom_df.to_csv(bom_csv_path, index=False)
    
    print(f"✅ Anomaly detection complete: {len(df_merged)} rows saved to {anomaly_csv_path}")
    print(f"✅ Standard BOMs generated: {len(bom_df)} rows saved to {bom_csv_path}")
    
    return df_merged, bom_df


def push_dataset_to_snowflake(df: pd.DataFrame, table_name: str):
    """Pushes a pandas DataFrame to Snowflake, creating or replacing the table."""
    print(f"\nPushing {len(df)} rows to Snowflake table: {table_name}...")
    conn = get_snowflake_connection()
    try:
        df_upload = df.copy()
        df_upload.columns = df_upload.columns.str.upper().str.replace(' ', '_')
        
        success, nchunks, nrows, _ = write_pandas(
            conn=conn,
            df=df_upload,
            table_name=table_name,
            auto_create_table=True,
            overwrite=True,
            quote_identifiers=False
        )
        if success:
            print(f"✅ Successfully inserted {nrows} rows into Snowflake table {table_name}.")
        else:
            print(f"❌ Failed to insert data into {table_name}.")
    except Exception as e:
        print(f"❌ Error uploading to Snowflake: {e}")
    finally:
        conn.close()


def run_pipeline():
    """Main orchestration function for the UC Health data pipeline."""
    print(f"\n=======================================================")
    print(f" STARTING MULTI-TENANT PIPELINE FOR {MultiTenantConfig.TENANT_NAME}")
    print(f"=======================================================")
    
    # 1. Extract from Snowflake
    df_raw = extract_from_snowflake(limit_top_procs=100)
    
    # 2. Run Anomaly Detection and Generate BOMs
    df_anomaly, df_bom = detect_anomalies_and_build_boms(df_raw)
    
    # 3. Push to Snowflake tenant tables
    push_dataset_to_snowflake(df_anomaly, MultiTenantConfig.anomaly_table())
    push_dataset_to_snowflake(df_bom, MultiTenantConfig.bom_table())
    
    print("\n🎉 UC Health Data Pipeline Completed Successfully!")


if __name__ == "__main__":
    run_pipeline()
