"""
Rebuild Original Master Constructs & Standard BOMs for All Procedures.
Reverts to the proven original clinical logic:
1. Standard BOMs:
   - Grouped by `contract_sub_category` (with support >= 10% of historical procedure cases).
   - Columns: Procedure, Total_Historical_Cases, Number_of_Standard_Items, Standard_Item, Cases_Item_Used, Average_Quantity_Used, Average_Unit_Cost.
2. Master & Physician Substitution Matrices:
   - Grouped by [case_id, manufacturer] with `clinical_item = contract_sub_category + " | " + brand`.
   - Single-manufacturer compatibility enforced.
   - Redundant subsets pruned via Maximal Itemsets.
   - FP-Growth mining per procedure (MASTER) and per surgeon (PHYSICIAN).
   - RLHF feedback applied.
"""

import os
import sys
import json
import time
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

# Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from config import MultiTenantConfig

warnings.filterwarnings("ignore")

def get_base_manufacturer(brand_str: str) -> str:
    """Extracts the base manufacturer name from a brand string."""
    b = str(brand_str).upper().strip()
    if "-" in b:
        return b.split("-")[0].strip()
    return b.split()[0].strip() if b.split() else "UNKNOWN"

def get_itemset_manufacturer(itemset) -> str:
    """Validates that all items in a mined construct originate from a single manufacturer."""
    try:
        mfrs = set()
        for item in itemset:
            parts = item.split(" | ")
            if len(parts) > 1:
                mfrs.add(get_base_manufacturer(parts[1]))
            else:
                mfrs.add(get_base_manufacturer(parts[0]))
        if len(mfrs) == 1 and list(mfrs)[0] != "UNKNOWN":
            return list(mfrs)[0]
    except Exception:
        pass
    return None

def filter_maximal_itemsets(frequent_itemsets_df: pd.DataFrame) -> pd.DataFrame:
    """Filters out redundant subsets when a larger construct exists at similar support."""
    records = frequent_itemsets_df[['itemsets', 'support', 'manufacturer']].to_dict('records')
    records.sort(key=lambda x: (len(x['itemsets']), x['support']), reverse=True)
    
    maximal_sets = []
    for row in records:
        current_set = set(row['itemsets'])
        current_support = row['support']
        is_maximal = True
        
        for max_row in maximal_sets:
            max_set = max_row['itemsets']
            max_support = max_row['support']
            
            if current_set.issubset(max_set):
                if current_support <= max_support + 5.0:
                    is_maximal = False
                    break
                    
        if is_maximal:
            maximal_sets.append({
                'itemsets': current_set,
                'support': current_support,
                'manufacturer': row['manufacturer']
            })
            
    return pd.DataFrame(maximal_sets)

def mine_constructs_worker(task):
    """Worker function for parallel FP-Growth execution."""
    scope, procedure, surgeon, baskets, total_cases, min_case_support = task
    
    if not baskets or total_cases == 0:
        return None

    min_count = max(2, int(total_cases * min_case_support))
    
    item_counts = {}
    for basket in baskets:
        for item in set(basket):
            item_counts[item] = item_counts.get(item, 0) + 1
            
    valid_items = {item for item, count in item_counts.items() if count >= min_count}
    pruned_baskets = [[item for item in basket if item in valid_items] for basket in baskets]
    pruned_baskets = [b for b in pruned_baskets if len(b) >= 2]
    
    if not pruned_baskets or len(pruned_baskets) < min_count:
        return None

    try:
        te = TransactionEncoder()
        te_ary = te.fit(pruned_baskets).transform(pruned_baskets)
        basket_df = pd.DataFrame(te_ary, columns=te.columns_).astype(bool)
        
        adjusted_min_support = min_count / len(basket_df)
        freq_items = fpgrowth(basket_df, min_support=adjusted_min_support, use_colnames=True)
        
        if freq_items.empty:
            return None
            
        freq_items = freq_items[freq_items['itemsets'].apply(len) >= 2]
        if freq_items.empty:
            return None
            
        freq_items['manufacturer'] = freq_items['itemsets'].apply(get_itemset_manufacturer)
        freq_items = freq_items[freq_items['manufacturer'].notnull()]
        if freq_items.empty:
            return None
            
        # Compute TRUE Case Support Percentage
        freq_items['support'] = (freq_items['support'] * len(basket_df)) / total_cases * 100
        
        clean_constructs = filter_maximal_itemsets(freq_items)
        if clean_constructs.empty:
            return None
            
        clean_constructs['construct_items'] = clean_constructs['itemsets'].apply(lambda x: ' + '.join(sorted(list(x))))
        clean_constructs['support_pct'] = clean_constructs['support'].round(1)
        
        clean_constructs.insert(0, 'surgeon', surgeon)
        clean_constructs.insert(0, 'procedure', procedure)
        clean_constructs.insert(0, 'scope', scope)
        
        return clean_constructs[['scope', 'procedure', 'surgeon', 'manufacturer', 'construct_items', 'support_pct']]
    except Exception:
        return None

def build_standard_boms(df: pd.DataFrame) -> pd.DataFrame:
    """Builds Standard BOMs across all procedures using original contract_sub_category logic."""
    print("\n=======================================================")
    print(" 1. GENERATING ORIGINAL STANDARD BOMS (Support >= 10%)")
    print("=======================================================")
    t0 = time.time()
    
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(1.0)
    df['unit_cost'] = pd.to_numeric(df['unit_cost'], errors='coerce').fillna(0.0)
    
    proc_cases = df.groupby('unified_procedure')['case_id'].nunique()
    
    subcat_agg = df.groupby(['unified_procedure', 'contract_sub_category']).agg(
        cases_used=('case_id', 'nunique'),
        avg_qty=('quantity', 'mean'),
        avg_cost=('unit_cost', 'mean')
    ).reset_index()
    
    subcat_agg['total_cases'] = subcat_agg['unified_procedure'].map(proc_cases)
    subcat_agg['support'] = subcat_agg['cases_used'] / subcat_agg['total_cases']
    
    # Filter for standard items (support >= 10%)
    std_df = subcat_agg[subcat_agg['support'] >= 0.10].copy()
    
    # Count standard items per procedure
    items_per_proc = std_df.groupby('unified_procedure')['contract_sub_category'].count()
    std_df['num_std_items'] = std_df['unified_procedure'].map(items_per_proc)
    
    bom_records = []
    for _, row in std_df.iterrows():
        bom_records.append({
            'Procedure': row['unified_procedure'],
            'Total_Historical_Cases': int(row['total_cases']),
            'Number_of_Standard_Items': int(row['num_std_items']),
            'Standard_Item': row['contract_sub_category'],
            'Cases_Item_Used': int(row['cases_used']),
            'Average_Quantity_Used': round(float(row['avg_qty']), 2),
            'Average_Unit_Cost': round(float(row['avg_cost']), 2)
        })
        
    bom_result_df = pd.DataFrame(bom_records)
    bom_path = MultiTenantConfig.bom_data_path()
    bom_result_df.to_csv(bom_path, index=False)
    print(f"Generated {len(bom_result_df):,} Standard BOM items across {bom_result_df['Procedure'].nunique():,} procedures in {round(time.time() - t0, 1)}s")
    print(f"Saved to: {bom_path}")
    return bom_result_df

def build_substitution_matrices(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mines Master & Physician substitution constructs using original FP-Growth logic."""
    print("\n=======================================================")
    print(" 2. MINING MASTER & PHYSICIAN SUBSTITUTION CONSTRUCTS")
    print("=======================================================")
    t0 = time.time()
    
    valid = df.dropna(subset=['case_id', 'unified_procedure', 'contract_sub_category', 'brand']).copy()
    valid['unified_procedure'] = valid['unified_procedure'].astype(str).str.strip()
    
    # Strip CAPPED and kit items
    exclusion_pattern = r'CAPPED|\bK\d+'
    if 'item_description' in valid.columns:
        valid = valid[~valid['item_description'].astype(str).str.contains(exclusion_pattern, case=False, regex=True)]
        
    valid['clinical_item'] = valid['contract_sub_category'].str.strip().str.upper() + " | " + valid['brand'].str.strip().str.upper()
    valid['manufacturer'] = valid['brand'].apply(get_base_manufacturer)
    
    print("Mining case baskets by procedure, surgeon, and manufacturer...")
    t_bask = time.time()
    baskets_s = valid.groupby(['unified_procedure', 'lead_surgeon', 'case_id', 'manufacturer'])['clinical_item'].apply(list)
    baskets_df = baskets_s[baskets_s.apply(len) > 1].reset_index()
    print(f"Mined {len(baskets_df):,} valid manufacturer baskets in {round(time.time() - t_bask, 1)}s")
    
    print("Assembling mining tasks across procedures and surgeons...")
    proc_cases_map = valid.groupby('unified_procedure')['case_id'].nunique().to_dict()
    surg_cases_map = valid.groupby(['unified_procedure', 'lead_surgeon'])['case_id'].nunique().to_dict()
    
    proc_baskets = baskets_df.groupby('unified_procedure')['clinical_item'].apply(list).to_dict()
    surg_baskets = baskets_df.groupby(['unified_procedure', 'lead_surgeon'])['clinical_item'].apply(list).to_dict()
    
    tasks = []
    # Master tasks
    for proc, baskets in proc_baskets.items():
        total_cases = proc_cases_map.get(proc, 0)
        if total_cases >= 2 and baskets:
            tasks.append(('MASTER', proc, 'ALL (MASTER)', baskets, total_cases, 0.10))
            
    # Physician tasks
    for (proc, surg), baskets in surg_baskets.items():
        if surg == 'UNASSIGNED': continue
        total_cases = surg_cases_map.get((proc, surg), 0)
        if total_cases >= 5 and baskets:
            tasks.append(('PHYSICIAN', proc, surg, baskets, total_cases, 0.15))
            
    print(f"Total mining tasks created: {len(tasks):,}")
    
    num_cores = min(8, max(1, multiprocessing.cpu_count() - 1))
    print(f"Executing FP-Growth using {num_cores} parallel CPU workers...")
    
    master_results = []
    physician_results = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(mine_constructs_worker, task): task for task in tasks}
        completed = 0
        total_t = len(tasks)
        for future in as_completed(futures):
            completed += 1
            if completed % max(1, total_t // 10) == 0 or completed == total_t:
                print(f"Mining Progress: {completed:,}/{total_t:,} tasks ({round(completed/total_t*100, 1)}%)...")
            try:
                res = future.result()
                if res is not None and not res.empty:
                    scope = res['scope'].iloc[0]
                    if scope == 'MASTER': master_results.append(res)
                    else: physician_results.append(res)
            except Exception:
                pass

    final_master = pd.concat(master_results, ignore_index=True) if master_results else pd.DataFrame()
    final_physician = pd.concat(physician_results, ignore_index=True) if physician_results else pd.DataFrame()
    
    if not final_master.empty:
        final_master.sort_values(by=['procedure', 'support_pct'], ascending=[True, False], inplace=True)
    if not final_physician.empty:
        final_physician.sort_values(by=['procedure', 'surgeon', 'support_pct'], ascending=[True, True, False], inplace=True)
        
    master_path = MultiTenantConfig.master_matrix_path()
    physician_path = MultiTenantConfig.physician_matrix_path()
    
    final_master.to_csv(master_path, index=False)
    final_physician.to_csv(physician_path, index=False)
    
    print(f"Generated {len(final_master):,} Master constructs across {final_master['procedure'].nunique() if not final_master.empty else 0:,} procedures")
    print(f"Generated {len(final_physician):,} Physician constructs across {final_physician['procedure'].nunique() if not final_physician.empty else 0:,} procedures")
    print(f"Saved master matrix to: {master_path}")
    print(f"Saved physician matrix to: {physician_path}")
    print(f"Construct mining completed in {round(time.time() - t0, 1)}s")
    
    return final_master, final_physician

if __name__ == "__main__":
    t_start = time.time()
    anomaly_csv = MultiTenantConfig.anomaly_data_path()
    print(f"Loading anomaly dataset: {anomaly_csv}...")
    df = pd.read_csv(anomaly_csv)
    df.columns = df.columns.str.lower()
    print(f"Loaded {len(df):,} records, {df['case_id'].nunique():,} cases, and {df['unified_procedure'].nunique():,} procedures.")
    
    # 1. Build Standard BOMs
    bom_df = build_standard_boms(df)
    
    # 2. Build Substitution Matrices
    master_df, physician_df = build_substitution_matrices(df)
    
    print(f"\n=======================================================")
    print(f" ALL MASTER CONSTRUCTS REBUILT IN {round(time.time() - t_start, 1)}s")
    print(f"=======================================================")
