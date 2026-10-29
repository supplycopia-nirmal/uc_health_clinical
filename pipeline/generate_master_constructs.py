"""
Multi-Tenant FP-Growth Construct & Substitution Matrix Mining Engine.
Performs:
1. Mining of Master & Physician clinical constructs via FP-Growth with True Case Support.
2. Redundant subset pruning (Maximal itemsets).
3. Reinforcement Learning from Human Feedback (RLHF) loop evaluated by OpenAI.
4. Export to local CSV artifacts and sync to Snowflake cloud tables.
"""

import os
import sys
import json
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from openai import OpenAI
from pydantic import BaseModel, Field

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MultiTenantConfig, get_snowflake_connection

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
    except Exception as e:
        return None

# --- OPENAI RLHF VALIDATION ---
class FeedbackEvaluation(BaseModel):
    is_valid: bool = Field(description="True if the comment provides clinical rationale. False if junk text, mistake, or rejection.")
    action: str = Field(description="Action: 'Standardization', 'Deletion', 'Addition', 'Modification', or 'Reject'")

def evaluate_feedback_with_openai(action: str, subcat: str, old_brand: str, new_brand: str, comment: str):
    """Uses OpenAI to evaluate preference card edits and user comments."""
    api_key = MultiTenantConfig.OPENAI_API_KEY
    if not api_key or not comment.strip():
        return True, action

    client = OpenAI(api_key=api_key)
    prompt = f"""
You are an expert clinical supply chain reviewer evaluating a surgeon's preference card feedback.
Requested Action: {action}
Item Sub-Category: {subcat}
Original Brand: {old_brand}
New Brand: {new_brand}
Surgeon/Nurse Comment: '{comment}'

Determine if this comment provides a valid clinical or operational rationale to update the master preference card matrix.
If the comment indicates an accidental submission, nonsensical junk text, or rejects the change, return is_valid=false and action='Reject'.
Return structured JSON matching: {{"is_valid": boolean, "action": string}}
"""
    for model_name in [MultiTenantConfig.OPENAI_MODEL, MultiTenantConfig.OPENAI_FALLBACK_MODEL, "gpt-4o-mini"]:
        try:
            response = client.beta.chat.completions.parse(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=FeedbackEvaluation,
                temperature=0.0
            )
            parsed = response.choices[0].message.parsed
            return parsed.is_valid, parsed.action
        except Exception:
            continue
            
    return True, action

def apply_rlhf_to_matrices(master_df: pd.DataFrame, physician_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Applies human feedback recorded in rlhf_preference_cards.jsonl to matrices."""
    rlhf_file = MultiTenantConfig.rlhf_path()
    if not rlhf_file.exists():
        return master_df, physician_df
        
    print(f"Applying RLHF feedback from {rlhf_file}...")
    try:
        with open(rlhf_file, "r") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                proc = data.get("procedure_name")
                surg = data.get("surgeon_name")
                subs = data.get("substitutions", [])
                
                for sub in subs:
                    raw_action = sub.get("action", "Standardization")
                    subcat = str(sub.get("sub_category", "")).strip().upper()
                    orig_brand = str(sub.get("original_brand", "")).strip().upper()
                    new_brand = str(sub.get("new_brand", "")).strip().upper()
                    comment = str(sub.get("comment", "")).strip()
                    
                    is_valid, action = evaluate_feedback_with_openai(raw_action, subcat, orig_brand, new_brand, comment)
                    if not is_valid or action == 'Reject':
                        continue
                        
                    orig_str = f"{subcat} | {orig_brand}"
                    new_str = f"{subcat} | {new_brand}"
                    
                    if not physician_df.empty and 'construct_items' in physician_df.columns:
                        p_mask = (physician_df['procedure'] == proc) & (physician_df['surgeon'] == surg)
                        if action in ['Standardization', 'Modification']:
                            physician_df.loc[p_mask, 'construct_items'] = physician_df.loc[p_mask, 'construct_items'].str.replace(orig_str, new_str, regex=False)
                        elif action == 'Deletion':
                            physician_df.loc[p_mask, 'construct_items'] = physician_df.loc[p_mask, 'construct_items'].str.replace(orig_str, '', regex=False).str.replace(r'\s*\+\s*\+', ' + ', regex=True).str.strip(' +')
                        elif action == 'Addition':
                            physician_df.loc[p_mask, 'construct_items'] = physician_df.loc[p_mask, 'construct_items'].apply(lambda x: f"{x} + {new_str}" if new_str not in str(x) else x)

                    if not master_df.empty and 'construct_items' in master_df.columns:
                        m_mask = (master_df['procedure'] == proc)
                        if action in ['Standardization', 'Modification']:
                            master_df.loc[m_mask, 'construct_items'] = master_df.loc[m_mask, 'construct_items'].str.replace(orig_str, new_str, regex=False)
                        elif action == 'Deletion':
                            master_df.loc[m_mask, 'construct_items'] = master_df.loc[m_mask, 'construct_items'].str.replace(orig_str, '', regex=False).str.replace(r'\s*\+\s*\+', ' + ', regex=True).str.strip(' +')
                        elif action == 'Addition':
                            master_df.loc[m_mask, 'construct_items'] = master_df.loc[m_mask, 'construct_items'].apply(lambda x: f"{x} + {new_str}" if new_str not in str(x) else x)
    except Exception as e:
        print(f"Error applying RLHF: {e}")
        
    return master_df, physician_df

def upload_matrix_to_snowflake(df: pd.DataFrame, table_name: str):
    """Safely pushes construct matrix to Snowflake."""
    if df.empty: return
    conn = get_snowflake_connection()
    try:
        from snowflake.connector.pandas_tools import write_pandas
        df_upload = df.copy()
        df_upload.columns = df_upload.columns.str.upper().str.replace(' ', '_')
        success, nchunks, nrows, _ = write_pandas(
            conn=conn, df=df_upload, table_name=table_name,
            auto_create_table=True, overwrite=True, quote_identifiers=False
        )
        if success:
            print(f"☁️ Uploaded {nrows} rows to Snowflake table: {table_name}")
    except Exception as e:
        print(f"❌ Snowflake matrix upload failed: {e}")
    finally:
        conn.close()

def generate_matrices():
    """Extracts anomaly data, runs parallel FP-Growth mining, and produces matrices."""
    print(f"\n=======================================================")
    print(f" MINING CLINICAL SUBSTITUTION CONSTRUCTS: {MultiTenantConfig.TENANT_NAME}")
    print(f"=======================================================")
    
    anomaly_csv = MultiTenantConfig.anomaly_data_path()
    if anomaly_csv.exists():
        print(f"Loading anomaly dataset from local cache: {anomaly_csv}...")
        df = pd.read_csv(anomaly_csv)
    else:
        print("Connecting to Snowflake to fetch anomaly data...")
        conn = get_snowflake_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {MultiTenantConfig.anomaly_table()}")
        df = cursor.fetch_pandas_all()
        cursor.close()
        conn.close()
        
    df.columns = df.columns.str.lower()
    df.dropna(subset=['case_id', 'unified_procedure', 'contract_sub_category', 'brand'], inplace=True)
    df['unified_procedure'] = df['unified_procedure'].astype(str).str.strip()
    
    # Strip K-kits and CAPPED items
    exclusion_pattern = r'CAPPED|\bK\d+'
    if 'item_description' in df.columns:
        df = df[~df['item_description'].astype(str).str.contains(exclusion_pattern, case=False, regex=True)]
        
    df['clinical_item'] = df['contract_sub_category'].str.strip().str.upper() + " | " + df['brand'].str.strip().str.upper()
    df['manufacturer'] = df['brand'].apply(get_base_manufacturer)
    
    procedures = df['unified_procedure'].unique()
    print(f"Preparing mining tasks for {len(procedures)} procedures...")
    
    tasks = []
    for proc in procedures:
        proc_df = df[df['unified_procedure'] == proc]
        total_cases = proc_df['case_id'].nunique()
        
        # MASTER TASK (All cases for the procedure)
        master_baskets = proc_df.groupby(['case_id', 'manufacturer'])['clinical_item'].apply(list).tolist()
        master_baskets = [b for b in master_baskets if len(b) > 1]
        if master_baskets:
            tasks.append(('MASTER', proc, 'ALL (MASTER)', master_baskets, total_cases, 0.10))
            
        # PHYSICIAN TASKS (Cases per surgeon)
        surgeons = proc_df['lead_surgeon'].dropna().unique()
        for surg in surgeons:
            if surg == 'UNASSIGNED': continue
            surg_df = proc_df[proc_df['lead_surgeon'] == surg]
            surg_cases = surg_df['case_id'].nunique()
            if surg_cases >= 5:
                surg_baskets = surg_df.groupby(['case_id', 'manufacturer'])['clinical_item'].apply(list).tolist()
                surg_baskets = [b for b in surg_baskets if len(b) > 1]
                if surg_baskets:
                    tasks.append(('PHYSICIAN', proc, surg, surg_baskets, surg_cases, 0.15))
                    
    print(f"Total isolated mining tasks created: {len(tasks)}")
    
    num_cores = min(4, max(1, multiprocessing.cpu_count() - 1))
    print(f"Executing FP-Growth using {num_cores} parallel CPU workers...")
    
    master_results = []
    physician_results = []
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = {executor.submit(mine_constructs_worker, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % max(1, len(tasks) // 10) == 0 or completed == len(tasks):
                print(f"Progress: {completed}/{len(tasks)} tasks processed...")
            try:
                res = future.result()
                if res is not None and not res.empty:
                    scope = res['scope'].iloc[0]
                    if scope == 'MASTER': master_results.append(res)
                    else: physician_results.append(res)
            except Exception as e:
                pass

    final_master = pd.concat(master_results, ignore_index=True) if master_results else pd.DataFrame()
    final_physician = pd.concat(physician_results, ignore_index=True) if physician_results else pd.DataFrame()
    
    if not final_master.empty:
        final_master.sort_values(by=['procedure', 'support_pct'], ascending=[True, False], inplace=True)
    if not final_physician.empty:
        final_physician.sort_values(by=['procedure', 'surgeon', 'support_pct'], ascending=[True, True, False], inplace=True)
        
    # Apply RLHF
    final_master, final_physician = apply_rlhf_to_matrices(final_master, final_physician)
    
    # Save local CSVs
    if not final_master.empty:
        final_master.to_csv(MultiTenantConfig.master_matrix_path(), index=False)
        print(f"✅ Master substitution matrix saved: {MultiTenantConfig.master_matrix_path()}")
    if not final_physician.empty:
        final_physician.to_csv(MultiTenantConfig.physician_matrix_path(), index=False)
        print(f"✅ Physician substitution matrix saved: {MultiTenantConfig.physician_matrix_path()}")
        
    # Upload to Snowflake
    upload_matrix_to_snowflake(final_master, MultiTenantConfig.master_matrix_table())
    upload_matrix_to_snowflake(final_physician, MultiTenantConfig.physician_matrix_table())
    
    print("\n🎉 Matrix Mining and Cloud Sync Completed!")

if __name__ == "__main__":
    generate_matrices()
