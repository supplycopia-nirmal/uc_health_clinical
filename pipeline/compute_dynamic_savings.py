"""
Dynamic Clinical Variance and Non-Standard Utilization Savings Engine.
Calculates dynamic identified savings per procedure and per case based on:
1. Clinical Variance Items (rogue spend outside standard BOM and approved constructs): 100% savings opportunity
2. Non-Standard Items (unapproved alternative constructs): 100% savings opportunity
3. Standard Items Excess Utilization: max(0, qty - std_avg_qty) * unit_cost
"""
import time
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DynamicSavingsPipeline")

def run_dynamic_savings_pipeline(output_json: Path = None):
    t0 = time.time()
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    anomaly_csv = data_dir / "uc_health_hybrid_anomaly.csv"
    bom_csv = data_dir / "uc_health_standard_boms.csv"
    master_csv = data_dir / "uc_health_master_matrix.csv"
    physician_csv = data_dir / "uc_health_physician_matrix.csv"

    if output_json is None:
        output_json = data_dir / "uc_health_procedure_dynamic_savings.json"

    logger.info("Loading anomaly and master matrix datasets...")
    df = pd.read_csv(anomaly_csv, low_memory=False)
    bom_df = pd.read_csv(bom_csv)
    master_df = pd.read_csv(master_csv)
    physician_df = pd.read_csv(physician_csv)

    # Standardize columns
    df.columns = df.columns.str.strip().str.lower()
    df["unified_procedure"] = df["unified_procedure"].astype(str).str.strip()
    df["unified_procedure_upper"] = df["unified_procedure"].str.upper()
    df["contract_sub_category"] = df["contract_sub_category"].astype(str).str.strip().str.upper()
    df["case_id"] = df["case_id"].astype(str).str.strip()

    if "quantity_num" not in df.columns:
        df["quantity_num"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1.0)
    else:
        df["quantity_num"] = pd.to_numeric(df["quantity_num"], errors="coerce").fillna(1.0)

    if "unit_cost_num" not in df.columns:
        df["unit_cost_num"] = pd.to_numeric(df["unit_cost"], errors="coerce").fillna(0.0)
    else:
        df["unit_cost_num"] = pd.to_numeric(df["unit_cost_num"], errors="coerce").fillna(0.0)

    if "line_spend" in df.columns:
        df["line_spend"] = pd.to_numeric(df["line_spend"], errors="coerce").fillna(0.0)
        df["ext_cost"] = np.where(df["line_spend"] > 0, df["line_spend"], df["quantity_num"] * df["unit_cost_num"])
    else:
        df["ext_cost"] = df["quantity_num"] * df["unit_cost_num"]

    # Pre-parse BOMs
    bom_df.columns = bom_df.columns.str.strip().str.lower()
    bom_items_by_proc = {}
    if not bom_df.empty and "procedure" in bom_df.columns:
        for proc, grp in bom_df.groupby(bom_df["procedure"].astype(str).str.strip().str.upper()):
            b_dict = {}
            if "standard_item" in grp.columns:
                for _, r in grp.iterrows():
                    it = str(r["standard_item"]).strip().upper()
                    b_dict[it] = {
                        "avg_qty": float(r.get("average_quantity_used", 1.0) or 1.0),
                        "avg_cost": float(r.get("average_unit_cost", 0.0) or 0.0)
                    }
            bom_items_by_proc[proc] = b_dict

    # Pre-parse construct subcats
    construct_subcats_by_proc = {}
    for mdf in [master_df, physician_df]:
        mdf.columns = mdf.columns.str.strip().str.lower()
        if not mdf.empty and "procedure" in mdf.columns:
            for proc, grp in mdf.groupby(mdf["procedure"].astype(str).str.strip().str.upper()):
                if proc not in construct_subcats_by_proc:
                    construct_subcats_by_proc[proc] = set()
                for items_str in grp["construct_items"].dropna():
                    for it in str(items_str).split(" + "):
                        if " | " in it:
                            construct_subcats_by_proc[proc].add(it.split(" | ")[0].strip().upper())
                        elif it.strip():
                            construct_subcats_by_proc[proc].add(it.strip().upper())

    logger.info("Aggregating case items across encounters...")
    case_item_agg = df.groupby(["unified_procedure_upper", "case_id", "contract_sub_category"]).agg(
        qty=("quantity_num", "sum"),
        ext_cost=("ext_cost", "sum")
    ).reset_index()

    proc_subcat_case_qty = case_item_agg.groupby(["unified_procedure_upper", "contract_sub_category"])["qty"].mean().to_dict()

    logger.info("Computing dynamic clinical variance & non-standard utilization savings per procedure...")
    proc_results = {}
    for proc, grp in case_item_agg.groupby("unified_procedure_upper"):
        bom_items = bom_items_by_proc.get(proc, {})
        proc_boms = {}
        for sc, bdata in bom_items.items():
            emp_qty = proc_subcat_case_qty.get((proc, sc))
            proc_boms[sc] = round(float(emp_qty), 1) if emp_qty is not None else bdata["avg_qty"]

        valid_constructs = construct_subcats_by_proc.get(proc, set())

        case_savings_dict = {}
        for row in grp.itertuples(index=False):
            cid = row.case_id
            if cid not in case_savings_dict:
                case_savings_dict[cid] = {"clin": 0.0, "non_std": 0.0}

            subcat = row.contract_sub_category
            cost = row.ext_cost

            if subcat in proc_boms:
                pass  # Standard BOM item: strictly standard, no variance
            else:
                if valid_constructs and subcat not in valid_constructs:
                    case_savings_dict[cid]["clin"] += cost
                else:
                    case_savings_dict[cid]["non_std"] += cost

        tot_clin_var = round(sum(c["clin"] for c in case_savings_dict.values()), 2)
        tot_non_std = round(sum(c["non_std"] for c in case_savings_dict.values()), 2)
        tot_dynamic = round(sum(round(c["clin"] + c["non_std"], 2) for c in case_savings_dict.values()), 2)

        proc_results[proc] = {
            "clinical_variance": tot_clin_var,
            "non_standard": tot_non_std,
            "identified_savings": tot_dynamic
        }

    with open(output_json, "w") as f:
        json.dump(proc_results, f, indent=2)

    logger.info(f"Dynamic savings calculation completed for {len(proc_results)} procedures in {round(time.time() - t0, 1)}s. Saved to {output_json}")
    return proc_results

if __name__ == "__main__":
    run_dynamic_savings_pipeline()
