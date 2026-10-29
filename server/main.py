"""
Multi-Tenant Clinical Intelligence Backend for UC Health.
Provides full analytical API for clinical variation, standard BOMs,
preference cards, price parity, savings leakage, and predictive modeling.
"""

import os
import sys
import json
import secrets
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from functools import lru_cache

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, status, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse, ORJSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field
from openai import OpenAI
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from config import MultiTenantConfig, get_snowflake_connection

# Global State
df = pd.DataFrame()
bom_df = pd.DataFrame()
master_constructs_df = pd.DataFrame()
physician_constructs_df = pd.DataFrame()
trained_models: Dict[str, Any] = {}

openai_client = OpenAI(api_key=MultiTenantConfig.OPENAI_API_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes clinical datasets from local artifacts or Snowflake."""
    global df, bom_df, master_constructs_df, physician_constructs_df
    print(f"\n[Lifespan] Initializing Clinical Backend for {MultiTenantConfig.TENANT_NAME}...")
    
    anomaly_csv = MultiTenantConfig.anomaly_data_path()
    bom_csv = MultiTenantConfig.bom_data_path()
    master_csv = MultiTenantConfig.master_matrix_path()
    physician_csv = MultiTenantConfig.physician_matrix_path()
    
    # 1. Load Anomaly Dataset
    if anomaly_csv.exists() and anomaly_csv.stat().st_size > 1000:
        print(f"Loading anomaly dataset from local artifact: {anomaly_csv}")
        df = pd.read_csv(anomaly_csv, low_memory=False)
    else:
        try:
            print(f"Fetching {MultiTenantConfig.anomaly_table()} from Snowflake...")
            conn = get_snowflake_connection()
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {MultiTenantConfig.anomaly_table()}")
            df = cur.fetch_pandas_all()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Notice: Snowflake anomaly table not ready: {e}")
            df = pd.DataFrame()

    if not df.empty:
        df.columns = df.columns.str.strip().str.lower()
        df['unified_procedure'] = df.get('unified_procedure', '').astype(str).str.strip()
        df['contract_sub_category'] = df.get('contract_sub_category', '').astype(str).str.strip().str.upper()
        df['brand'] = df.get('brand', '').astype(str).str.strip().str.upper()
        df['case_id'] = df.get('case_id', '').astype(str).str.strip()
        df['spend_category'] = df.get('spend_category', 'SUPPLIES').astype(str).str.strip().str.upper()
        df['item_description'] = df.get('item_description', '').astype(str).str.strip().str.upper()
        
        df['quantity_num'] = pd.to_numeric(df.get('quantity', 1.0), errors='coerce').fillna(1.0)
        df['unit_cost_num'] = pd.to_numeric(df.get('unit_cost', 0.0), errors='coerce').fillna(0.0)
        df['line_spend_num'] = pd.to_numeric(df.get('line_spend', 0.0), errors='coerce').fillna(0.0)
        df['cost_per_case_num'] = pd.to_numeric(df.get('cost_per_case', 0.0), errors='coerce').fillna(0.0)
        df['ext_cost'] = np.where(df['line_spend_num'] > 0, df['line_spend_num'], df['quantity_num'] * df['unit_cost_num'])

    # 2. Load Standard BOMs
    if bom_csv.exists() and bom_csv.stat().st_size > 100:
        print(f"Loading standard BOMs from local artifact: {bom_csv}")
        bom_df = pd.read_csv(bom_csv)
    else:
        try:
            print(f"Fetching {MultiTenantConfig.bom_table()} from Snowflake...")
            conn = get_snowflake_connection()
            cur = conn.cursor()
            cur.execute(f"SELECT * FROM {MultiTenantConfig.bom_table()}")
            bom_df = cur.fetch_pandas_all()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"Notice: Snowflake BOM table not ready: {e}")
            bom_df = pd.DataFrame()

    if not bom_df.empty:
        bom_df.columns = bom_df.columns.str.strip().str.lower()
        if 'procedure' in bom_df.columns:
            bom_df['procedure'] = bom_df['procedure'].astype(str).str.strip()

    # 3. Load Master & Physician Constructs
    if master_csv.exists() and master_csv.stat().st_size > 100:
        master_constructs_df = pd.read_csv(master_csv)
        master_constructs_df.columns = master_constructs_df.columns.str.strip().str.lower()
    
    if physician_csv.exists() and physician_csv.stat().st_size > 100:
        physician_constructs_df = pd.read_csv(physician_csv)
        physician_constructs_df.columns = physician_constructs_df.columns.str.strip().str.lower()

    print(f"✅ Backend ready for {MultiTenantConfig.TENANT_NAME} ({len(df)} rows, {len(bom_df)} BOM items)")
    yield
    
    df = pd.DataFrame()
    bom_df = pd.DataFrame()
    master_constructs_df = pd.DataFrame()
    physician_constructs_df = pd.DataFrame()
    trained_models.clear()

app = FastAPI(
    title=f"{MultiTenantConfig.TENANT_NAME} Clinical Intelligence Platform",
    lifespan=lifespan,
    default_response_class=ORJSONResponse
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

security = HTTPBasic()
def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_user = secrets.compare_digest(credentials.username, os.getenv("APP_USERNAME", "uchealth_admin"))
    correct_pass = secrets.compare_digest(credentials.password, os.getenv("APP_PASSWORD", "clinical_copia"))
    if not (correct_user and correct_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"}
        )
    return credentials.username

# --- FRONTEND ROUTING ---
WEB_DIR = BASE_DIR / "web"

@app.get("/")
def serve_index(user: str = Depends(authenticate)):
    return FileResponse(WEB_DIR / "index.html")

@app.get("/static/{file_path:path}")
def serve_static(file_path: str):
    target = WEB_DIR / "static" / file_path
    if target.exists():
        return FileResponse(target)
    raise HTTPException(status_code=404, detail="Static file not found")

@app.get("/api/tenant-info")
def get_tenant_info():
    return {
        "tenant_id": MultiTenantConfig.TENANT_ID,
        "tenant_name": MultiTenantConfig.TENANT_NAME,
        "total_cases": int(df['case_id'].nunique()) if not df.empty else 0,
        "total_spend": round(float(df['ext_cost'].sum()), 2) if not df.empty else 0.0,
        "facilities": sorted(df['hospital_name'].dropna().unique().tolist()) if not df.empty and 'hospital_name' in df.columns else []
    }

# --- PROCEDURES & CATALOG ---
@lru_cache(maxsize=1)
def _get_procedures_cached():
    if df.empty or 'unified_procedure' not in df.columns: return []
    return sorted(df['unified_procedure'].dropna().unique().tolist())

@app.get("/api/procedures")
def get_procedures(user: str = Depends(authenticate)):
    return _get_procedures_cached()

@lru_cache(maxsize=1)
def _get_catalog_cached():
    if df.empty or 'contract_sub_category' not in df.columns or 'brand' not in df.columns:
        return {"catalog": {}}
    valid = df.dropna(subset=['contract_sub_category', 'brand'])
    valid = valid[(valid['contract_sub_category'] != 'NAN') & (valid['brand'] != 'NAN')]
    cat_map = {}
    for subcat, grp in valid.groupby('contract_sub_category'):
        cat_map[subcat] = sorted(list(set([str(b).strip().upper() for b in grp['brand']])))
    return {"catalog": cat_map}

@app.get("/api/catalog")
def get_catalog():
    return _get_catalog_cached()

# --- PROCEDURE LEVEL VARIANCE & CASE DETAILS ---
@lru_cache(maxsize=100)
def _get_procedure_data_cached(procedure_name: str):
    if df.empty or 'unified_procedure' not in df.columns or 'case_id' not in df.columns:
        return {}

    mask = df['unified_procedure'] == procedure_name.strip()
    proc_df = df[mask].copy().replace({np.nan: None})
    if proc_df.empty: return {}

    # BOM standard items
    bom_items = {}
    if not bom_df.empty and 'procedure' in bom_df.columns and 'standard_item' in bom_df.columns:
        p_bom = bom_df[bom_df['procedure'] == procedure_name.strip()]
        for _, r in p_bom.iterrows():
            bom_items[str(r['standard_item']).strip().upper()] = {
                'avg_qty': float(r.get('average_quantity_used', 0) or 0),
                'avg_cost': float(r.get('average_unit_cost', 0) or 0)
            }

    # Construct recognized items
    recognized_construct_subcats = set()
    if not master_constructs_df.empty and 'procedure' in master_constructs_df.columns:
        m_mask = master_constructs_df['procedure'].astype(str).str.strip().str.upper() == procedure_name.strip().upper()
        for _, r in master_constructs_df[m_mask].iterrows():
            for itm in str(r.get('construct_items', '')).split(' + '):
                recognized_construct_subcats.add(itm.split(' | ')[0].strip().upper())

    # Aggregate item spend per UNSPSC subcategory
    item_agg = proc_df.groupby('contract_sub_category')['ext_cost'].sum().reset_index()
    item_agg = item_agg[item_agg['contract_sub_category'] != 'NAN'].sort_values(by='ext_cost', ascending=False)
    top_10 = item_agg.head(10).to_dict(orient='records')
    bottom_10 = item_agg[item_agg['ext_cost'] > 0].tail(10).sort_values(by='ext_cost', ascending=True).to_dict(orient='records')

    # Case level itemization
    case_items_agg = proc_df.groupby(['case_id', 'contract_sub_category']).agg(
        qty=('quantity_num', 'sum'),
        ext_cost=('ext_cost', 'sum'),
        brand=('brand', lambda x: ', '.join(x.dropna().astype(str).unique()))
    ).reset_index()

    items_dict = {}
    for row in case_items_agg.to_dict(orient='records'):
        c_id = row['case_id']
        subcat = row['contract_sub_category'] or 'UNKNOWN'
        qty = float(row['qty'])
        cost = float(row['ext_cost'])
        brand = row['brand']

        if c_id not in items_dict:
            items_dict[c_id] = {'items_list': [], 'standard_spend': 0.0, 'non_standard_spend': 0.0, 'used_subcats': set()}

        if subcat in bom_items:
            status = 'Standard'
            avg_qty = bom_items[subcat]['avg_qty']
            items_dict[c_id]['standard_spend'] += cost
        else:
            status = 'Non-Standard' if (not recognized_construct_subcats or subcat in recognized_construct_subcats) else 'Clinical Variance'
            avg_qty = 0.0
            items_dict[c_id]['non_standard_spend'] += cost

        items_dict[c_id]['used_subcats'].add(subcat)
        items_dict[c_id]['items_list'].append({
            'sub_category': subcat, 'status': status, 'qty': qty,
            'std_avg_qty': avg_qty, 'unit_cost': cost / qty if qty > 0 else 0,
            'brand': brand, 'total_cost': cost
        })

    # Add omitted standard items
    for c_id, data in items_dict.items():
        used = data['used_subcats']
        for std_item, std_val in bom_items.items():
            if std_item not in used:
                data['items_list'].append({
                    'sub_category': std_item, 'status': 'Omitted', 'qty': 0,
                    'std_avg_qty': std_val['avg_qty'], 'unit_cost': std_val['avg_cost'],
                    'brand': '-', 'total_cost': 0.0
                })
        used_total = len([x for x in data['items_list'] if x['status'] != 'Omitted'])
        compliant_total = len([x for x in data['items_list'] if x['status'] in ('Standard', 'Non-Standard')])
        data['construct_match_pct'] = round((compliant_total / used_total * 100), 1) if used_total > 0 else 100.0
        
        sort_order = {'Clinical Variance': 0, 'Non-Standard': 1, 'Standard': 2, 'Omitted': 3}
        data['items_list'].sort(key=lambda x: (sort_order.get(x['status'], 4), -x['total_cost']))
        del data['used_subcats']

    # Case level summary records
    case_summary = proc_df.drop_duplicates(subset=['case_id']).copy()
    final_cases = []
    for row in case_summary.to_dict(orient='records'):
        c_id = str(row.get('case_id', '')).strip()
        final_cases.append({
            'hospital_name': row.get('hospital_name'),
            'case_id': c_id,
            'Total_Case_Quantity': row.get('total_case_quantity'),
            'cost_per_case': row.get('cost_per_case'),
            'Cost_Volume_Outlier': row.get('cost_volume_outlier', 'No'),
            'Anomaly_Type_Detail': row.get('anomaly_type_detail', 'Normal Case'),
            'lead_surgeon': row.get('lead_surgeon'),
            'asa_rating': row.get('asa_rating', 'Not Specified'),
            'variance_data': items_dict.get(c_id, {'items_list': [], 'standard_spend': 0, 'non_standard_spend': 0, 'construct_match_pct': 100.0})
        })

    return {
        "cases": final_cases,
        "top_10": top_10,
        "bottom_10": bottom_10
    }

@app.get("/api/data/{procedure_name:path}")
def get_procedure_data(procedure_name: str, user: str = Depends(authenticate)):
    return _get_procedure_data_cached(procedure_name)

# --- SUNBURST HIERARCHICAL SPEND ---
class SunburstPayload(BaseModel):
    procedure_name: str = ""
    hospitals: List[str] = []
    surgeons: List[str] = []
    outliers: List[str] = []

@app.post("/api/sunburst-data")
def get_sunburst_data(payload: SunburstPayload):
    mask = df['unified_procedure'] == payload.procedure_name.strip()
    proc_df = df[mask].copy()
    if proc_df.empty: return {"data": []}

    if payload.hospitals:
        proc_df = proc_df[proc_df['hospital_name'].isin(payload.hospitals)]
    if payload.surgeons:
        proc_df = proc_df[proc_df['lead_surgeon'].isin(payload.surgeons)]
    if payload.outliers:
        proc_df = proc_df[proc_df['cost_volume_outlier'].isin(payload.outliers)]

    cols = ['spend_category', 'contract_category', 'contract_sub_category', 'brand', 'item_description']
    for c in cols:
        if c not in proc_df.columns:
            proc_df[c] = 'Unknown'
        else:
            proc_df[c] = proc_df[c].astype(str).str.strip().str.upper().replace({'NAN': 'Unknown', '': 'Unknown'})

    agg_df = proc_df.groupby(cols, as_index=False)['ext_cost'].sum()
    agg_df = agg_df[agg_df['ext_cost'] > 0]
    return {"data": agg_df.to_dict(orient='records')}

# --- AI INSIGHTS ---
class InsightPayload(BaseModel):
    procedure_name: str = ""
    total_spend: float = 0.0
    total_cases: int = 0
    outlier_pct: float = 0.0
    surgeon_stats: str = ""
    brand_pricing: str = ""

@app.post("/api/insights")
def generate_insights(payload: InsightPayload):
    prompt = f"""
You are the Chief Medical Officer and Lead Clinical Supply Chain Consultant at UC Health.
Analyze the following clinical utilization and spend variance metrics for:
Procedure: {payload.procedure_name}
Total Historical Spend: ${payload.total_spend:,.2f}
Total Surgical Cases: {payload.total_cases:,}
Outlier Case Rate: {payload.outlier_pct:.1f}%

Surgeon Utilization Breakdown:
{str(payload.surgeon_stats)[:1200]}

Brand & Price Disparity Data:
{str(payload.brand_pricing)[:1200]}

Provide a professional, executive-level clinical report structured into:
1. Executive Clinical Summary
2. Surgeon Variation & Preference Observations
3. Cost Drivers & Standardization Opportunities
4. Strategic Action Items for Clinical Quality & Value Committee

Format with clean HTML (<h3>, <p>, <ul>, <li>, <strong>, <span>). Do not wrap in markdown code blocks.
"""
    for model_name in [MultiTenantConfig.OPENAI_MODEL, MultiTenantConfig.OPENAI_FALLBACK_MODEL, "gpt-4o"]:
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return {"insight": response.choices[0].message.content}
        except Exception:
            continue
    return {"insight": "<p>⚠️ AI Insights temporarily unavailable.</p>"}

# --- CLINICAL SUBSTITUTIONS ---
class SubstitutionPayload(BaseModel):
    procedure_name: str = ""

@app.post("/api/substitutions")
def generate_substitutions(payload: SubstitutionPayload):
    mask = df['unified_procedure'] == payload.procedure_name.strip()
    proc_df = df[mask].copy()
    if proc_df.empty: return {"substitutions": []}

    candidates = []
    for subcat, grp in proc_df.groupby('contract_sub_category'):
        if subcat in ('NAN', 'UNKNOWN', 'GENERAL MEDICAL'): continue
        brand_agg = grp.groupby('brand').agg(
            avg_cost=('unit_cost_num', 'mean'),
            total_qty=('quantity_num', 'sum')
        ).reset_index()
        valid = brand_agg[brand_agg['avg_cost'] > 0].sort_values('avg_cost', ascending=False)
        
        if len(valid) > 1:
            records = valid.to_dict(orient='records')
            for i in range(len(records)):
                for j in range(len(records) - 1, i, -1):
                    exp = records[i]
                    chp = records[j]
                    if exp['brand'] == chp['brand']: continue
                    savings = (exp['avg_cost'] - chp['avg_cost']) * exp['total_qty']
                    if savings >= 100:
                        candidates.append({
                            "sub_category": subcat,
                            "target_brand": exp['brand'],
                            "replacement_brand": chp['brand'],
                            "projected_savings": float(savings),
                            "target_cost": float(exp['avg_cost']),
                            "replacement_cost": float(chp['avg_cost']),
                            "qty": float(exp['total_qty']),
                            "is_equivalent": True,
                            "rationale": f"Standard clinically comparable equivalent with ${savings:,.2f} projected savings."
                        })
                        
    candidates.sort(key=lambda x: x['projected_savings'], reverse=True)
    return {"substitutions": candidates[:10]}

# --- ASSOCIATION RULE MINING ---
class ARMPayload(BaseModel):
    procedure_name: str = ""

@app.post("/api/association-rules")
def get_association_rules(payload: ARMPayload):
    mask = df['unified_procedure'] == payload.procedure_name.strip()
    proc_df = df[mask].copy()
    if proc_df.empty: return {"rules": []}

    baskets = [grp['contract_sub_category'].unique().tolist() for _, grp in proc_df.groupby('case_id')]
    baskets = [b for b in baskets if len(b) > 1]
    if len(baskets) < 5: return {"rules": []}

    try:
        te = TransactionEncoder()
        te_ary = te.fit(baskets).transform(baskets)
        frequent = fpgrowth(pd.DataFrame(te_ary, columns=te.columns_), min_support=0.20, use_colnames=True)
        if frequent.empty: return {"rules": []}

        rules = association_rules(frequent, metric="confidence", min_threshold=0.6)
        if rules.empty: return {"rules": []}

        rules_list = []
        for _, r in rules.iterrows():
            rules_list.append({
                "if_item": ", ".join(list(r['antecedents'])),
                "then_item": ", ".join(list(r['consequents'])),
                "support": float(r['support']),
                "confidence": float(r['confidence']),
                "lift": float(r['lift'])
            })
        rules_list.sort(key=lambda x: (x['lift'], x['confidence']), reverse=True)
        return {"rules": rules_list[:15]}
    except Exception:
        return {"rules": []}

# --- PREFERENCE CARDS ---
class PrefCardPayload(BaseModel):
    procedure_name: str = ""
    surgeon_name: str = ""

@app.post("/api/preference-card")
def get_preference_card(payload: PrefCardPayload):
    proc = payload.procedure_name.strip()
    surg = payload.surgeon_name.strip()
    
    proc_mask = df['unified_procedure'] == proc
    full_proc_df = df[proc_mask]
    if full_proc_df.empty: return {"total_cases": 0, "open_list": [], "hold_list": []}

    # Facility standards
    facility_standards = {}
    for subcat, grp in full_proc_df.groupby('contract_sub_category'):
        facility_standards[subcat] = {
            "brand": grp['brand'].mode()[0] if not grp['brand'].empty else "Unknown",
            "median_qty": float(grp.groupby('case_id')['quantity_num'].sum().median())
        }

    surg_df = full_proc_df[full_proc_df['lead_surgeon'].astype(str).str.strip() == surg]
    total_cases = surg_df['case_id'].nunique()
    if total_cases == 0:
        return {"total_cases": 0, "open_list": [], "hold_list": []}

    open_list = []
    hold_list = []

    for subcat, grp in surg_df.groupby('contract_sub_category'):
        if subcat in ('NAN', 'UNKNOWN'): continue
        cases_used = grp['case_id'].nunique()
        util_pct = round((cases_used / total_cases) * 100, 1)
        
        surg_brand = grp['brand'].mode()[0] if not grp['brand'].empty else "Unknown"
        median_qty = float(grp.groupby('case_id')['quantity_num'].sum().median())
        
        fac_std = facility_standards.get(subcat, {})
        std_brand = fac_std.get("brand", "Unknown")
        std_qty = fac_std.get("median_qty", 1.0)
        has_conflict = (surg_brand != std_brand and std_brand != "Unknown")

        item_data = {
            "sub_category": subcat,
            "surgeon_brand": surg_brand,
            "standard_brand": std_brand,
            "has_brand_conflict": has_conflict,
            "median_quantity": median_qty,
            "standard_quantity": std_qty,
            "historical_utilization_pct": util_pct,
            "cases_used": cases_used
        }

        if util_pct >= 70.0:
            open_list.append(item_data)
        elif util_pct >= 30.0:
            hold_list.append(item_data)

    open_list.sort(key=lambda x: x['historical_utilization_pct'], reverse=True)
    hold_list.sort(key=lambda x: x['historical_utilization_pct'], reverse=True)

    return {
        "total_cases": total_cases,
        "open_list": open_list,
        "hold_list": hold_list
    }

# --- COMPARE PREFERENCE CARDS ---
class CompareCardsPayload(BaseModel):
    procedure_name: str = ""
    surgeon_a: str = ""
    surgeon_b: str = ""

@app.post("/api/compare-preference-cards")
def compare_cards(payload: CompareCardsPayload):
    card_a = get_preference_card(PrefCardPayload(procedure_name=payload.procedure_name, surgeon_name=payload.surgeon_a))
    card_b = get_preference_card(PrefCardPayload(procedure_name=payload.procedure_name, surgeon_name=payload.surgeon_b))

    items_a = {x['sub_category']: x for x in (card_a.get('open_list', []) + card_a.get('hold_list', []))}
    items_b = {x['sub_category']: x for x in (card_b.get('open_list', []) + card_b.get('hold_list', []))}

    all_subcats = sorted(list(set(items_a.keys()).union(set(items_b.keys()))))
    comparison = []

    for sc in all_subcats:
        in_a = sc in items_a
        in_b = sc in items_b
        a_data = items_a.get(sc, {})
        b_data = items_b.get(sc, {})

        comparison.append({
            "sub_category": sc,
            "in_surgeon_a": in_a,
            "surgeon_a_brand": a_data.get("surgeon_brand", "-"),
            "surgeon_a_qty": a_data.get("median_quantity", 0),
            "in_surgeon_b": in_b,
            "surgeon_b_brand": b_data.get("surgeon_brand", "-"),
            "surgeon_b_qty": b_data.get("median_quantity", 0),
            "status": "Shared" if (in_a and in_b) else (f"Only {payload.surgeon_a}" if in_a else f"Only {payload.surgeon_b}")
        })

    return {
        "procedure": payload.procedure_name,
        "surgeon_a": payload.surgeon_a,
        "surgeon_b": payload.surgeon_b,
        "comparison": comparison
    }

# --- SAVINGS LEAKAGE ---
@app.post("/api/savings-leakage")
def get_savings_leakage(payload: ARMPayload):
    mask = df['unified_procedure'] == payload.procedure_name.strip()
    proc_df = df[mask].copy()
    if proc_df.empty: return {"leakage": []}

    results = []
    for subcat, grp in proc_df.groupby('contract_sub_category'):
        total_spend = float(grp['ext_cost'].sum())
        savings_opp = float(pd.to_numeric(grp.get('savings_opportunity', 0), errors='coerce').fillna(0).sum())
        non_compliant_spend = float(grp[grp['contract_flag'] == 'OFF CONTRACT']['ext_cost'].sum())
        
        if savings_opp > 0 or non_compliant_spend > 0:
            results.append({
                "sub_category": subcat,
                "total_spend": total_spend,
                "non_compliant_spend": non_compliant_spend,
                "savings_opportunity": savings_opp,
                "compliance_pct": round(((total_spend - non_compliant_spend) / total_spend * 100), 1) if total_spend > 0 else 100.0
            })

    results.sort(key=lambda x: x['savings_opportunity'], reverse=True)
    return {"leakage": results[:20]}

# --- PRICE PARITY ---
@app.post("/api/price-parity")
def get_price_parity(payload: ARMPayload):
    mask = df['unified_procedure'] == payload.procedure_name.strip()
    proc_df = df[mask].copy()
    if proc_df.empty: return {"parity": []}

    parity_results = []
    for subcat, grp in proc_df.groupby('contract_sub_category'):
        fac_prices = grp.groupby('hospital_name')['unit_cost_num'].mean().reset_index()
        valid = fac_prices[fac_prices['unit_cost_num'] > 0]
        if len(valid) > 1:
            min_p = float(valid['unit_cost_num'].min())
            max_p = float(valid['unit_cost_num'].max())
            spread = max_p - min_p
            spread_pct = (spread / min_p) * 100 if min_p > 0 else 0
            
            # Opportunity if all paid min_p
            total_qty = float(grp['quantity_num'].sum())
            total_spend = float(grp['ext_cost'].sum())
            optimal_spend = total_qty * min_p
            potential_savings = max(0.0, total_spend - optimal_spend)

            if spread_pct >= 10.0 and potential_savings >= 50.0:
                parity_results.append({
                    "sub_category": subcat,
                    "min_price": min_p,
                    "max_price": max_p,
                    "price_spread": spread,
                    "spread_pct": round(spread_pct, 1),
                    "potential_savings": round(potential_savings, 2),
                    "facilities_count": int(len(valid))
                })

    parity_results.sort(key=lambda x: x['potential_savings'], reverse=True)
    return {"parity": parity_results[:20]}

# --- PREDICTIVE COST MODELING (XGBoost) ---
class PredictPayload(BaseModel):
    procedure_name: str = ""
    age_bucket: str = "45-64"
    asa_rating: str = "3 - Severe Systemic Disease"
    bmi_bucket: str = "Below 30"
    smoking_status: str = "Non-Smoker"
    diabetic_status: str = "0"
    patient_type: str = "Inpatient"

@app.post("/api/predict-cost")
def predict_case_cost(payload: PredictPayload):
    mask = df['unified_procedure'] == payload.procedure_name.strip()
    proc_df = df[mask].copy()
    if proc_df.empty: return {"error": "Procedure not found"}

    cases = proc_df.drop_duplicates(subset=['case_id']).copy()
    if len(cases) < 15:
        avg_cost = float(cases['cost_per_case_num'].mean()) if not cases.empty else 0.0
        return {
            "predicted_cost": round(avg_cost, 2),
            "feature_importance": {"Baseline Average": 1.0},
            "model_r2": 0.0,
            "message": "Sample size < 15 cases. Returning baseline mean."
        }

    try:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.metrics import r2_score

        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)

        features = ['patient_age_bucket', 'asa_rating', 'bmi_bucket', 'smoking_status', 'diabetic_status', 'patient_type_bucket']
        cases_clean = cases.dropna(subset=['cost_per_case_num']).copy()
        
        for f in features:
            cases_clean[f] = cases_clean.get(f, 'Not Specified').fillna('Not Specified').astype(str)

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X = encoder.fit_transform(cases_clean[features])
        y = cases_clean['cost_per_case_num'].values
        model.fit(X, y)

        input_df = pd.DataFrame([{
            'patient_age_bucket': payload.age_bucket,
            'asa_rating': payload.asa_rating,
            'bmi_bucket': payload.bmi_bucket,
            'smoking_status': payload.smoking_status,
            'diabetic_status': payload.diabetic_status,
            'patient_type_bucket': payload.patient_type
        }])
        X_test = encoder.transform(input_df)
        pred = float(model.predict(X_test)[0])

        feature_names = encoder.get_feature_names_out(features)
        importances = model.feature_importances_
        imp_dict = {name: round(float(imp), 4) for name, imp in zip(feature_names, importances) if imp > 0.01}
        imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:5])

        r2 = round(float(r2_score(y, model.predict(X))), 3)

        return {
            "predicted_cost": round(max(0.0, pred), 2),
            "historical_mean": round(float(np.mean(y)), 2),
            "feature_importance": imp_dict,
            "model_r2": max(0.0, r2)
        }
    except Exception as e:
        return {"error": str(e)}

# --- RLHF HUMAN FEEDBACK ---
class FeedbackPayload(BaseModel):
    procedure_name: str
    surgeon_name: str
    substitutions: List[Dict[str, Any]]

@app.post("/api/preference-card/feedback")
def record_feedback(payload: FeedbackPayload):
    rlhf_file = MultiTenantConfig.rlhf_path()
    try:
        with open(rlhf_file, "a") as f:
            f.write(json.dumps({
                "procedure_name": payload.procedure_name,
                "surgeon_name": payload.surgeon_name,
                "substitutions": payload.substitutions,
                "timestamp": pd.Timestamp.now().isoformat()
            }) + "\n")
        return {"status": "success", "message": "Feedback recorded for AI retraining."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- ASK THE BEE (AI Clinical Assistant) ---
class BeePayload(BaseModel):
    question: str
    procedure_context: str = ""

@app.post("/api/ask-the-bee")
def ask_the_bee(payload: BeePayload):
    prompt = f"""
You are "The Bee", UC Health's Clinical and Surgical Intelligence Copilot.
User Question: "{payload.question}"
Active Procedure Context: "{payload.procedure_context}"

Answer concisely and accurately based on clinical supply chain principles, surgeon preference standardization, Bill of Materials optimization, and price parity.
"""
    for model_name in [MultiTenantConfig.OPENAI_MODEL, MultiTenantConfig.OPENAI_FALLBACK_MODEL, "gpt-4o"]:
        try:
            res = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return {"answer": res.choices[0].message.content}
        except Exception:
            continue
    return {"answer": "I'm sorry, I could not process your clinical inquiry at this moment."}
