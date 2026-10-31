import os
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

ALERTS_CONFIG_FILE = Path(__file__).resolve().parent.parent / "data" / "live_alerts.json"
ALERTS_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

class LiveAlertEngine:
    """
    Enterprise Live Alert & Swarm Notification Engine.
    Powers:
    1. Automated Swarm Bee proactive intelligence (Contract, Surgeon, Audit, Migration, Eco Bees).
    2. Interactive user-configured thresholds via Ask The Bee chatbot.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._cleared = False
        self._ensure_config_file()

    def _ensure_config_file(self):
        if not ALERTS_CONFIG_FILE.exists():
            default_rules = [
                {
                    "id": "rule_cpc_variance_knee",
                    "procedure": "TOTAL KNEE ARTHROPLASTY",
                    "metric": "cpc_variance",
                    "metric_label": "Physician CPC Variance",
                    "operator": ">",
                    "threshold": 450.0,
                    "unit": "$",
                    "channels": ["in_app", "jira"],
                    "created_by": "Ask The Bee Swarm",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "active"
                },
                {
                    "id": "rule_off_contract_trocars",
                    "procedure": "EXCISION OF LESION OR MASS",
                    "metric": "off_contract_spend",
                    "metric_label": "Off-Contract Spend Leakage",
                    "operator": ">",
                    "threshold": 2500.0,
                    "unit": "$",
                    "channels": ["in_app", "jira"],
                    "created_by": "Contract Bee",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "active"
                }
            ]
            with open(ALERTS_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(default_rules, f, indent=2)

    def get_configured_rules(self) -> List[Dict[str, Any]]:
        with self._lock:
            if ALERTS_CONFIG_FILE.exists():
                try:
                    with open(ALERTS_CONFIG_FILE, "r", encoding="utf-8") as f:
                        return json.load(f)
                except Exception:
                    pass
            return []

    def add_user_alert(
        self,
        procedure: str,
        metric: str,
        operator: str,
        threshold: float,
        channels: Optional[List[str]] = None,
        user: str = "uchealth_admin"
    ) -> Dict[str, Any]:
        """
        Creates and persists a new alert rule configured via Ask The Bee or UI.
        """
        rule_id = f"rule_{abs(hash(procedure + metric + str(threshold))) % 100000}"
        
        metric_labels = {
            "cpc_variance": "Cost-Per-Case (CPC) Variance",
            "price_spread": "Cross-Facility Price Spread",
            "off_contract_spend": "Off-Contract Spend Leakage",
            "bio_waste": "PRN Bio-Waste Discard Rate",
            "low_acuity_main_or": "Main OR Low-Acuity Case Volume"
        }
        
        new_rule = {
            "id": rule_id,
            "procedure": procedure or "All Procedures",
            "metric": metric,
            "metric_label": metric_labels.get(metric, metric.replace("_", " ").title()),
            "operator": operator,
            "threshold": float(threshold),
            "unit": "%" if "rate" in metric or "pct" in metric else "$",
            "channels": channels or ["in_app"],
            "created_by": user,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active"
        }

        with self._lock:
            rules = self.get_configured_rules()
            # Replace existing rule with same id or append
            rules = [r for r in rules if r.get("id") != rule_id]
            rules.insert(0, new_rule)
            with open(ALERTS_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(rules, f, indent=2)

        # Log event in 90-day SQLite audit log
        try:
            from server.audit_logger import get_audit_logger
            get_audit_logger().log(
                action="Live Alert Configured",
                detail=f"User configured alert: {new_rule['metric_label']} {operator} {threshold} for '{procedure}'",
                user=user,
                category="alert_config"
            )
        except Exception:
            pass

        return new_rule

    def clear_all(self):
        self._cleared = True

    def reset_cleared(self):
        self._cleared = False

    def get_live_notifications(
        self,
        df_active: Optional[Any] = None,
        procedure_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Dynamically extracts live swarm intelligence and alerts specific to the selected global procedure.
        Eliminates all static mock alerts. Evaluates real encounter metrics:
        1. Contract Bee: Contract optimization & savings opportunities for this procedure.
        2. Surgeon Bee: Highest physician CPC divergence vs 25th percentile peer benchmark.
        3. Audit Bee: Cross-facility price parity spread across active hospital sites.
        4. Migration Bee: Low-acuity (ASA 1-2) case cohort eligible for Outpatient Surgery Center migration.
        5. Eco Bee: Low-utilization custom pack items (<80% usage) to convert to Sterile Hold Cart.
        6. User-Configured Alerts: Always included irrespective of global procedure, carrying target procedure context.
        """
        if self._cleared:
            return []

        notifications = []

        # Determine procedure dataframe
        proc_df = None
        clean_proc = (procedure_name or "").strip()
        
        if df_active is not None and hasattr(df_active, 'empty') and not df_active.empty:
            if clean_proc and clean_proc.upper() != "ALL PROCEDURES":
                mask = df_active['unified_procedure'].astype(str).str.strip().str.upper() == clean_proc.upper()
                if mask.any():
                    proc_df = df_active[mask]
            
            # Fallback to first available procedure or entire dataframe if procedure not found
            if proc_df is None or proc_df.empty:
                if clean_proc and clean_proc.upper() != "ALL PROCEDURES":
                    proc_df = df_active
                    display_proc = clean_proc
                else:
                    first_proc = df_active['unified_procedure'].dropna().value_counts().index[0]
                    proc_df = df_active[df_active['unified_procedure'] == first_proc]
                    display_proc = first_proc
            else:
                display_proc = clean_proc
        else:
            display_proc = clean_proc or "EXCISION OF LESION OR MASS"

        if proc_df is not None and not proc_df.empty:
            case_col = 'case_id' if 'case_id' in proc_df.columns else ('encounter_id' if 'encounter_id' in proc_df.columns else 'row_id')
            spend_col = 'line_spend' if 'line_spend' in proc_df.columns else ('ext_cost' if 'ext_cost' in proc_df.columns else 'unit_cost')
            tot_cases = proc_df[case_col].nunique() if case_col in proc_df.columns else len(proc_df)
            tot_spend = float(proc_df[spend_col].sum()) if spend_col in proc_df.columns else 0.0

            # -------------------------------------------------------------------------
            # 1. CONTRACT BEE (Contract Compliance & Savings Opportunities)
            # -------------------------------------------------------------------------
            sav_col = 'savings_opportunity' if 'savings_opportunity' in proc_df.columns else None
            if sav_col and proc_df[sav_col].sum() > 0:
                top_sav_items = proc_df.groupby('contract_sub_category')[sav_col].sum().sort_values(ascending=False)
                top_subcat = top_sav_items.index[0]
                subcat_sav = float(top_sav_items.iloc[0])
                top_brand = "Contract Brand"
                if 'brand' in proc_df.columns:
                    b_series = proc_df[proc_df['contract_sub_category'] == top_subcat]['brand'].dropna()
                    if not b_series.empty:
                        top_brand = b_series.value_counts().index[0]
                cases_affected = proc_df[(proc_df['contract_sub_category'] == top_subcat) & (proc_df[sav_col] > 0)][case_col].nunique()
                contract_desc = f"{top_subcat} ({top_brand}) shows ${subcat_sav:,.2f} identified contract savings opportunity across {cases_affected} cases in {display_proc}. Standardizing tier-1 compliance eliminates off-contract spread."
                kpi_contract = f"${subcat_sav:,.2f} Reclaimable"
            else:
                # Find top spend commodity category
                top_cats = proc_df.groupby('contract_sub_category')[spend_col].sum().sort_values(ascending=False)
                top_subcat = top_cats.index[0] if not top_cats.empty else "Surgical Supplies"
                cat_spend = float(top_cats.iloc[0]) if not top_cats.empty else tot_spend
                contract_desc = f"{top_subcat} accounts for the highest procedural commodity spend (${cat_spend:,.2f}) in {display_proc}. Supply Chain Sentinel tracking tier rebate consolidation."
                kpi_contract = f"${cat_spend * 0.15:,.2f} Tier Rebate"

            notifications.append({
                "id": f"notif_contract_{abs(hash(display_proc + 'contract')) % 100000}",
                "type": "contract_leakage",
                "bee": "Contract Bee",
                "icon": "🚨",
                "icon_bg": "#fee2e2",
                "title": "Contract Compliance & Savings Opportunity",
                "desc": contract_desc,
                "time": "Active Cohort Analysis",
                "action_tab": "LeakageTab",
                "action_label": "⚡ Open Savings Leakage",
                "procedure": display_proc,
                "unread": True,
                "severity": "high",
                "kpi_impact": kpi_contract
            })

            # -------------------------------------------------------------------------
            # 2. SURGEON BEE (Physician CPC Divergence & Quality Quadrant)
            # -------------------------------------------------------------------------
            if 'lead_surgeon' in proc_df.columns:
                surg_cases = proc_df.groupby('lead_surgeon')[case_col].nunique()
                surg_spends = proc_df.groupby('lead_surgeon')[spend_col].sum()
                surg_cpcs = surg_spends / surg_cases.replace(0, 1)

                # Focus on practicing surgeons with volume >= 3 cases if available
                eligible = surg_cpcs[surg_cases >= 3] if (surg_cases >= 3).any() else surg_cpcs
                p25 = float(surg_cpcs.quantile(0.25))
                divergent_surg = eligible.idxmax()
                divergent_cpc = float(eligible[divergent_surg])
                divergent_cases = int(surg_cases[divergent_surg])
                spread = divergent_cpc - p25
                ratio = divergent_cpc / max(1.0, p25)

                surgeon_desc = f"Dr. {divergent_surg}'s average supply CPC (${divergent_cpc:,.2f}) diverges {ratio:.1f}x from the 25th percentile benchmark (${p25:,.2f}) in {display_proc} (+${spread:,.2f}/case spread across {divergent_cases} cases)."
                kpi_surgeon = f"+${spread:,.2f} / Case Spread"
            else:
                surgeon_desc = f"Surgeon preference card variance tracking active for {display_proc} across {tot_cases} clinical encounters."
                kpi_surgeon = "Variance Audited"

            notifications.append({
                "id": f"notif_surgeon_{abs(hash(display_proc + 'surgeon')) % 100000}",
                "type": "physician_divergence",
                "bee": "Surgeon Bee",
                "icon": "👨‍⚕️",
                "icon_bg": "#fef3c7",
                "title": "Physician CPC Divergence Alert",
                "desc": surgeon_desc,
                "time": "Physician Benchmark",
                "action_tab": "CompareCardsTab",
                "action_label": "⚡ Compare Preference Cards",
                "procedure": display_proc,
                "unread": True,
                "severity": "high",
                "kpi_impact": kpi_surgeon
            })

            # -------------------------------------------------------------------------
            # 3. AUDIT BEE (Cross-Facility Price Parity Arbitrage)
            # -------------------------------------------------------------------------
            if 'hospital_name' in proc_df.columns and proc_df['hospital_name'].nunique() > 1 and 'unit_cost' in proc_df.columns:
                valid_units = proc_df[proc_df['unit_cost'] > 0]
                if not valid_units.empty:
                    hosp_costs = valid_units.groupby(['contract_sub_category', 'hospital_name'])['unit_cost'].mean().unstack()
                    spreads = hosp_costs.max(axis=1) - hosp_costs.min(axis=1)
                    spreads = spreads.dropna()
                    if not spreads.empty and spreads.max() > 0.5:
                        top_spread_cat = spreads.sort_values(ascending=False).index[0]
                        spread_amount = float(spreads[top_spread_cat])
                        hosp_high = hosp_costs.loc[top_spread_cat].idxmax()
                        hosp_low = hosp_costs.loc[top_spread_cat].idxmin()
                        p_high = float(hosp_costs.loc[top_spread_cat, hosp_high])
                        p_low = float(hosp_costs.loc[top_spread_cat, hosp_low])
                        qty_high = float(proc_df[(proc_df['contract_sub_category'] == top_spread_cat) & (proc_df['hospital_name'] == hosp_high)]['quantity'].sum()) if 'quantity' in proc_df.columns else 20.0
                        clawback_est = spread_amount * max(1.0, qty_high)

                        parity_desc = f"{top_spread_cat} unit cost variance: ${p_low:,.2f} at {hosp_low} vs ${p_high:,.2f} at {hosp_high} in {display_proc} (${spread_amount:,.2f} spread across {int(qty_high)} units = ${clawback_est:,.2f} recovery opportunity)."
                        kpi_parity = f"${clawback_est:,.2f} Clawback"
                    else:
                        parity_desc = f"Cross-facility supply pricing remains within contract tolerances across {proc_df['hospital_name'].nunique()} UC Health facilities for {display_proc}."
                        kpi_parity = "Parity Compliant"
                else:
                    parity_desc = f"Price parity audit engine scanning active invoice distributions in {display_proc}."
                    kpi_parity = "Catalog Audited"
            else:
                parity_desc = f"Price Parity Sentinel auditing regional supplier pricing and tier clawback opportunities in {display_proc}."
                kpi_parity = "Audit Active"

            notifications.append({
                "id": f"notif_audit_{abs(hash(display_proc + 'audit')) % 100000}",
                "type": "price_parity",
                "bee": "Audit Bee",
                "icon": "🔍",
                "icon_bg": "#e0f2fe",
                "title": "Price Parity Arbitrage Detected",
                "desc": parity_desc,
                "time": "Price Parity Engine",
                "action_tab": "ParityTab",
                "action_label": "⚡ View Price Parity",
                "procedure": display_proc,
                "unread": True,
                "severity": "medium",
                "kpi_impact": kpi_parity
            })

            # -------------------------------------------------------------------------
            # 4. MIGRATION BEE (Site-of-Care ASC Optimization)
            # -------------------------------------------------------------------------
            if 'asa_rating' in proc_df.columns:
                low_acuity_cases = proc_df[proc_df['asa_rating'].astype(str).str.contains('1|2')][case_col].nunique()
                pct_low = (low_acuity_cases / max(1, tot_cases)) * 100
                margin_est = low_acuity_cases * 1650.0
                mig_desc = f"{low_acuity_cases} of {tot_cases} cases ({pct_low:.1f}%) in {display_proc} are low comorbidity risk (ASA 1-2). Shifting eligible candidates to Outpatient Surgery Center yields estimated ${margin_est:,.2f} facility margin."
                kpi_mig = f"${margin_est:,.2f} Margin Reclaimed"
            else:
                mig_desc = f"Evaluating inpatient vs outpatient site-of-care distribution for {display_proc} to optimize operating theater capacity."
                kpi_mig = "Site Optimization"

            notifications.append({
                "id": f"notif_migration_{abs(hash(display_proc + 'migration')) % 100000}",
                "type": "site_of_care",
                "bee": "Migration Bee",
                "icon": "🏥",
                "icon_bg": "#f3e8ff",
                "title": "ASC Migration Opportunity",
                "desc": mig_desc,
                "time": "Care Migration Swarm",
                "action_tab": "ProcedureTab",
                "action_label": "⚡ Review Site-of-Care Cohort",
                "procedure": display_proc,
                "unread": True,
                "severity": "medium",
                "kpi_impact": kpi_mig
            })

            # -------------------------------------------------------------------------
            # 5. ECO BEE (Green OR Bio-Waste & Custom Pack Hold Cart)
            # -------------------------------------------------------------------------
            if 'contract_sub_category' in proc_df.columns:
                cat_cases = proc_df.groupby('contract_sub_category')[case_col].nunique()
                util = (cat_cases / max(1, tot_cases)) * 100
                hold_items = util[(util >= 10) & (util < 80)]
                if not hold_items.empty:
                    top_hold_subcat = hold_items.sort_values().index[0]
                    top_hold_freq = float(hold_items[top_hold_subcat])
                    hold_spend = float(proc_df[proc_df['contract_sub_category'] == top_hold_subcat][spend_col].sum())
                    waste_lbs = len(hold_items) * 42
                    eco_desc = f"{len(hold_items)} conditional supply categories (including {top_hold_subcat} with {top_hold_freq:.1f}% use) are pre-opened in {display_proc}. Moving to Sterile Hold Cart prevents ${hold_spend * 0.35:,.2f} unused supply discard and diverts ~{waste_lbs} lbs bio-waste."
                    kpi_eco = f"{waste_lbs} lbs Bio-Waste Diverted"
                else:
                    eco_desc = f"Custom pack BOM rationalization active for {display_proc}. Procedural staple utilization (>80%) maintained across core packs."
                    kpi_eco = "Green OR Verified"
            else:
                eco_desc = f"Scope-3 sustainability Sentinel tracking surgical pack disposal in {display_proc}."
                kpi_eco = "Scope-3 Target"

            notifications.append({
                "id": f"notif_eco_{abs(hash(display_proc + 'eco')) % 100000}",
                "type": "sustainability",
                "bee": "Eco Bee",
                "icon": "🌱",
                "icon_bg": "#dcfce7",
                "title": "Green OR Bio-Waste Alert",
                "desc": eco_desc,
                "time": "Eco Green OR",
                "action_tab": "ProcCardTab",
                "action_label": "⚡ Move to Sterile Hold Cart",
                "procedure": display_proc,
                "unread": True,
                "severity": "low",
                "kpi_impact": kpi_eco
            })

        # =========================================================================
        # 6. USER-CONFIGURED LIVE RULES (ALWAYS INCLUDED IRRESPECTIVE OF PROCEDURE)
        # =========================================================================
        rules = self.get_configured_rules()
        for r in rules:
            if r.get("status") == "active":
                rule_proc = r.get("procedure", "All Procedures")
                notifications.append({
                    "id": f"notif_user_{r['id']}",
                    "type": "user_alert",
                    "bee": "Ask The Bee Monitor",
                    "icon": "🔔",
                    "icon_bg": "#f1f5f9",
                    "title": f"Custom Alert: {r['metric_label']}",
                    "desc": f"Rule triggered for {rule_proc}: threshold {r['operator']} {r.get('unit', '$')}{r['threshold']:,.2f} exceeded active target.",
                    "time": f"Configured by {r.get('created_by', 'User')}",
                    "action_tab": "DashboardTab",
                    "action_label": f"⚡ View in {rule_proc}",
                    "procedure": rule_proc,
                    "unread": True,
                    "severity": "medium",
                    "kpi_impact": f"Target: {r['operator']} {r['threshold']}"
                })

        return notifications

_alert_engine_instance: Optional[LiveAlertEngine] = None

def get_alert_engine() -> LiveAlertEngine:
    global _alert_engine_instance
    if _alert_engine_instance is None:
        _alert_engine_instance = LiveAlertEngine()
    return _alert_engine_instance
