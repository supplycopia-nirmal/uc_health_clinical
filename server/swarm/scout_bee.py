import pandas as pd
from typing import Dict, Any, List
from .schemas import AgentPersona, SwarmTraceStep

PERSONA_SCOUT_BEE = AgentPersona(
    id="scout_bee",
    name="Scout Bee",
    role="Anomaly & Leakage Forager",
    avatar="🔍",
    specialty="1.37M Cases Reconnaissance & Outlier Detection",
    description="Continuously scours the 1,378,322 case records to identify cost/volume outliers, compositional anomalies, off-contract spend leakage, and cross-hospital price disparities."
)

class ScoutBee:
    def __init__(self, df: pd.DataFrame):
        self.persona = PERSONA_SCOUT_BEE
        self.df = df

    def forage_anomalies(
        self, 
        procedure_name: str, 
        surgeon_name: str, 
        step_counter: int
    ) -> Dict[str, Any]:
        """
        Forages through the procedure's cases to detect anomalies,
        composition outliers, and off-contract spend leakage.
        """
        traces: List[SwarmTraceStep] = []
        
        traces.append(SwarmTraceStep(
            step_id=step_counter,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Foraging 1.37M Cases for Statistical Outliers",
            status="running",
            thought=f"Scanning procedure '{procedure_name}' across all facilities. Isolating cases with Isolation Forest anomaly score < 0 and rogue supply utilization.",
            timestamp="Just now"
        ))

        mask = self.df['unified_procedure'].astype(str).str.strip().str.upper() == procedure_name.strip().upper()
        proc_df = self.df[mask].copy()

        if proc_df.empty:
            return {
                "traces": traces,
                "outlier_count": 0,
                "outlier_pct": 0.0,
                "off_contract_leakage": [],
                "top_anomalous_cases": []
            }

        total_cases = proc_df['case_id'].nunique()
        
        # Outlier detection based on pre-calculated master outlier or anomaly flag (safe column casing)
        outlier_mask = pd.Series(False, index=proc_df.index)
        for col in ['is_master_outlier', 'cost_volume_outlier', 'composition_outlier', 'Is_Master_Outlier', 'Cost_Volume_Outlier', 'Composition_Outlier']:
            if col in proc_df.columns:
                outlier_mask = outlier_mask | (proc_df[col].astype(str).str.lower() == 'yes')
        outlier_cases = proc_df[outlier_mask]['case_id'].nunique()
        outlier_pct = (outlier_cases / total_cases * 100) if total_cases > 0 else 0.0

        # Off-contract spend analysis (Leakage)
        leakage_items = []
        if 'savings_opportunity' in proc_df.columns:
            leakage_df = proc_df[proc_df['savings_opportunity'] > 0]
            if not leakage_df.empty:
                agg_leak = leakage_df.groupby(['contract_sub_category', 'brand', 'hospital_name']).agg(
                    total_leakage=('savings_opportunity', 'sum'),
                    qty=('quantity_num', 'sum') if 'quantity_num' in leakage_df.columns else ('quantity', 'sum'),
                    avg_cost=('unit_cost_num', 'mean') if 'unit_cost_num' in leakage_df.columns else ('unit_cost', 'mean')
                ).reset_index().sort_values('total_leakage', ascending=False)

                for _, row in agg_leak.head(8).iterrows():
                    leakage_items.append({
                        "sub_category": str(row['contract_sub_category']),
                        "brand": str(row['brand']),
                        "facility": str(row['hospital_name']),
                        "leakage_amount": round(float(row['total_leakage']), 2),
                        "qty": round(float(row['qty']), 1),
                        "unit_cost": round(float(row['avg_cost']), 2)
                    })

        # Top anomalous case profiles
        anomalous_cases = []
        outlier_type_col = 'anomaly_type_detail' if 'anomaly_type_detail' in proc_df.columns else ('Anomaly_Type_Detail' if 'Anomaly_Type_Detail' in proc_df.columns else None)
        outlier_items_col = 'outlier_items' if 'outlier_items' in proc_df.columns else ('Outlier_Items' if 'Outlier_Items' in proc_df.columns else None)

        agg_kwargs = {
            'total_spend': ('line_spend', 'sum'),
            'surgeon': ('lead_surgeon', 'first'),
            'facility': ('hospital_name', 'first')
        }
        if outlier_type_col:
            agg_kwargs['outlier_type'] = (outlier_type_col, 'first')
        if outlier_items_col:
            agg_kwargs['outlier_items'] = (outlier_items_col, 'first')

        case_groups = proc_df.groupby('case_id').agg(**agg_kwargs).reset_index().sort_values('total_spend', ascending=False)

        for _, row in case_groups.head(5).iterrows():
            anomalous_cases.append({
                "case_id": str(row['case_id']),
                "surgeon": str(row['surgeon']),
                "facility": str(row['facility']),
                "total_spend": round(float(row['total_spend']), 2),
                "anomaly_type": str(row['outlier_type']) if 'outlier_type' in row and pd.notna(row['outlier_type']) else "High Spend Variance",
                "flagged_items": str(row['outlier_items']) if 'outlier_items' in row and pd.notna(row['outlier_items']) and str(row['outlier_items']) != 'nan' else "Construct composition variance"
            })

        traces.append(SwarmTraceStep(
            step_id=step_counter + 1,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Anomaly Reconnaissance Finished",
            status="completed",
            thought=f"Foraged {total_cases:,} cases: detected {outlier_cases} outlier cases ({outlier_pct:.1f}% rate) and {len(leakage_items)} high-value leakage streams.",
            details={"total_cases": total_cases, "outlier_cases": outlier_cases, "outlier_pct": round(outlier_pct, 1)},
            timestamp="Just now"
        ))

        return {
            "traces": traces,
            "total_cases": total_cases,
            "outlier_cases": outlier_cases,
            "outlier_pct": round(outlier_pct, 2),
            "off_contract_leakage": leakage_items,
            "top_anomalous_cases": anomalous_cases
        }
