import pandas as pd
from typing import Dict, Any, List
from .schemas import AgentPersona, SwarmTraceStep

PERSONA_WORKER_BEE = AgentPersona(
    id="worker_bee",
    name="Worker Bee",
    role="BOM & Pack Optimizer",
    avatar="📦",
    specialty="FP-Growth Item Bundling & Standard Custom Packs",
    description="Mines clinical utilization patterns to design lean, standardized custom surgical packs, eliminates opened-but-unused hold waste, and aligns physician preference cards."
)

class WorkerBee:
    def __init__(self, df: pd.DataFrame, standard_boms_df: pd.DataFrame):
        self.persona = PERSONA_WORKER_BEE
        self.df = df
        self.boms_df = standard_boms_df

    def optimize_packs_and_boms(
        self, 
        procedure_name: str, 
        step_counter: int
    ) -> Dict[str, Any]:
        """
        Analyzes standard BOM items and item bundling to identify
        custom pack consolidation opportunities and hold-to-open ratio.
        """
        traces: List[SwarmTraceStep] = []
        
        traces.append(SwarmTraceStep(
            step_id=step_counter,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Optimizing Surgical BOMs & Pack Bundles",
            status="running",
            thought=f"Analyzing item utilization frequencies in '{procedure_name}'. Separating procedural staples (Open List, freq >= 80%) from conditional items (Hold List, freq < 80%).",
            timestamp="Just now"
        ))

        mask = self.df['unified_procedure'].astype(str).str.strip().str.upper() == procedure_name.strip().upper()
        proc_df = self.df[mask].copy()

        if proc_df.empty:
            return {"traces": traces, "pack_recommendations": [], "hold_list_reduction_roi": 0.0}

        total_cases = proc_df['case_id'].nunique()
        pack_recs = []
        hold_reduction_roi = 0.0

        if total_cases > 0:
            subcat_freq = proc_df.groupby('contract_sub_category').agg(
                case_count=('case_id', 'nunique'),
                total_qty=('quantity_num', 'sum'),
                avg_cost=('unit_cost_num', 'mean'),
                desc=('item_description', 'first'),
                brand=('brand', 'first')
            ).reset_index()

            subcat_freq['freq_pct'] = (subcat_freq['case_count'] / total_cases) * 100
            
            # Staples: >= 80% frequency (Core Pack candidates)
            staples = subcat_freq[subcat_freq['freq_pct'] >= 75.0].sort_values('freq_pct', ascending=False)
            # Low frequency items: 10% - 40% frequency (Hold candidates currently opened unnecessarily)
            hold_candidates = subcat_freq[(subcat_freq['freq_pct'] >= 10.0) & (subcat_freq['freq_pct'] < 40.0)]

            for _, row in staples.head(6).iterrows():
                pack_recs.append({
                    "sub_category": str(row['contract_sub_category']),
                    "item_description": str(row['desc'])[:60],
                    "preferred_brand": str(row['brand']),
                    "utilization_frequency": f"{row['freq_pct']:.1f}%",
                    "pack_placement": "Core Custom Pack (Pre-Opened)",
                    "action": "Bundle into standard procedure kit"
                })

            for _, row in hold_candidates.head(4).iterrows():
                # Potential waste savings if moved to Hold: 30% waste reduction on low freq items
                est_waste = (row['total_qty'] * row['avg_cost']) * 0.35
                hold_reduction_roi += est_waste
                pack_recs.append({
                    "sub_category": str(row['contract_sub_category']),
                    "item_description": str(row['desc'])[:60],
                    "preferred_brand": str(row['brand']),
                    "utilization_frequency": f"{row['freq_pct']:.1f}%",
                    "pack_placement": "Sterile Hold Cart (Open on Demand)",
                    "action": "Relocate from core pack to hold list"
                })

        traces.append(SwarmTraceStep(
            step_id=step_counter + 1,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Pack Rationalization Complete",
            status="completed",
            thought=f"Engineered standard pack configuration. Identified ${hold_reduction_roi:,.2f} in potential waste elimination by moving low-frequency items to the sterile hold cart.",
            details={"hold_reduction_roi": round(hold_reduction_roi, 2), "bundle_items": len(pack_recs)},
            timestamp="Just now"
        ))

        return {
            "traces": traces,
            "pack_recommendations": pack_recs,
            "hold_list_reduction_roi": round(hold_reduction_roi, 2)
        }
