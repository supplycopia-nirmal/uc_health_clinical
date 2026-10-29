import pandas as pd
from typing import Dict, Any, List
from .schemas import AgentPersona, SwarmTraceStep

PERSONA_SURGEON_BEE = AgentPersona(
    id="surgeon_bee",
    name="Surgeon Bee",
    role="Physician Advocate & RLHF Alignment",
    avatar="👨‍⚕️",
    specialty="Surgeon Autonomy, Nuance & Case Complexity Adjustments",
    description="Represents physician clinical judgment, protects surgeon choice in high-acuity/revision cases, prevents burnout from aggressive procurement mandates, and powers the RLHF alignment loop."
)

class SurgeonBee:
    def __init__(self, df: pd.DataFrame):
        self.persona = PERSONA_SURGEON_BEE
        self.df = df

    def evaluate_physician_alignment(
        self, 
        procedure_name: str, 
        surgeon_name: str, 
        step_counter: int
    ) -> Dict[str, Any]:
        """
        Analyzes physician variance, patient risk mix (ASA rating),
        and historical brand loyalty to propose realistic alignment scripts.
        """
        traces: List[SwarmTraceStep] = []
        
        traces.append(SwarmTraceStep(
            step_id=step_counter,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Advocating Physician Clinical Nuance & Acuity",
            status="running",
            thought=f"Reviewing surgeon '{surgeon_name}' on '{procedure_name}'. Factoring ASA risk adjustments (ASA 3-4 complexity) and tactile preference before standardization.",
            timestamp="Just now"
        ))

        mask = self.df['unified_procedure'].astype(str).str.strip().str.upper() == procedure_name.strip().upper()
        proc_df = self.df[mask].copy()

        if proc_df.empty:
            return {"traces": traces, "surgeon_scorecard": {}, "advocacy_notes": ""}

        # Surgeon specific breakdown
        surg_stats = proc_df.groupby('lead_surgeon').agg(
            cases=('case_id', 'nunique'),
            total_spend=('line_spend', 'sum'),
            high_acuity_cases=('asa_rating', lambda x: sum(1 for a in x if '3' in str(a) or '4' in str(a)))
        ).reset_index()

        surg_stats['avg_cpc'] = surg_stats['total_spend'] / surg_stats['cases']
        surg_stats['acuity_pct'] = (surg_stats['high_acuity_cases'] / surg_stats['cases']) * 100

        system_avg = proc_df['line_spend'].sum() / proc_df['case_id'].nunique() if proc_df['case_id'].nunique() > 0 else 0.0

        top_surgeons = []
        for _, row in surg_stats.sort_values('cases', ascending=False).head(5).iterrows():
            diff = row['avg_cpc'] - system_avg
            top_surgeons.append({
                "surgeon": str(row['lead_surgeon']),
                "cases": int(row['cases']),
                "avg_cpc": round(float(row['avg_cpc']), 2),
                "system_diff": round(float(diff), 2),
                "high_acuity_rate": f"{row['acuity_pct']:.1f}%",
                "alignment_status": "Aligned" if diff <= 0 else ("Acuity-Justified" if row['acuity_pct'] > 50 else "High Variance")
            })

        traces.append(SwarmTraceStep(
            step_id=step_counter + 1,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Physician Alignment Complete",
            status="completed",
            thought=f"Benchmarked {len(surg_stats)} surgeons against system average (${system_avg:,.2f}). Defended complex cases and proposed peer-to-peer clinical engagement.",
            details={"surgeons_benchmarked": len(surg_stats), "system_avg_cpc": round(system_avg, 2)},
            timestamp="Just now"
        ))

        return {
            "traces": traces,
            "surgeon_scorecard": {
                "system_avg_cpc": round(system_avg, 2),
                "surgeons": top_surgeons
            },
            "advocacy_notes": "Surgeons handling >50% ASA 3-4 patients require risk-adjusted benchmarking rather than raw cost targets to preserve clinical outcomes."
        }
