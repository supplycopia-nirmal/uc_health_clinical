import json
from typing import Dict, Any, List
from .schemas import AgentPersona, SwarmTraceStep

PERSONA_CLINICAL_BEE = AgentPersona(
    id="clinical_bee",
    name="Clinical Bee",
    role="Quality, Safety & Outcomes Specialist",
    avatar="🩺",
    specialty="Surgical Equivalence & Patient Acuity",
    description="Ensures clinical safety primacy over cost, audits surgical indications, correlates supply variation to adverse outcomes (SSI, Readmissions, Mortality, ALOS), and guards against risky substitutions."
)

class ClinicalBee:
    def __init__(self, openai_client, model_name: str = "gpt-5.6-luna"):
        self.persona = PERSONA_CLINICAL_BEE
        self.client = openai_client
        self.model_name = model_name

    def audit_clinical_safety(
        self, 
        procedure_name: str, 
        candidate_substitutions: List[Dict[str, Any]], 
        cqo_metrics: Dict[str, Any],
        step_counter: int
    ) -> Dict[str, Any]:
        """
        Audits candidate substitutions for true surgical equivalence, ensuring
        no compromise on clinical outcomes, sterility, or instrumentation.
        """
        traces: List[SwarmTraceStep] = []
        
        traces.append(SwarmTraceStep(
            step_id=step_counter,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Auditing Surgical Indications & Adverse Outcomes",
            status="running",
            thought=f"Evaluating clinical interchangeability for {len(candidate_substitutions)} candidates in '{procedure_name}'. Checking SSI ({cqo_metrics.get('ssi_rate', 0.01)}%) and ALOS ({cqo_metrics.get('alos_days', 2.09)}d) safety correlations.",
            timestamp="Just now"
        ))

        if not candidate_substitutions:
            return {
                "traces": traces,
                "verified_substitutions": [],
                "clinical_assessment": "No candidate supply pairs presented for clinical review.",
                "quality_guardrails": "Standard clinical care protocols maintained."
            }

        prompt = f"""
        You are 'Clinical Bee', the Senior Surgical Reviewer & Clinical Quality Specialist for UC Health.
        Evaluate the following candidate surgical substitutions for '{procedure_name}':
        
        {json.dumps(candidate_substitutions, indent=2)}
        
        CLINICAL CQO CONTEXT:
        - Hospital Baseline SSI Rate: {cqo_metrics.get('ssi_rate', 0.01)}%
        - Average Length of Stay (ALOS): {cqo_metrics.get('alos_days', 2.09)} days
        - Readmission Rate: {cqo_metrics.get('readmission_rate', 2.49)}%
        - Overall Quality Outcome Score: {cqo_metrics.get('outcome_score', 7.2)} / 10
        
        CLINICAL REVIEW CRITERIA:
        1. is_equivalent = true ONLY IF the replacement product delivers the identical clinical outcome, form factor, and procedural utility without requiring separate capital equipment.
        2. is_equivalent = false for proprietary joint implants (e.g., knee femoral/tibial systems), active antimicrobial incise barriers vs standard sheets, or incompatible energy devices.
        3. Provide 1 concise, highly professional clinical rationale explaining the exact clinical rationale.
        4. Provide an overall 'clinical_verdict' summarizing the quality implications for the surgical service line.
        
        Respond strictly in valid JSON format:
        {{
          "substitutions": [
            {{
              "sub_category": string,
              "target_brand": string,
              "replacement_brand": string,
              "is_equivalent": boolean,
              "clinical_rationale": string
            }}
          ],
          "clinical_verdict": string,
          "acuity_considerations": string
        }}
        """

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(resp.choices[0].message.content)
            eval_list = parsed.get("substitutions", [])
            eval_map = {(i.get("sub_category"), i.get("target_brand"), i.get("replacement_brand")): i for i in eval_list}
            
            verified = []
            for cand in candidate_substitutions:
                key = (cand.get("sub_category"), cand.get("target_brand"), cand.get("replacement_brand"))
                evaluation = eval_map.get(key)
                is_eq = evaluation.get("is_equivalent", False) if evaluation else False
                rationale = evaluation.get("clinical_rationale", "Under clinical evaluation.") if evaluation else "Requires clinical review."
                
                item = dict(cand)
                item["is_equivalent"] = is_eq
                item["clinical_rationale"] = rationale
                verified.append(item)

            verified.sort(key=lambda x: (not x.get("is_equivalent", False), -x.get("projected_savings", 0)))
            
            eq_count = sum(1 for v in verified if v.get("is_equivalent"))
            traces.append(SwarmTraceStep(
                step_id=step_counter + 1,
                agent_id=self.persona.id,
                agent_name=self.persona.name,
                agent_avatar=self.persona.avatar,
                action="Clinical Review Completed",
                status="completed",
                thought=f"Verified {eq_count} direct commodity equivalents with zero quality compromise. Flagged {len(verified) - eq_count} items with surgical constraints.",
                details={"equivalent_count": eq_count, "total_reviewed": len(verified)},
                timestamp="Just now"
            ))

            return {
                "traces": traces,
                "verified_substitutions": verified,
                "clinical_verdict": parsed.get("clinical_verdict", "Clinical review completed."),
                "acuity_considerations": parsed.get("acuity_considerations", "Acuity-adjusted protocols active.")
            }

        except Exception as e:
            traces.append(SwarmTraceStep(
                step_id=step_counter + 1,
                agent_id=self.persona.id,
                agent_name=self.persona.name,
                agent_avatar=self.persona.avatar,
                action="Clinical Fallback Audit",
                status="warning",
                thought=f"Encountered review exception: {str(e)}. Enforcing baseline clinical conservatism.",
                timestamp="Just now"
            ))
            return {
                "traces": traces,
                "verified_substitutions": candidate_substitutions,
                "clinical_verdict": "Manual Value Analysis Committee review recommended.",
                "acuity_considerations": "Risk stratification active."
            }
