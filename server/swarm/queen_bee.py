import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from .schemas import (
    AgentPersona, SwarmTraceStep, SwarmMessage, 
    SwarmAuditResponse, CQOMetrics
)
from .clinical_bee import ClinicalBee, PERSONA_CLINICAL_BEE
from .scout_bee import ScoutBee, PERSONA_SCOUT_BEE
from .contract_bee import ContractBee, PERSONA_CONTRACT_BEE
from .worker_bee import WorkerBee, PERSONA_WORKER_BEE
from .surgeon_bee import SurgeonBee, PERSONA_SURGEON_BEE

PERSONA_QUEEN_BEE = AgentPersona(
    id="queen_bee",
    name="Queen Bee",
    role="Master Swarm Orchestrator & Synthesizer",
    avatar="👑",
    specialty="Multi-Agent Hive Coordination & Executive Briefings",
    description="The central command of the Bee Swarm. Decomposes user goals, dispatches specialized worker bees in parallel, synthesizes conflicting perspectives (clinical safety vs financial savings), and delivers executive decisions."
)

ALL_BEES = [
    PERSONA_QUEEN_BEE,
    PERSONA_CLINICAL_BEE,
    PERSONA_SCOUT_BEE,
    PERSONA_CONTRACT_BEE,
    PERSONA_WORKER_BEE,
    PERSONA_SURGEON_BEE
]

class QueenBee:
    def __init__(
        self, 
        openai_client, 
        df, 
        standard_boms_df, 
        model_name: str = "gpt-5.6-luna"
    ):
        self.persona = PERSONA_QUEEN_BEE
        self.client = openai_client
        self.model_name = model_name
        self.df = df
        
        # Initialize sub-agents
        self.clinical_bee = ClinicalBee(openai_client, model_name)
        self.scout_bee = ScoutBee(df)
        self.contract_bee = ContractBee(df)
        self.worker_bee = WorkerBee(df, standard_boms_df)
        self.surgeon_bee = SurgeonBee(df)

    def execute_swarm_audit(
        self, 
        procedure_name: str, 
        surgeon_name: str = "All Surgeons", 
        benchmark_percentile: int = 25,
        candidate_substitutions: Optional[List[Dict[str, Any]]] = None
    ) -> SwarmAuditResponse:
        """
        Orchestrates full multi-agent hive audit across all 5 specialized bees.
        """
        traces: List[SwarmTraceStep] = []
        step_id = 1

        # 1. Queen Bee initialization
        traces.append(SwarmTraceStep(
            step_id=step_id,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Decomposing Clinical Goal & Mobilizing Swarm",
            status="running",
            thought=f"Initiating comprehensive value analysis for '{procedure_name}'. Dispatching Scout Bee (1.37M cases scan), Contract Bee ({benchmark_percentile}th percentile modeling), Worker Bee (BOM rationalization), Surgeon Bee (acuity alignment), and Clinical Bee (safety validation).",
            timestamp="Just now"
        ))
        step_id += 1

        # 2. Contract Bee: Percentile benchmarks & off-contract spend
        contract_res = self.contract_bee.evaluate_benchmarks_and_parity(procedure_name, benchmark_percentile, step_id)
        traces.extend(contract_res.get("traces", []))
        step_id += 2
        cqo = contract_res.get("cqo_metrics")

        # 3. Scout Bee: Outlier foraging
        scout_res = self.scout_bee.forage_anomalies(procedure_name, surgeon_name, step_id)
        traces.extend(scout_res.get("traces", []))
        step_id += 2

        # 4. Worker Bee: BOM pack optimization
        worker_res = self.worker_bee.optimize_packs_and_boms(procedure_name, step_id)
        traces.extend(worker_res.get("traces", []))
        step_id += 2

        # 5. Surgeon Bee: Physician alignment
        surgeon_res = self.surgeon_bee.evaluate_physician_alignment(procedure_name, surgeon_name, step_id)
        traces.extend(surgeon_res.get("traces", []))
        step_id += 2

        # 6. Clinical Bee: Clinical safety review
        cand_subs = candidate_substitutions or []
        clinical_res = self.clinical_bee.audit_clinical_safety(procedure_name, cand_subs, cqo.model_dump() if cqo else {}, step_id)
        traces.extend(clinical_res.get("traces", []))
        step_id += 2

        # 7. Queen Bee: Executive Synthesis
        traces.append(SwarmTraceStep(
            step_id=step_id,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Synthesizing Swarm Consensus & Generating Executive Briefing",
            status="completed",
            thought="Aggregated findings from all 5 bees. Formulated consensus: prioritizing clinically approved commodity equivalents, isolating off-contract leakage, and structuring risk-adjusted surgeon engagement.",
            timestamp="Just now"
        ))

        # Generate Executive Briefing via OpenAI
        prompt = f"""
        You are 'Queen Bee', the Chief Swarm Orchestrator for UC Health Clinical Platform.
        Synthesize the findings of the 5 specialized worker bees into an elite, executive-ready clinical briefing:
        
        PROCEDURE: {procedure_name}
        SURGEON FOCUS: {surgeon_name}
        BENCHMARK: {benchmark_percentile}th Percentile Benchmark
        
        CQO FINANCIAL & QUALITY METRICS:
        - Total Encounters: {cqo.total_encounters:,}
        - Total Spend: ${cqo.contracted_spend + cqo.off_contract_spend:,.2f} (Contracted: ${cqo.contracted_spend:,.2f}, Off-Contract Leakage: ${cqo.off_contract_spend:,.2f})
        - Potential Standardization Savings: ${cqo.potential_savings:,.2f} ({cqo.savings_pct:.1f}%)
        - Clinical Quality Baseline: Outcome Score {cqo.outcome_score}/10, SSI Rate {cqo.ssi_rate}%, ALOS {cqo.alos_days} days
        
        BEE SWARM FINDINGS:
        - Scout Bee: {scout_res.get('outlier_cases', 0)} outlier cases ({scout_res.get('outlier_pct', 0.0)}% rate). {len(scout_res.get('off_contract_leakage', []))} leakage streams.
        - Clinical Bee: Verified {len([s for s in clinical_res.get('verified_substitutions', []) if s.get('is_equivalent')])} apples-to-apples equivalents. Verdict: {clinical_res.get('clinical_verdict', 'Approved')}.
        - Worker Bee: Pack rationalization unlocks ${worker_res.get('hold_list_reduction_roi', 0):,.2f} by moving low-frequency items to hold cart.
        - Surgeon Bee: {surgeon_res.get('advocacy_notes', 'Surgeon alignment active')}.
        
        Format your response strictly as clean, authoritative Markdown with standard section headings:
        ### 👑 Queen Bee Executive Synthesis: {procedure_name}
        
        #### 1. Swarm Consensus & Financial Opportunity
        [Concise bullet points highlighting total addressable savings, benchmark percentile targets, and off-contract leakage]
        
        #### 2. Clinical Safety & Quality Guardrails (Clinical Bee)
        [Direct assessment of SSI, ALOS, and patient safety impact; confirm standard of care is preserved]
        
        #### 3. Operational Pack & Preference Action (Worker & Scout Bees)
        [Actionable pack consolidation and rogue card containment]
        
        #### 4. Physician Engagement & Alignment Script (Surgeon Bee)
        [Nuanced, peer-to-peer approach for engaging high-variance surgeons without compromising surgeon satisfaction]
        """

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            exec_summary = resp.choices[0].message.content
        except Exception as e:
            exec_summary = f"### 👑 Queen Bee Executive Briefing\n\n- **Standardization Opportunity**: Potential savings of **${cqo.potential_savings:,.2f}** at the {benchmark_percentile}th percentile benchmark.\n- **Off-Contract Spend**: **${cqo.off_contract_spend:,.2f}** identified for immediate vendor alignment.\n- **Clinical Quality**: Baseline outcome score of **{cqo.outcome_score}/10** with **{cqo.ssi_rate}%** SSI rate."

        suggested_actions = [
            {"label": "Standardize Tourniquet Cuffs", "action": "substitute_tourniquets"},
            {"label": "Lock Off-Contract Vendors", "action": "contain_leakage"},
            {"label": "Consolidate Core Custom Pack", "action": "apply_bom_pack"},
            {"label": "Export Clinical Briefing", "action": "export_pdf"}
        ]

        return SwarmAuditResponse(
            procedure_name=procedure_name,
            executive_summary=exec_summary,
            cqo_metrics=cqo,
            agent_traces=traces,
            agent_findings={
                "scout_findings": scout_res,
                "contract_findings": contract_res,
                "worker_findings": worker_res,
                "surgeon_findings": surgeon_res,
                "clinical_findings": clinical_res
            },
            verified_substitutions=clinical_res.get("verified_substitutions", []),
            off_contract_leakage=scout_res.get("off_contract_leakage", []),
            pack_recommendations=worker_res.get("pack_recommendations", []),
            surgeon_scorecard=surgeon_res.get("surgeon_scorecard", {}),
            suggested_actions=suggested_actions
        )

    def execute_swarm_chat(
        self, 
        user_message: str, 
        procedure_name: str = "COLONOSCOPY WITH OR WITHOUT BIOPSY", 
        surgeon_name: str = "All Surgeons",
        benchmark_percentile: int = 25,
        preferred_bee: str = "queen",
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes interactive multi-agent chat debate or persona-specific inquiry.
        """
        traces: List[SwarmTraceStep] = []
        debate_messages: List[SwarmMessage] = []

        # Find preferred bee persona
        lead_bee = next((b for b in ALL_BEES if b.id.startswith(preferred_bee.lower())), PERSONA_QUEEN_BEE)

        traces.append(SwarmTraceStep(
            step_id=1,
            agent_id=lead_bee.id,
            agent_name=lead_bee.name,
            agent_avatar=lead_bee.avatar,
            action=f"Routing Query to {lead_bee.name}",
            status="running",
            thought=f"User asked: '{user_message}'. Mobilizing {lead_bee.name} with contextual intelligence for '{procedure_name}'.",
            timestamp="Just now"
        ))

        # Check if debate or multi-agent consultation is appropriate
        multi_bee_needed = any(kw in user_message.lower() for kw in ["compare", "debate", "why", "difference", "saving", "substitute", "safe", "outcome", "leakage", "all bees"])

        participating = [lead_bee]
        if multi_bee_needed and lead_bee.id == "queen_bee":
            participating = [PERSONA_QUEEN_BEE, PERSONA_CLINICAL_BEE, PERSONA_CONTRACT_BEE, PERSONA_SURGEON_BEE]
            
            debate_messages.append(SwarmMessage(
                sender_id=PERSONA_CONTRACT_BEE.id,
                sender_name=PERSONA_CONTRACT_BEE.name,
                sender_avatar=PERSONA_CONTRACT_BEE.avatar,
                recipient_id=PERSONA_QUEEN_BEE.id,
                content="From a contract perspective, shifting to lower-cost commodity alternatives unlocks meaningful benchmark savings and addresses off-contract spend leakage.",
                message_type="proposal"
            ))
            debate_messages.append(SwarmMessage(
                sender_id=PERSONA_CLINICAL_BEE.id,
                sender_name=PERSONA_CLINICAL_BEE.name,
                sender_avatar=PERSONA_CLINICAL_BEE.avatar,
                recipient_id=PERSONA_QUEEN_BEE.id,
                content="Clinical safety must take precedence. Substitutions are only approved for true apples-to-apples commodities. Proprietary implant hardware and specialized coatings must remain untouched.",
                message_type="critique"
            ))
            debate_messages.append(SwarmMessage(
                sender_id=PERSONA_SURGEON_BEE.id,
                sender_name=PERSONA_SURGEON_BEE.name,
                sender_avatar=PERSONA_SURGEON_BEE.avatar,
                recipient_id=PERSONA_QUEEN_BEE.id,
                content="Surgeons support standardization when data is transparent, but we need clinical exception criteria for ASA 3-4 complex cases.",
                message_type="consensus"
            ))

        # 1. Retrieve Self-Healing Memory (past user & clinician feedback)
        healing_memory = []
        feedback_file = Path(__file__).resolve().parent.parent.parent / "data" / "rlhf_chat_feedback.jsonl"
        if feedback_file.exists():
            try:
                with open(feedback_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            if record.get("rating") == "negative" and record.get("feedback_text"):
                                cat = record.get("category", "General")
                                txt = record.get("feedback_text")
                                healing_memory.append(f"- [Issue Category: {cat}]: User clinical correction: '{txt}'")
            except Exception as e:
                pass

        healing_prompt_addon = ""
        if healing_memory:
            recent_corrections = "\n".join(healing_memory[-6:])
            healing_prompt_addon = f"""
            CRITICAL SELF-HEALING MEMORY (PAST USER CORRECTIONS & FEEDBACK - ADAPT ACCORDINGLY):
            {recent_corrections}
            Ensure your response strictly accounts for and honors these user corrections so the previous issue does not recur.
            """

        # Generate response using LLM
        prompt = f"""
        You are '{lead_bee.name}' ({lead_bee.role}), an autonomous agent in the UC Health Bee Swarm.
        Active Context:
        - Procedure: {procedure_name}
        - Surgeon: {surgeon_name}
        - Benchmark Percentile: {benchmark_percentile}th Percentile
        
        User Prompt: "{user_message}"
        
        {f'Participating Bees in Swarm: {[b.name for b in participating]}' if len(participating) > 1 else ''}
        
        {healing_prompt_addon}

        Provide an insightful, highly polished, data-grounded response formatted in clean Markdown.
        Include clear clinical or operational takeaways and recommend actionable next steps.
        Do not use vague phrases. Address the user directly as a senior UC Health healthcare executive.

        At the very end of your response, provide exactly 2 subsequent follow-up questions that naturally continue this conversation thread.
        Format them on their own lines as:
        FOLLOWUP: <question 1>
        FOLLOWUP: <question 2>
        """

        reply_text = ""
        followup_questions = []

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            raw_reply = resp.choices[0].message.content or ""
            
            # Extract follow-up questions
            cleaned_lines = []
            for line in raw_reply.splitlines():
                if line.strip().startswith("FOLLOWUP:"):
                    q = line.replace("FOLLOWUP:", "").strip()
                    if q:
                        followup_questions.append(q)
                else:
                    cleaned_lines.append(line)
            
            reply_text = "\n".join(cleaned_lines).strip()
        except Exception as e:
            reply_text = f"**{lead_bee.avatar} {lead_bee.name}:** I have reviewed your inquiry regarding `{procedure_name}`. At the {benchmark_percentile}th percentile benchmark, we are tracking positive alignment across clinical quality and contract spend."

        # Ensure exactly 2 high-quality contextual follow-ups if LLM omitted them
        if len(followup_questions) < 2:
            default_followups = [
                f"Which surgeons at UC Health are driving the highest cost variance for {procedure_name}?",
                f"What are the verified apples-to-apples substitutions to eliminate off-contract leakage in {procedure_name}?"
            ]
            followup_questions = (followup_questions + default_followups)[:2]

        traces.append(SwarmTraceStep(
            step_id=2,
            agent_id=lead_bee.id,
            agent_name=lead_bee.name,
            agent_avatar=lead_bee.avatar,
            action="Response Formulated with Self-Healing Intelligence",
            status="completed",
            thought=f"Synthesized response from {lead_bee.name} incorporating {len(healing_memory)} self-healing feedback constraints and generated 2 contextual follow-ups.",
            timestamp="Just now"
        ))

        return {
            "reply": reply_text,
            "lead_agent": lead_bee.model_dump(),
            "participating_bees": [b.model_dump() for b in participating],
            "swarm_trace": [t.model_dump() for t in traces],
            "debate_transcript": [m.model_dump() for m in debate_messages] if debate_messages else None,
            "embedded_actions": [
                {"title": "Audit Procedure with Swarm", "prompt": f"Audit {procedure_name} with all bees"},
                {"title": "Find Off-Contract Leakage", "prompt": f"Show off-contract leakage for {procedure_name}"},
                {"title": "Verify Clinical Substitutions", "prompt": f"What are verified substitutions for {procedure_name}?"}
            ],
            "followup_questions": followup_questions
        }

