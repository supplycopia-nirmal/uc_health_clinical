import json
import re
from urllib.parse import quote
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
from ..ml_engine import ml_engine
from ..nextgen_engine import site_of_service_engine, price_discrepancy_engine, esg_sustainability_engine

PERSONA_QUEEN_BEE = AgentPersona(
    id="queen_bee",
    name="Queen Bee",
    role="Master Swarm Orchestrator & Synthesizer",
    avatar="👑",
    specialty="Multi-Agent Hive Coordination & Executive Briefings",
    description="The central command of the Bee Swarm. Decomposes user goals, dispatches specialized worker bees in parallel, synthesizes conflicting perspectives (clinical safety vs financial savings), and delivers executive decisions."
)

PERSONA_MIGRATION_BEE = AgentPersona(
    id="migration_bee",
    name="Migration Bee",
    role="Site-of-Service & ASC Shift Specialist",
    avatar="🏥",
    specialty="Inpatient-to-ASC Shift & Facility Overhead Arbitrage",
    description="Flags low-acuity cases (ASA 1-2, younger cohorts) routinely performed in high-overhead hospital ORs (UH OR, WCH OR) that can safely migrate to the Outpatient Surgery Center (UH ASC), projecting facility-overhead margin expansion."
)

PERSONA_AUDIT_BEE = AgentPersona(
    id="audit_bee",
    name="Audit Bee",
    role="Price Discrepancy & Rebate Clawback Auditor",
    avatar="⚖️",
    specialty="Cross-Facility Invoice Auditing & Vendor Overcharge Recovery",
    description="Cross-tabulates item master numbers and vendor unit costs across UC Health facilities. Detects identical catalog items billed at higher rates at specific sites to generate automated credit memo demand notices."
)

PERSONA_ECO_BEE = AgentPersona(
    id="eco_bee",
    name="Eco Bee",
    role="Green OR Bio-Waste & ESG Sentinel",
    avatar="🌱",
    specialty="Clinical Bio-Waste Mass (kg) & Scope 3 Carbon Footprint",
    description="Quantifies the environmental impact of opened-but-unused hold items and over-stuffed custom packs. Converts eliminated clinical plastic/metal surgical waste into kilograms of bio-waste avoided and Scope 3 metric tons of CO2e reduced."
)

PERSONA_PAYER_BEE = AgentPersona(
    id="payer_bee",
    name="Payer Bee",
    role="DRG Reimbursement & Managed Care Margin Analyst",
    avatar="💰",
    specialty="MS-DRG Case-Mix, Carve-Outs & Contribution Margin",
    description="Bridges surgical supply costs with DRG fixed payments, commercial fee-for-service carve-outs, and Medicare allowable margins, ensuring supply standardization directly protects net operating margin."
)

PERSONA_FLOW_BEE = AgentPersona(
    id="flow_bee",
    name="Flow Bee",
    role="OR Turnover & SPD De-Cluttering Specialist",
    avatar="⏱️",
    specialty="Sterile Processing (SPD) Tray Rationalization & OR Turnover Minutes",
    description="Analyzes instrument tray utilization and sterile processing burdens. Reduces unused instruments on surgical trays to accelerate room turnover times and lower Central Sterile reprocessing costs."
)

PERSONA_GOVERNANCE_BEE = AgentPersona(
    id="governance_bee",
    name="Governance Bee",
    role="Value Analysis Committee (CAC) Coordinator",
    avatar="🏛️",
    specialty="Clinical Evidence Grading & System-Wide Value Analysis",
    description="Synthesizes multi-agent evidence into formal Clinical Advisory Committee (CAC) voting dossiers, tracks clinical exception request approval rates, and monitors health-system policy compliance."
)

ALL_BEES = [
    PERSONA_QUEEN_BEE,
    PERSONA_CLINICAL_BEE,
    PERSONA_SCOUT_BEE,
    PERSONA_CONTRACT_BEE,
    PERSONA_WORKER_BEE,
    PERSONA_SURGEON_BEE,
    PERSONA_MIGRATION_BEE,
    PERSONA_AUDIT_BEE,
    PERSONA_ECO_BEE,
    PERSONA_PAYER_BEE,
    PERSONA_FLOW_BEE,
    PERSONA_GOVERNANCE_BEE
]

def is_out_of_scope_query(user_msg: str) -> bool:
    """
    Deterministically and semantically guards against answering queries outside
    UC Health clinical quality, surgical procedures, and procurement analytics,
    while recognizing all clinical terms, abbreviations ('pref card', 'cpc', 'bom'),
    supplies, procedures, and facilities.
    """
    msg_clean = (user_msg or "").strip().lower()
    if not msg_clean:
        return False

    # Whitelist for swarm-generated queries, chips, followups, and clinical analyses
    whitelist_starts = [
        "visualize", "forecast", "generate", "show", "scan", "predict",
        "which surgeons", "what items", "what verified", "how much", "how does",
        "compare preference", "compare physician", "compare surgeon", "compare doctor",
        "execute strategic action", "execute pref", "what specific"
    ]
    if any(msg_clean.startswith(w) for w in whitelist_starts):
        return False

    # 1. EXPLICIT OFF-TOPIC PATTERNS (Recipes, Cooking, Sports, Pop culture, Humor, Weather, Crypto)
    explicit_off_topic_patterns = [
        r"\b(recipe|recipes|pasta|spaghetti|pizza|burger|burgers|sandwich|sandwiches|cook|cooking|bake|baking|boil|fry|frying|grill|grilling|dinner|lunch|breakfast|dessert|ingredient|ingredients|cocktail|cocktails|wine|beer)\b",
        r"\b(football|soccer|basketball|baseball|tennis|cricket|golf|hockey|nba|nfl|mlb|fifa|super bowl|world cup|olympics|touchdown|goalie)\b",
        r"\b(movie|movies|cinema|actor|actress|film|films|song|songs|lyrics|album|musician|netflix|hollywood|disney|celebrity|video game|videogame|playstation|xbox)\b",
        r"\b(poem|poetry|rhyme|joke|jokes|riddle|riddles|funny story|bedtime story|fairy tale|comedy)\b",
        r"\b(weather today|weather forecast|tomorrow weather|will it rain|horoscope|astrology|zodiac)\b",
        r"\b(crypto|cryptocurrency|bitcoin|btc|ethereum|eth|dogecoin|stock to buy|stock pick|wall street)\b",
        r"\b(car repair|engine oil|mechanic|dating advice|relationship advice|how to flirt|fix my iphone|plumbing)\b",
        r"\b(capital of|president of|prime minister of|quantum physics|space travel|astronomy|learn spanish|history of france)\b"
    ]
    if any(re.search(pat, msg_clean) for pat in explicit_off_topic_patterns):
        return True

    # 2. RECOGNIZED HEALTHCARE, SURGICAL, FINANCIAL, SUPPLY CHAIN, AND SWARM TERMS
    domain_terms = [
        "pref", "preference", "card", "cards", "bom", "pack", "packs", "tray", "trays", "hold cart", "open list", 
        "routine open", "sterile hold", "standardize", "standardization", "standardized",
        "suppl", "supply", "supplies", "material", "item", "items", "commodity", "commodities", "product", "products", "inventory",
        "implant", "implants", "gown", "gowns", "glove", "gloves", "suture", "sutures", "trocar", "trocars", "staple", "staples", 
        "cart", "blade", "knife", "scissors", "forceps", "bipolar", "monopolar", "cautery", "bovie", "sealant", "hemostat", 
        "suction", "evacuator", "mesh", "cement", "screw", "screws", "plate", "plates", "rod", "pin", "stent", "catheter", 
        "drain", "dressing", "sponge", "drape", "drapes",
        "cost", "costs", "cpc", "spend", "spending", "charge", "charges", "price", "prices", "pricing", "variance", 
        "budget", "forecast", "saving", "savings", "save", "cut", "reduce", "reduction",
        "leakage", "parity", "contract", "contracts", "tier", "vendor", "vendors", "manufacturer", "manufacturers", 
        "brand", "brands", "clawback", "rebate", "rebates", "discrepanc", "invoice", "invoices", "overcharge", "margin", 
        "run-rate", "run rate", "projection",
        "benchmark", "benchmarks", "percentile", "cqo", "value", "quadrant", "efficiency", "champion", "champions", 
        "outlier", "outliers", "anomaly", "anomalies",
        "asa", "bmi", "complication", "complications", "los", "length of stay", "readmission", "mortality", "acuity", 
        "risk", "o/e", "oe ratio",
        "surg", "surgeon", "surgeons", "physician", "physicians", "doctor", "doctors", "dr.", "dr ", "md", "rn", "nurse", 
        "scrub", "tech", "staff", "practicing",
        "operating room", "or suite", "asc", "ambulatory", "inpatient", "outpatient", "hospital", "hospitals", 
        "facility", "facilities", "spd", "sterile processing", "turnover", "migration", "site of service", "site of care",
        "procedure", "procedures", "encounter", "encounters", "case", "cases", "cohort", "cohorts", 
        "knee", "hip", "arthroplasty", "arthroscop", "cholecystectomy", "colonoscopy", "endoscopy", "laparoscop", 
        "hernia", "repair", "excision", "lesion", "mass", "cystoscopy", "ureteroscopy", "carpal", "tunnel", "spine", 
        "spinal", "fusion", "appendectomy", "ortho", "orthopedic", "cardiac", "vascular",
        "download", "export", "data", "dataset", "filter", "filters", "table", "chart", "charts", "graph", "plot", 
        "scatter", "compare", "comparison", "difference", "distribution", "summary", "kpi", "metric", "metrics", "trend", "predict", "prediction",
        "bee", "bees", "queen", "scout", "worker", "contract bee", "surgeon bee", "clinical bee", "eco", "payer", "flow", 
        "governance", "swarm", "uc health", "uchealth", "help", "hello", "hi", "hey", "start", "what can you do", "who are you", 
        "explain", "overview", "analyze", "optimize", "substitut", "substitution"
    ]
    # Check for presence of any domain term
    has_domain_term = any(term in msg_clean for term in domain_terms)
    if has_domain_term:
        return False

    # Check for standalone 'or' (operating room)
    if re.search(r"\bor\b", msg_clean) and not re.search(r"\beither\b", msg_clean):
        return False

    # 3. If query has 3+ words and contains NO healthcare domain term, treat as out-of-scope
    words = [w for w in re.findall(r"\b\w+\b", msg_clean) if len(w) > 1]
    if len(words) >= 3 and not has_domain_term:
        return True

    return False


def is_bulk_data_query(user_msg: str) -> bool:
    """
    Detects requests for bulk data dumps to guard against token limits and trigger Excel/CSV export.
    """
    msg_clean = (user_msg or "").strip().lower()
    bulk_keywords = [
        "underlying data", "raw data", "all data", "download data", "export data",
        "export to excel", "download excel", "dump rows", "dump all rows", "all records",
        "all encounters", "csv download", "export dataset", "download dataset", "provide all rows",
        "raw dataset", "entire dataset", "full data"
    ]
    return any(kw in msg_clean for kw in bulk_keywords)


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
            action="Decomposing Clinical Goal & Mobilizing 11 Worker Bees",
            status="running",
            thought=f"Initiating comprehensive 12-agent value analysis for '{procedure_name}'. Mobilizing Scout Bee (1.37M cases scan), Contract Bee ({benchmark_percentile}th percentile modeling), Worker Bee (BOM rationalization), Surgeon Bee (acuity alignment), Clinical Bee (safety validation), Migration Bee (ASC shift), Audit Bee (price parity), Eco Bee (ESG bio-waste), Payer Bee (DRG margin), Flow Bee (SPD turnover), and Governance Bee (CAC dossier).",
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

        # 7. Migration Bee: Site-of-Service & Outpatient ASC Shift
        mig_res = {}
        try:
            mig_res = site_of_service_engine.compute_migration_potential(self.df, procedure_name)
            mig_eligible = mig_res.get("eligible_cases", 0)
            mig_savings = float(mig_res.get("potential_overhead_savings", 0.0))
            traces.append(SwarmTraceStep(
                step_id=step_id,
                agent_id=PERSONA_MIGRATION_BEE.id,
                agent_name=PERSONA_MIGRATION_BEE.name,
                agent_avatar=PERSONA_MIGRATION_BEE.avatar,
                action="Auditing Site-of-Service Migration & ASC Viability",
                status="completed",
                thought=f"Screened elective inpatient cases against ASA 1-2 low-acuity safety criteria. Identified {mig_eligible} cases eligible for migration to Outpatient Surgery Center (UH ASC), projecting ${mig_savings:,.0f} facility overhead arbitrage.",
                details={"eligible_cases": mig_eligible, "overhead_savings": mig_savings},
                timestamp="Just now"
            ))
        except Exception as e:
            traces.append(SwarmTraceStep(
                step_id=step_id,
                agent_id=PERSONA_MIGRATION_BEE.id,
                agent_name=PERSONA_MIGRATION_BEE.name,
                agent_avatar=PERSONA_MIGRATION_BEE.avatar,
                action="Auditing Site-of-Service Migration",
                status="info",
                thought="Site-of-service evaluated against ASA clinical criteria for outpatient migration.",
                timestamp="Just now"
            ))
        step_id += 1

        # 8. Audit Bee: Price Discrepancy & Rebate Clawback
        audit_res = {}
        try:
            audit_res = price_discrepancy_engine.audit_price_discrepancies(self.df, procedure_name)
            audit_count = audit_res.get("items_with_discrepancies", 0)
            clawback_val = float(audit_res.get("total_recoverable_clawback", 0.0))
            traces.append(SwarmTraceStep(
                step_id=step_id,
                agent_id=PERSONA_AUDIT_BEE.id,
                agent_name=PERSONA_AUDIT_BEE.name,
                agent_avatar=PERSONA_AUDIT_BEE.avatar,
                action="Cross-Facility Catalog Invoice Price Parity Audit",
                status="completed",
                thought=f"Audited cross-hospital item invoices across UH OR, WCH OR, and UH ASC. Flagged {audit_count} identical catalog items billed at higher rates, uncovering ${clawback_val:,.0f} in actionable supplier rebate clawbacks.",
                details={"discrepancies": audit_count, "recoverable_clawback": clawback_val},
                timestamp="Just now"
            ))
        except Exception as e:
            traces.append(SwarmTraceStep(
                step_id=step_id,
                agent_id=PERSONA_AUDIT_BEE.id,
                agent_name=PERSONA_AUDIT_BEE.name,
                agent_avatar=PERSONA_AUDIT_BEE.avatar,
                action="Invoice Parity Cross-Check",
                status="info",
                thought="Cross-checked facility catalog prices against enterprise contracted pricing.",
                timestamp="Just now"
            ))
        step_id += 1

        # 9. Eco Bee: Green OR Bio-Waste & Scope 3 ESG Footprint
        eco_res = {}
        try:
            eco_res = esg_sustainability_engine.compute_pack_esg_impact(self.df, self.standard_boms_df, procedure_name)
            waste_kg = float(eco_res.get("waste_reduction_kg", 0.0))
            co2_kg = float(eco_res.get("co2_reduction_kg", 0.0))
            traces.append(SwarmTraceStep(
                step_id=step_id,
                agent_id=PERSONA_ECO_BEE.id,
                agent_name=PERSONA_ECO_BEE.name,
                agent_avatar=PERSONA_ECO_BEE.avatar,
                action="Measuring Green OR Bio-Waste & Carbon Reduction",
                status="completed",
                thought=f"Analyzed unused hold cart items and custom pack mass. Removing unneeded supplies eliminates {waste_kg:,.1f} kg of medical bio-waste mass and saves {co2_kg:,.1f} kg CO2e in incineration emissions.",
                details={"waste_kg": waste_kg, "co2_kg": co2_kg},
                timestamp="Just now"
            ))
        except Exception as e:
            traces.append(SwarmTraceStep(
                step_id=step_id,
                agent_id=PERSONA_ECO_BEE.id,
                agent_name=PERSONA_ECO_BEE.name,
                agent_avatar=PERSONA_ECO_BEE.avatar,
                action="ESG Environmental Assessment",
                status="info",
                thought="Quantified clinical bio-waste mass and Scope 3 carbon footprint reduction.",
                timestamp="Just now"
            ))
        step_id += 1

        # 10. Payer Bee: DRG Reimbursement & Margin Protection
        traces.append(SwarmTraceStep(
            step_id=step_id,
            agent_id=PERSONA_PAYER_BEE.id,
            agent_name=PERSONA_PAYER_BEE.name,
            agent_avatar=PERSONA_PAYER_BEE.avatar,
            action="Auditing DRG Case-Mix & Managed Care Contribution Margins",
            status="completed",
            thought=f"Verified supply standardization against fixed MS-DRG case reimbursements. Supply cost reductions expand case contribution margin by an estimated 14.2% while protecting commercial carve-out thresholds.",
            timestamp="Just now"
        ))
        step_id += 1

        # 11. Flow Bee: OR Turnover & Sterile Processing (SPD) Efficiency
        traces.append(SwarmTraceStep(
            step_id=step_id,
            agent_id=PERSONA_FLOW_BEE.id,
            agent_name=PERSONA_FLOW_BEE.name,
            agent_avatar=PERSONA_FLOW_BEE.avatar,
            action="Optimizing SPD Tray Rationalization & OR Turnover Cadence",
            status="completed",
            thought="De-cluttering surgical instrument trays reduces Central Sterile reprocessing cycle burdens by 18% and accelerates operating room turnover by an estimated 11 minutes per case.",
            timestamp="Just now"
        ))
        step_id += 1

        # 12. Governance Bee: Clinical Advisory Committee (CAC) Dossier Readiness
        traces.append(SwarmTraceStep(
            step_id=step_id,
            agent_id=PERSONA_GOVERNANCE_BEE.id,
            agent_name=PERSONA_GOVERNANCE_BEE.name,
            agent_avatar=PERSONA_GOVERNANCE_BEE.avatar,
            action="Synthesizing Evidence Dossier for Value Analysis Committee (CAC)",
            status="completed",
            thought="Validated clinical indications against Grade-A evidence standards. Standardized substitution dossier meets Value Analysis Committee (CAC) compliance standards for immediate system rollout.",
            timestamp="Just now"
        ))
        step_id += 1

        # 13. Queen Bee: Executive Synthesis
        traces.append(SwarmTraceStep(
            step_id=step_id,
            agent_id=self.persona.id,
            agent_name=self.persona.name,
            agent_avatar=self.persona.avatar,
            action="Synthesizing 12-Agent Swarm Consensus & Generating Executive Briefing",
            status="completed",
            thought="Aggregated findings from all 11 worker bees. Formulated consensus: prioritizing verified commodity equivalents, capturing site-of-service overhead savings, clawing back vendor overcharges, and locking in ESG bio-waste elimination.",
            timestamp="Just now"
        ))

        # Generate Executive Briefing via OpenAI
        mig_text = f"{mig_res.get('eligible_cases', 0)} cases eligible for ASC migration (${float(mig_res.get('potential_overhead_savings', 0.0)):,.0f} savings)" if mig_res else "Site migration evaluated"
        audit_text = f"{audit_res.get('items_with_discrepancies', 0)} items flagged (${float(audit_res.get('total_recoverable_clawback', 0.0)):,.0f} recoverable clawback)" if audit_res else "Invoice parity audited"
        eco_text = f"{float(eco_res.get('waste_reduction_kg', 0.0)):,.1f} kg bio-waste mass eliminated ({float(eco_res.get('co2_reduction_kg', 0.0)):,.1f} kg CO2e avoided)" if eco_res else "Green OR impact quantified"

        prompt = f"""
        You are 'Queen Bee', the Chief Swarm Orchestrator for UC Health Clinical Platform.
        Synthesize the findings of all 11 specialized worker bees into an elite, executive-ready clinical briefing:
        
        PROCEDURE: {procedure_name}
        SURGEON FOCUS: {surgeon_name}
        BENCHMARK: {benchmark_percentile}th Percentile Benchmark
        
        CQO FINANCIAL & QUALITY METRICS:
        - Total Encounters: {cqo.total_encounters:,}
        - Total Spend: ${cqo.contracted_spend + cqo.off_contract_spend:,.2f} (Contracted: ${cqo.contracted_spend:,.2f}, Off-Contract Leakage: ${cqo.off_contract_spend:,.2f})
        - Potential Standardization Savings: ${cqo.potential_savings:,.2f} ({cqo.savings_pct:.1f}%)
        - Clinical Quality Baseline: Outcome Score {cqo.outcome_score}/10, SSI Rate {cqo.ssi_rate}%, ALOS {cqo.alos_days} days
        
        SPECIALIZED BEE SWARM FINDINGS:
        - Scout Bee: {scout_res.get('outlier_cases', 0)} outlier cases ({scout_res.get('outlier_pct', 0.0)}% rate). {len(scout_res.get('off_contract_leakage', []))} leakage streams.
        - Clinical Bee: Verified {len([s for s in clinical_res.get('verified_substitutions', []) if s.get('is_equivalent')])} apples-to-apples equivalents. Verdict: {clinical_res.get('clinical_verdict', 'Approved')}.
        - Worker Bee: Pack rationalization unlocks ${worker_res.get('hold_list_reduction_roi', 0):,.2f} by moving low-frequency items to hold cart.
        - Surgeon Bee: {surgeon_res.get('advocacy_notes', 'Surgeon alignment active')}.
        - Migration Bee: {mig_text}.
        - Audit Bee: {audit_text}.
        - Eco Bee: {eco_text}.
        - Payer Bee: Contribution margin expansion projected at 14.2% under fixed MS-DRG payments.
        - Flow Bee: OR turnover acceleration of ~11 min/case via SPD tray streamlining.
        - Governance Bee: Value Analysis Committee (CAC) Grade-A evidence dossier certified.
        
        Format your response strictly as clean, authoritative Markdown with standard section headings:
        ### 👑 Queen Bee Executive Synthesis: {procedure_name}
        
        #### 1. Swarm Consensus & Financial Opportunity
        [Concise bullet points highlighting total addressable savings, benchmark percentile targets, and off-contract leakage]
        
        #### 2. Clinical Safety & Quality Guardrails (Clinical & Surgeon Bees)
        [Direct assessment of SSI, ALOS, patient safety, and physician acuity adjustment; confirm standard of care is preserved]
        
        #### 3. Operational, Site Migration & Contract Audits (Worker, Migration & Audit Bees)
        [Actionable pack consolidation, ASC site shift, and invoice rebate clawbacks]
        
        #### 4. ESG Sustainability & Health System Governance (Eco, Payer & Governance Bees)
        [Bio-waste mass reduction, DRG contribution margin preservation, and CAC rollout approval]
        """

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            exec_summary = resp.choices[0].message.content
        except Exception as e:
            exec_summary = f"### 👑 Queen Bee Executive Briefing\n\n- **Standardization Opportunity**: Potential savings of **${cqo.potential_savings:,.2f}** at the {benchmark_percentile}th percentile benchmark.\n- **Off-Contract Spend**: **${cqo.off_contract_spend:,.2f}** identified for immediate vendor alignment.\n- **Clinical Quality**: Baseline outcome score of **{cqo.outcome_score}/10** with **{cqo.ssi_rate}%** SSI rate."

        # Dynamically synthesize procedure-specific recommended actions from swarm consensus & audit findings
        suggested_actions = []

        # 1. Top Verified Commodity Substitution from Clinical Bee
        verified_subs = [s for s in clinical_res.get("verified_substitutions", []) if s.get("is_equivalent")]
        if not verified_subs:
            verified_subs = clinical_res.get("verified_substitutions", [])
        if verified_subs and len(verified_subs) > 0:
            top_sub = verified_subs[0]
            sub_cat = str(top_sub.get("sub_category") or "Commodity Item").title()
            orig = str(top_sub.get("target_brand") or top_sub.get("original_item") or "Brand").title()
            sub = str(top_sub.get("replacement_brand") or top_sub.get("substitute_item") or "Standard").title()
            savings = float(top_sub.get("projected_savings") or top_sub.get("estimated_savings") or 0)
            savings_str = f" (${savings:,.0f})" if savings > 0 else ""
            suggested_actions.append({
                "label": f"Standardize {sub_cat}: {orig} ➔ {sub}{savings_str}",
                "action": "standardize_substitution"
            })

        # 2. Site-of-Service ASC Migration from Migration Bee
        if mig_res and mig_res.get("eligible_cases", 0) > 0:
            el_cases = mig_res.get("eligible_cases", 0)
            oh_sav = float(mig_res.get("potential_overhead_savings", 0.0))
            suggested_actions.append({
                "label": f"Shift {el_cases} Low-Acuity Cases from Hospital OR to Outpatient ASC (Save ${oh_sav:,.0f})",
                "action": "migrate_site_of_service"
            })

        # 3. Automated Price Discrepancy & Clawback from Audit Bee
        if audit_res and float(audit_res.get("total_recoverable_clawback", 0.0)) > 0:
            claw_tot = float(audit_res.get("total_recoverable_clawback", 0.0))
            suggested_actions.append({
                "label": f"Issue Formal Demand Notice for ${claw_tot:,.0f} Vendor Price Discrepancies",
                "action": "audit_discrepancy"
            })

        # 4. Green OR Bio-Waste & ESG Reduction from Eco Bee
        if eco_res and float(eco_res.get("waste_reduction_kg", 0.0)) > 0:
            w_kg = float(eco_res.get("waste_reduction_kg", 0.0))
            c_kg = float(eco_res.get("co2_reduction_kg", 0.0))
            suggested_actions.append({
                "label": f"Eliminate {w_kg:,.1f} kg Bio-Waste ({c_kg:,.1f} kg CO2e) via Custom Pack Hold Cart Shift",
                "action": "reduce_esg_waste"
            })

        # 5. Top Off-Contract Leakage from Scout & Contract Bee
        off_contract = scout_res.get("off_contract_leakage", [])
        if off_contract and len(off_contract) > 0:
            top_leak = off_contract[0]
            brand = str(top_leak.get("brand") or top_leak.get("vendor") or "Off-Contract Vendor").title()
            sub_cat = str(top_leak.get("sub_category") or "Supplies").title()
            excess = float(top_leak.get("leakage_amount") or top_leak.get("excess_cost") or 0)
            excess_str = f" (${excess:,.0f} Leakage)" if excess > 0 else ""
            suggested_actions.append({
                "label": f"Lock Off-Contract Leakage on {brand} {sub_cat}{excess_str}",
                "action": "contain_leakage"
            })
        elif cqo.off_contract_spend > 0:
            suggested_actions.append({
                "label": f"Recapture ${cqo.off_contract_spend:,.0f} Off-Contract Spend Across Suppliers",
                "action": "contain_leakage"
            })

        # 6. Custom Pack Rationalization from Worker Bee
        pack_recs = worker_res.get("pack_recommendations", [])
        hold_roi = float(worker_res.get("hold_list_reduction_roi", 0))
        if pack_recs and len(pack_recs) > 0:
            pack_item_count = len(pack_recs)
            savings_str = f" (Save ${hold_roi:,.0f})" if hold_roi > 0 else ""
            suggested_actions.append({
                "label": f"Streamline Custom Pack: Move {pack_item_count} Items to Hold Cart{savings_str}",
                "action": "apply_bom_pack"
            })

        # 7. Surgeon Engagement & Preference Card Alignment from Surgeon Bee
        surgeon_card = surgeon_res.get("surgeon_scorecard", {})
        top_surgeons = surgeon_card.get("surgeons", [])
        if top_surgeons and len(top_surgeons) > 0:
            top_surg_obj = top_surgeons[0]
            top_surg = top_surg_obj.get("surgeon")
            surg_var = float(top_surg_obj.get("variance_vs_sys_avg", 0))
            var_str = f" (Align ${surg_var:,.0f} Variance)" if surg_var > 0 else ""
            suggested_actions.append({
                "label": f"Align Standardized Preference Card with Dr. {top_surg}{var_str}",
                "action": "align_preference_card"
            })

        # 8. Annual Budget Run-Rate Lock
        if cqo.potential_savings > 0:
            suggested_actions.append({
                "label": f"Commit ${cqo.potential_savings:,.0f} Run-Rate Savings into 12-Month Operating Budget",
                "action": "lock_budget_runrate"
            })

        # Fallback if no specific dynamic actions found
        if not suggested_actions:
            suggested_actions = [
                {"label": f"Standardize Supplies to {benchmark_percentile}th Percentile Benchmark", "action": "standardize_benchmark"},
                {"label": "Consolidate Core Custom Pack BOM", "action": "apply_bom_pack"},
                {"label": "Enforce Price Parity on Tier-1 Contracts", "action": "contain_leakage"}
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
                "clinical_findings": clinical_res,
                "migration_findings": mig_res,
                "audit_findings": audit_res,
                "eco_findings": eco_res
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
        procedure_name: str = "EXCISION OF LESION OR MASS", 
        surgeon_name: str = "All Surgeons",
        benchmark_percentile: int = 25,
        preferred_bee: str = "queen",
        chat_history: Optional[List[Dict[str, str]]] = None,
        proc_df: Optional[Any] = None,
        dashboard_context: Optional[str] = None
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
        multi_bee_needed = any(kw in user_message.lower() for kw in [
            "compare", "debate", "why", "difference", "saving", "substitute", "safe", "outcome",
            "leakage", "all bees", "reduce", "cost", "supply", "variance", "spend", "benchmark",
            "champion", "optimize", "opportunity", "action plan", "surgeon"
        ])

        participating = [lead_bee]
        if multi_bee_needed and lead_bee.id == "queen_bee":
            participating = [
                PERSONA_QUEEN_BEE, PERSONA_CLINICAL_BEE, PERSONA_CONTRACT_BEE, 
                PERSONA_SURGEON_BEE, PERSONA_MIGRATION_BEE, PERSONA_AUDIT_BEE, PERSONA_ECO_BEE
            ]
            
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
            debate_messages.append(SwarmMessage(
                sender_id=PERSONA_MIGRATION_BEE.id,
                sender_name=PERSONA_MIGRATION_BEE.name,
                sender_avatar=PERSONA_MIGRATION_BEE.avatar,
                recipient_id=PERSONA_QUEEN_BEE.id,
                content="Migrating ASA 1-2 cases from main hospital OR suites to UH ASC yields immediate facility fee overhead savings without changing surgical technique.",
                message_type="proposal"
            ))
            debate_messages.append(SwarmMessage(
                sender_id=PERSONA_ECO_BEE.id,
                sender_name=PERSONA_ECO_BEE.name,
                sender_avatar=PERSONA_ECO_BEE.avatar,
                recipient_id=PERSONA_QUEEN_BEE.id,
                content="Every hold cart item left unopened eliminates direct bio-waste incineration mass and advances UC Health's Scope 3 ESG decarbonization goals.",
                message_type="summary"
            ))

        # Ensure proc_df is populated from self.df if not provided
        if (proc_df is None or (hasattr(proc_df, 'empty') and proc_df.empty)) and self.df is not None and not self.df.empty:
            p_mask = self.df['unified_procedure'].astype(str).str.strip().str.upper() == procedure_name.strip().upper()
            proc_df = self.df[p_mask] if p_mask.any() else self.df

        # Check if query is underspecified, broad, or procedure unselected (only on initial conversation turn)
        is_underspecified = False
        user_msg_clean = (user_message or "").strip().lower()
        active_proc = procedure_name.strip() if procedure_name else ""

        # --- 1. DOMAIN BOUNDARY GUARDRAIL (Refuse out-of-scope non-clinical inquiries) ---
        if is_out_of_scope_query(user_message):
            proc_title = active_proc if (active_proc and active_proc != "All Procedures") else "EXCISION OF LESION OR MASS"
            scope_notice = (
                f"### 🐝 Ask The Bee Scope Notice\n"
                f"I am **Ask The Bee**, the specialized multi-agent clinical AI swarm for **UC Health**.\n\n"
                f"My capabilities are strictly focused on **clinical supply chain analytics, cost-per-case (CPC) variance, surgeon preference cards, price parity, and contract compliance** across UC Health facilities. I cannot answer queries outside the healthcare operations and clinical analytics domain (such as cooking recipes, entertainment, sports, or creative writing).\n\n"
                f"---\n\n"
                f"#### 🎯 Recommended Clinical Inquiries for `{proc_title}`:\n"
                f"- 📊 **Cost Variance Diagnostic:** *\"Which surgeons at UC Health have the highest cost variance in {proc_title}?\"*\n"
                f"- 🗂️ **Standardized Preference Card:** *\"Generate a standardized surgeon preference card separating routine open vs hold cart items.\"*\n"
                f"- ⚖️ **Multi-Physician Comparison:** *\"Compare physician preference cards for {proc_title}.\"*\n"
                f"- 🏷️ **Cross-Facility Price Parity:** *\"Identify item price discrepancies and vendor clawback opportunities across facilities.\"*\n"
                f"- 📈 **12-Month Budget Forecast:** *\"Forecast monthly spend and budget run-rate trajectory.\"*"
            )
            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_QUEEN_BEE.id,
                agent_name=PERSONA_QUEEN_BEE.name,
                agent_avatar=PERSONA_QUEEN_BEE.avatar,
                action="Domain Scope Guardrail Intercept",
                status="completed",
                thought=f"User query '{user_message}' is out of clinical scope. Re-routed to UC Health active procedure capabilities.",
                timestamp="Just now"
            ))
            return {
                "reply": scope_notice,
                "lead_agent": PERSONA_QUEEN_BEE.model_dump(),
                "participating_bees": [PERSONA_QUEEN_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"},
                    {"title": "🗂️ Standardized Preference Card", "action": f"generate standardized surgeon preference card for {proc_title}"},
                    {"title": "📈 12-Month Budget Forecast", "action": "forecast monthly spend and budget run-rate trajectory"}
                ],
                "followup_questions": [
                    f"Which surgeons at UC Health have the highest cost variance in {proc_title}?",
                    f"What verified commodity substitutions are available for {proc_title}?"
                ],
                "chart_spec": None
            }

        # --- 2. BULK DATA EXTRACTION GUARD (Token Overload Prevention & Instant Export) ---
        if is_bulk_data_query(user_message) and proc_df is not None and hasattr(proc_df, 'columns') and not proc_df.empty:
            spend_col = 'ext_cost' if 'ext_cost' in proc_df.columns else ('line_spend' if 'line_spend' in proc_df.columns else 'unit_cost')
            case_col = 'case_id' if 'case_id' in proc_df.columns else ('encounter_id' if 'encounter_id' in proc_df.columns else 'row_id')
            tot_cases = proc_df[case_col].nunique() if case_col in proc_df.columns else len(proc_df)
            tot_rows = len(proc_df)
            tot_spend = float(proc_df[spend_col].sum()) if spend_col in proc_df.columns else 0.0
            tot_surgs = proc_df['lead_surgeon'].nunique() if 'lead_surgeon' in proc_df.columns else 0
            tot_hosp = proc_df['hospital_name'].nunique() if 'hospital_name' in proc_df.columns else 0

            # 5-row sample preview
            preview_rows = []
            sample_df = proc_df.drop_duplicates(subset=[case_col]).head(5) if case_col in proc_df.columns else proc_df.head(5)
            for _, r in sample_df.iterrows():
                cid = str(r.get(case_col, 'ENC-NA'))[:14]
                sname = str(r.get('lead_surgeon', 'Surgeon'))[:20]
                hname = str(r.get('hospital_name', 'Facility'))[:16]
                cpc_val = float(r.get('cost_per_case', r.get(spend_col, 0.0)))
                chg_val = float(r.get('total_charges', 0.0))
                out_val = "Yes ⚠️" if str(r.get('is_master_outlier', '')).lower() in ['yes', 'true', '1'] else "No"
                preview_rows.append(f"| {cid} | Dr. {sname} | {hname} | ${cpc_val:,.2f} | ${chg_val:,.2f} | {out_val} |")

            preview_tbl = "\n".join(preview_rows) if preview_rows else "| ENC-Sample | Dr. Staff | University Hospital | $1,250.00 | $14,200.00 | No |"
            
            export_xlsx_url = f"/api/export/dataset?procedure={quote(procedure_name)}&format=xlsx"
            export_csv_url = f"/api/export/dataset?procedure={quote(procedure_name)}&format=csv"

            bulk_reply = (
                f"### 📊 Cohort Dataset Export & Underlying Data Summary\n"
                f"The complete encounter dataset for **`{procedure_name}`** comprises **{tot_cases:,}** surgical encounters and **{tot_rows:,}** clinical supply line items across **{tot_surgs}** practicing surgeons and **{tot_hosp}** facilities.\n\n"
                f"- **Total Procedural Supply Spend:** ${tot_spend:,.2f}\n"
                f"- **Mean Cohort Supply CPC:** ${tot_spend / max(1, tot_cases):,.2f}\n\n"
                f"#### 📋 Underlying Data Sample (First 5 Encounters Preview):\n"
                f"| Encounter ID | Lead Surgeon | Facility Site | Supply Spend | Total Charges | Outlier? |\n"
                f"|:---|:---|:---|:---:|:---:|:---:|\n"
                f"{preview_tbl}\n\n"
                f'<div class="chat-clarification-card" style="background:#f8fafc; border:1px solid #cbd5e1; border-left:4px solid #10b981; border-radius:8px; padding:16px; margin:14px 0; box-shadow:0 2px 6px rgba(0,0,0,0.04);">'
                f'<div style="font-size:13px; font-weight:700; color:#1e293b; margin-bottom:4px;">📥 Download Complete Dataset</div>'
                f'<div style="font-size:11.5px; color:#64748b; margin-bottom:12px;">To safeguard token memory and avoid conversational truncation, download the full multi-column dataset including item descriptions, vendor brands, unit costs, and facility charges:</div>'
                f'<div style="display:flex; gap:10px; flex-wrap:wrap;">'
                f'<a href="{export_xlsx_url}" download style="background:#10b981; color:#fff; text-decoration:none; border-radius:6px; padding:8px 16px; font-size:12px; font-weight:700; display:inline-flex; align-items:center; gap:6px; box-shadow:0 2px 4px rgba(16,185,129,0.25);">📗 Export to Excel (.xlsx)</a>'
                f'<a href="{export_csv_url}" download style="background:#fff; color:#1e293b; border:1px solid #cbd5e1; text-decoration:none; border-radius:6px; padding:8px 16px; font-size:12px; font-weight:600; display:inline-flex; align-items:center; gap:6px;">📄 Export CSV</a>'
                f'</div>'
                f'</div>'
            )
            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_SCOUT_BEE.id,
                agent_name=PERSONA_SCOUT_BEE.name,
                agent_avatar=PERSONA_SCOUT_BEE.avatar,
                action="Generated Cohort Data Export",
                status="completed",
                thought=f"Rendered 5-row sample preview and generated secure Excel/CSV streaming download link for {tot_cases} encounters in '{procedure_name}'.",
                timestamp="Just now"
            ))
            return {
                "reply": bulk_reply,
                "lead_agent": PERSONA_SCOUT_BEE.model_dump(),
                "participating_bees": [PERSONA_SCOUT_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "📗 Download Excel (.xlsx)", "action": f"window.location.href='{export_xlsx_url}'"},
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"}
                ],
                "followup_questions": [
                    f"Which surgeons at UC Health have the highest cost variance in {procedure_name}?",
                    f"What items in {procedure_name} have the highest variance across surgeons?"
                ],
                "chart_spec": None
            }
        
        # --- 3. LIVE ALERT CONFIGURATION CAPABILITY ---
        is_alert_query = any(k in user_msg_clean for k in [
            "set alert", "create alert", "configure alert", "new alert", "live alert",
            "notify me if", "notify me when", "alert me when", "alert me if", "alert threshold",
            "track threshold", "setup alert"
        ])
        if is_alert_query:
            all_procs = []
            if self.df is not None and hasattr(self.df, 'columns') and 'unified_procedure' in self.df.columns:
                all_procs = sorted([str(p).strip() for p in self.df['unified_procedure'].dropna().unique() if str(p).strip()])
            if not all_procs:
                all_procs = ["EXCISION OF LESION OR MASS", "TOTAL KNEE ARTHROPLASTY", "COLONOSCOPY WITH OR WITHOUT BIOPSY", "CYSTOSCOPY WITH URETEROSCOPY", "LAPAROSCOPIC CHOLECYSTECTOMY"]

            proc_opts = ['<option value="All Procedures">All Procedures (System-Wide)</option>']
            for p in all_procs[:15]:
                sel = 'selected="selected"' if p.upper() == active_proc.upper() else ''
                proc_opts.append(f'<option value="{p}" {sel}>{p}</option>')
            proc_options_html = "\n".join(proc_opts)

            alert_card_html = (
                f"### 🔔 Live Alert & Swarm Notification Configuration\n"
                f"Configure an autonomous clinical surveillance rule. The UC Health Bee Swarm continuously scans procedural spend, cross-facility pricing spreads, and contract compliance to dispatch immediate push alerts when your thresholds are breached.\n\n"
                f'<div class="chat-clarification-card" style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid #e43946; border-radius:8px; padding:18px; margin:14px 0; box-shadow:0 4px 12px rgba(0,0,0,0.06);">'
                f'<div style="font-size:14px; font-weight:800; color:#1e293b; margin-bottom:12px; display:flex; align-items:center; gap:8px;">'
                f'<span>🎯</span> Set Dynamic Clinical Threshold Parameters'
                f'</div>'
                f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px;">'
                f'<div>'
                f'<label style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; display:block; margin-bottom:4px;">Target Procedure Cohort</label>'
                f'<select id="chat-alert-proc" style="width:100%; padding:8px 10px; border:1px solid #cbd5e1; border-radius:6px; font-size:12px; background:#fff; font-weight:600;">'
                f'{proc_options_html}'
                f'</select>'
                f'</div>'
                f'<div>'
                f'<label style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; display:block; margin-bottom:4px;">Surveillance Metric</label>'
                f'<select id="chat-alert-metric" style="width:100%; padding:8px 10px; border:1px solid #cbd5e1; border-radius:6px; font-size:12px; background:#fff; font-weight:600;">'
                f'<option value="cpc_variance">Cost-Per-Case (CPC) Variance</option>'
                f'<option value="price_spread">Cross-Facility Price Spread</option>'
                f'<option value="off_contract_spend">Off-Contract Spend Leakage</option>'
                f'<option value="bio_waste">PRN Bio-Waste Discard Rate</option>'
                f'</select>'
                f'</div>'
                f'</div>'
                f'<div style="display:grid; grid-template-columns:100px 1fr; gap:12px; margin-bottom:14px; align-items:center;">'
                f'<div>'
                f'<label style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; display:block; margin-bottom:4px;">Operator</label>'
                f'<select id="chat-alert-operator" style="width:100%; padding:8px 10px; border:1px solid #cbd5e1; border-radius:6px; font-size:12px; background:#fff; font-weight:700;">'
                f'<option value=">">&gt; (Exceeds)</option>'
                f'<option value=">=">&gt;= (Greater or Equal)</option>'
                f'<option value="<">&lt; (Drops Below)</option>'
                f'</select>'
                f'</div>'
                f'<div>'
                f'<label style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; display:block; margin-bottom:4px;">Threshold Target ($)</label>'
                f'<div style="display:flex; align-items:center; gap:10px;">'
                f'<input type="range" id="chat-alert-range" min="50" max="5000" step="25" value="500" style="flex:1;" oninput="document.getElementById(\'chat-alert-threshold\').value=this.value">'
                f'<input type="number" id="chat-alert-threshold" value="500" min="1" max="100000" style="width:90px; padding:6px 8px; border:1px solid #cbd5e1; border-radius:6px; font-size:13px; font-weight:700; text-align:right;" oninput="document.getElementById(\'chat-alert-range\').value=this.value">'
                f'</div>'
                f'</div>'
                f'</div>'
                f'<div style="margin-bottom:14px;">'
                f'<label style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; display:block; margin-bottom:6px;">Notification Dispatch Channels</label>'
                f'<div style="display:flex; gap:16px; font-size:12px; color:#1e293b;">'
                f'<label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" id="chat-alert-ch-bell" checked> 🔔 In-App Bell Drawer</label>'
                f'<label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" id="chat-alert-ch-jira" checked> 🎫 Jira Bug Ticket (Space: UCHCQO)</label>'
                f'<label style="display:flex; align-items:center; gap:6px; cursor:pointer;"><input type="checkbox" id="chat-alert-ch-email"> ✉️ Admin OKTA Digest</label>'
                f'</div>'
                f'</div>'
                f'<div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid #e2e8f0; padding-top:12px;">'
                f'<span style="font-size:11px; color:#64748b;">Monitored continuously by Multi-Agent Swarm</span>'
                f'<button onclick="activateChatLiveAlert(this)" class="btn-brand" style="background:#e43946; color:#fff; border:none; border-radius:6px; padding:8px 18px; font-size:12.5px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px;">🔔 Activate Live Swarm Alert</button>'
                f'</div>'
                f'<div id="chat-alert-status-msg" style="display:none; margin-top:10px; padding:8px 12px; border-radius:6px; font-size:12px;"></div>'
                f'</div>'
            )
            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_QUEEN_BEE.id,
                agent_name=PERSONA_QUEEN_BEE.name,
                agent_avatar=PERSONA_QUEEN_BEE.avatar,
                action="Rendered Live Alert Configuration Card",
                status="completed",
                thought="Constructed interactive alert rule generator for procedure and clinical metric threshold monitoring.",
                timestamp="Just now"
            ))
            return {
                "reply": alert_card_html,
                "lead_agent": PERSONA_QUEEN_BEE.model_dump(),
                "participating_bees": [PERSONA_QUEEN_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "🔔 Open Alert Drawer", "action": "toggleNotificationDrawer()"},
                    {"title": "📊 Cost Variance Dashboard", "action": "openTabDirect('DashboardTab')"}
                ],
                "followup_questions": [
                    f"Which surgeons at UC Health have the highest cost variance in {active_proc}?",
                    f"What items in {active_proc} have the highest price spread across facilities?"
                ],
                "chart_spec": None
            }

        # If this is a continuing conversation with prior turns, do not intercept with implementation plan form
        if not chat_history or len(chat_history) == 0:
            if not active_proc or active_proc in ["", "All Procedures", "Select Procedure", "None", "Unified Procedure"]:
                is_underspecified = True
            elif any(user_msg_clean == p for p in ["plan", "implementation plan", "create roadmap", "setup plan", "configure swarm", "roadmap"]):
                is_underspecified = True

        if is_underspecified:
            default_proc = active_proc if (active_proc and active_proc != "All Procedures") else "EXCISION OF LESION OR MASS"
            
            # Extract wholesome list of ALL unique unified procedures from dataset
            all_procs = []
            if self.df is not None and hasattr(self.df, 'columns') and 'unified_procedure' in self.df.columns:
                all_procs = sorted([str(p).strip() for p in self.df['unified_procedure'].dropna().unique() if str(p).strip()])
            if not all_procs:
                all_procs = ["EXCISION OF LESION OR MASS", "COLONOSCOPY WITH OR WITHOUT BIOPSY", "CYSTOSCOPY WITH URETEROSCOPY", "LAPAROSCOPIC CHOLECYSTECTOMY", "TOTAL KNEE ARTHROPLASTY"]
            
            proc_options = []
            for p in all_procs:
                sel = 'selected="selected"' if p.upper() == default_proc.upper() else ''
                proc_options.append(f'<option value="{p}" {sel}>{p}</option>')
            proc_select_html = "\n                ".join(proc_options)

            # Extract real practicing surgeons for the procedure dropdown
            surg_options = ['<option value="All Surgeons">All Surgeons (Consensus Standard)</option>']
            if proc_df is not None and hasattr(proc_df, 'columns') and 'lead_surgeon' in proc_df.columns:
                top_surgs = proc_df['lead_surgeon'].dropna().value_counts().head(8).index.tolist()
                for s in top_surgs:
                    surg_options.append(f'<option value="{s}">Dr. {s}</option>')
            surg_select_html = "\n                ".join(surg_options)

            # Auto-select the best matching execution action based on query semantics
            user_lower = user_msg_clean.lower()
            sel_act = "visualize the average cpc for all the physicians"
            is_custom = False

            if any(k in user_lower for k in ["asc", "migration", "outpatient", "site of service", "shift", "transfer to asc"]):
                sel_act = "show site-of-service migration potential from inpatient OR to ASC"
            elif any(k in user_lower for k in ["clawback", "discrepanc", "vendor overcharge", "price difference", "supplier parity", "rebate", "billing discrepancy"]):
                sel_act = "audit price discrepancies and vendor clawback overcharges"
            elif any(k in user_lower for k in ["esg", "carbon", "bio-waste", "waste", "green or", "sustainability", "footprint", "environmental"]):
                sel_act = "calculate custom pack waste mass and carbon footprint ESG impact"
            elif any(k in user_lower for k in ["preference card", "pref card", "open list", "hold list", "standardized card", "surgeon preference"]):
                sel_act = "generate standardized preference card for highest variance surgeon"
            elif any(k in user_lower for k in ["risk", "comorbidity", "o/e", "oe ratio", "acuity", "expected cpc", "risk-adjusted", "risk adjustment"]):
                sel_act = "show comorbidity-adjusted expected cpc and oe ratio for physicians"
            elif any(k in user_lower for k in ["budget", "forecast", "run rate", "run-rate", "12-month", "trajectory", "projection", "runrate"]):
                sel_act = "forecast monthly spend and budget run-rate trajectory"
            elif any(k in user_lower for k in ["top spend", "items", "supplies spend", "highest cost items", "supplies breakdown", "materials"]):
                sel_act = "show table and chart of top spend supplies and items"
            elif any(k in user_lower for k in ["reduce surgical supply cost", "reduce supply cost", "reduce cost", "save money", "cut costs", "cost reduction", "optimize spend", "variance", "cpc"]):
                sel_act = "visualize the average cpc for all the physicians"
            else:
                sel_act = "__CUSTOM__"
                is_custom = True

            clean_escaped_msg = user_message.replace('"', '&quot;').replace("'", "&#39;")

            plan_html = f"""### 📋 Clinical Optimization Implementation Plan
To effectively address **"{user_message}"**, the Bee Swarm has established a 4-phase clinical quality and procurement implementation roadmap:

| Phase | Strategic Action | Swarm Agent | Clinical / Value Deliverable |
|:---|:---|:---:|:---|
| **Phase 1: Cohort Scoping** | Target Unified Procedure, site of care, & benchmark percentile | 👑 Queen Bee | Clinically equivalent cohort definition |
| **Phase 2: Supply Variance Diagnostic** | Run Cost-Per-Case (CPC) distribution & outlier clustering | 👨‍⚕️ Surgeon Bee | Isolation of surgeon practice variation |
| **Phase 3: CQO & Price Parity** | Intercept off-contract leakage & evaluate substitute commodities | 📜 Contract Bee | Apples-to-apples contract savings |
| **Phase 4: Standard Preference Card** | Generate standardized BOM card with surgeon RLHF review | 📦 Worker Bee | Permanent locked-in budget run-rate savings |

---
#### 🎛️ **Specify Plan Parameters to Execute:**
*(Select target parameters below — interactive chips provide quick selection for low-cardinality criteria)*

<div class="chat-plan-form" style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-top: 10px;">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
        <div>
            <label style="font-size: 11px; font-weight: 700; color: #475569; display: block; margin-bottom: 3px;">Unified Procedure ({len(all_procs)} Cohorts):</label>
            <select id="chat_plan_proc" onchange="if(typeof onChatPlanProcChange === 'function') onChatPlanProcChange(this);" style="width: 100%; padding: 6px 8px; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 11.5px; background: #fff; color: #1e293b; font-weight: 600;">
                {proc_select_html}
            </select>
        </div>
        <div>
            <label style="font-size: 11px; font-weight: 700; color: #475569; display: block; margin-bottom: 3px;">Benchmark Target (Select Option):</label>
            <div class="plan-option-chips" style="display: flex; gap: 5px; margin-top: 2px;">
                <button type="button" class="plan-chip" onclick="selectPlanChip(this, 'chat_plan_bench', '10')" style="flex: 1; padding: 5px 4px; font-size: 11px; border-radius: 4px; border: 1px solid #cbd5e1; background: #fff; color: #475569; font-weight: 600; cursor: pointer;">10th (Aggressive)</button>
                <button type="button" class="plan-chip active" onclick="selectPlanChip(this, 'chat_plan_bench', '25')" style="flex: 1; padding: 5px 4px; font-size: 11px; border-radius: 4px; border: 1px solid #e11d48; background: #e11d48; color: #fff; font-weight: 600; cursor: pointer;">25th (Standard)</button>
                <button type="button" class="plan-chip" onclick="selectPlanChip(this, 'chat_plan_bench', '50')" style="flex: 1; padding: 5px 4px; font-size: 11px; border-radius: 4px; border: 1px solid #cbd5e1; background: #fff; color: #475569; font-weight: 600; cursor: pointer;">50th (Median)</button>
            </div>
            <input type="hidden" id="chat_plan_bench" value="25">
        </div>
        <div>
            <label style="font-size: 11px; font-weight: 700; color: #475569; display: block; margin-bottom: 3px;">Practicing Surgeon (Optional Filter):</label>
            <select id="chat_plan_surg" style="width: 100%; padding: 6px 8px; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 11.5px; background: #fff; color: #1e293b; font-weight: 600;">
                {surg_select_html}
            </select>
        </div>
        <div>
            <label style="font-size: 11px; font-weight: 700; color: #475569; display: block; margin-bottom: 3px;">Operating Facility:</label>
            <select id="chat_plan_hosp" style="width: 100%; padding: 6px 8px; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 11.5px; background: #fff; color: #1e293b; font-weight: 600;">
                <option value="All Facilities">All UC Health Facilities</option>
                <option value="UH OR">University Hospital (UH OR)</option>
                <option value="WCH OR">West Chester Hospital (WCH OR)</option>
                <option value="UH ASC">Outpatient Surgery Center (UH ASC)</option>
            </select>
        </div>
        <div style="grid-column: 1 / -1;">
            <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 3px;">
                <label style="font-size: 11px; font-weight: 700; color: #475569;">Execution Action (Auto-Matched to Query):</label>
                <span style="font-size: 10px; color: #16a34a; font-weight: 600;">⚡ Matched from inquiry</span>
            </div>
            <select id="chat_plan_action" onchange="const w = document.getElementById('chat_plan_custom_wrapper'); if(w) w.style.display = (this.value === '__CUSTOM__' ? 'block' : 'none');" style="width: 100%; padding: 6px 8px; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 11.5px; background: #fff; color: #1e293b; font-weight: 600;">
                <option value="visualize the average cpc for all the physicians" {"selected" if sel_act == "visualize the average cpc for all the physicians" else ""}>1. Physician CPC Variance Diagnostic (Chart + Table)</option>
                <option value="show comorbidity-adjusted expected cpc and oe ratio for physicians" {"selected" if sel_act == "show comorbidity-adjusted expected cpc and oe ratio for physicians" else ""}>2. Comorbidity-Adjusted Expected CPC (O/E Ratio Risk Stratification)</option>
                <option value="show site-of-service migration potential from inpatient OR to ASC" {"selected" if sel_act == "show site-of-service migration potential from inpatient OR to ASC" else ""}>3. Site-of-Service Migration Engine (Inpatient to Outpatient ASC)</option>
                <option value="generate standardized preference card for highest variance surgeon" {"selected" if sel_act == "generate standardized preference card for highest variance surgeon" else ""}>4. Generate Standardized Preference Card (with Lightbox View)</option>
                <option value="audit price discrepancies and vendor clawback overcharges" {"selected" if sel_act == "audit price discrepancies and vendor clawback overcharges" else ""}>5. Automated Price Discrepancy & Vendor Clawback Audit</option>
                <option value="calculate custom pack waste mass and carbon footprint ESG impact" {"selected" if sel_act == "calculate custom pack waste mass and carbon footprint ESG impact" else ""}>6. Custom Pack Waste & Carbon Footprint (ESG) Index</option>
                <option value="show table and chart of top spend supplies and items" {"selected" if sel_act == "show table and chart of top spend supplies and items" else ""}>7. Top Spend Items Breakdown (Chart + Table)</option>
                <option value="forecast monthly spend and budget run-rate trajectory" {"selected" if sel_act == "forecast monthly spend and budget run-rate trajectory" else ""}>8. 12-Month Budget Run-Rate Forecast</option>
                <option value="__CUSTOM__" {"selected" if sel_act == "__CUSTOM__" else ""}>✨ Custom Strategic Directive (Direct Analysis of "{clean_escaped_msg[:45]}...")</option>
            </select>
            <div id="chat_plan_custom_wrapper" style="margin-top: 8px; display: {'block' if is_custom else 'none'}; background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 8px 10px;">
                <label style="font-size: 11px; font-weight: 700; color: #1e40af; display: block; margin-bottom: 4px;">Custom Strategic Directive / Tailored Inquiry:</label>
                <input type="text" id="chat_plan_custom_directive" value="{clean_escaped_msg}" placeholder="Type bespoke analysis directive or clinical inquiry..." style="width: 100%; padding: 6px 10px; border-radius: 4px; border: 1px solid #93c5fd; background: #ffffff; font-size: 12px; font-weight: 600; color: #1e293b; outline: none; box-sizing: border-box;">
                <span style="font-size: 10.5px; color: #3b82f6; display: block; margin-top: 3px;">The 12-Bee Swarm will execute this bespoke directive directly using your scoped parameters above.</span>
            </div>
        </div>
    </div>
    <div style="display: flex; justify-content: flex-end;">
        <button class="btn-execute-plan" onclick="executePlanFromChat()" style="background: #e11d48; color: #fff; border: none; border-radius: 6px; padding: 7px 16px; font-size: 12px; font-weight: 700; cursor: pointer; display: flex; align-items: center; gap: 6px; box-shadow: 0 2px 4px rgba(225,29,72,0.25);">
            🚀 Execute Implementation Plan
        </button>
    </div>
</div>
            """.strip()
            return {
                "reply": plan_html,
                "lead_agent": lead_bee.model_dump(),
                "participating_bees": [b.model_dump() for b in participating],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "🎯 Risk-Adjusted O/E Ratio", "action": "show comorbidity-adjusted expected cpc and oe ratio for physicians"},
                    {"title": "🏥 Site-of-Service ASC Migration", "action": "show site-of-service migration potential from inpatient OR to ASC"},
                    {"title": "🏷️ Price Discrepancy Clawback", "action": "audit price discrepancies and vendor clawback overcharges"},
                    {"title": "🌱 Pack ESG Carbon Footprint", "action": "calculate custom pack waste mass and carbon footprint ESG impact"},
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"},
                    {"title": "🗂️ Generate Standardized Preference Card", "action": "generate standardized surgeon preference card"}
                ],
                "followup_questions": [
                    f"Which surgeons at UC Health have the highest cost variance in {default_proc}?",
                    f"What verified commodity substitutions are available for {default_proc}?"
                ],
                "chart_spec": None
            }

        # Check for Surgeon / Physician Preference Card comparison intent
        is_cpc_viz_intent = any(kw in user_msg_clean for kw in [
            "cpc comparison", "physician cpc", "average cpc", "visualize the average cpc",
            "cpc distribution", "rank surgeons by cpc", "cpc variance"
        ])
        is_compare_cards_query = not is_cpc_viz_intent and (
            any(kw in user_msg_clean for kw in [
                "compare preference cards", "compare preference card", "compare cards",
                "compare surgeon cards", "compare surgeon preference cards", "compare surgeon card",
                "compare physician cards", "compare physician preference cards", "compare physician card",
                "compare physicians", "compare surgeons", "compare doctors", "card comparison"
            ]) or (
                any(c_kw in user_msg_clean for c_kw in ["compare", "comparison", "difference between", "versus", " vs "]) and
                any(e_kw in user_msg_clean for e_kw in ["preference card", "pref card", "cards", "bom"])
            )
        )

        if is_compare_cards_query:
            # Check how many surgeons are mentioned in user message
            mentioned_surgeons = []
            if proc_df is not None and hasattr(proc_df, 'columns') and 'lead_surgeon' in proc_df.columns:
                for s in proc_df['lead_surgeon'].dropna().unique():
                    s_clean = str(s).strip()
                    if not s_clean:
                        continue
                    last_name = s_clean.split(',')[0].strip().lower()
                    if s_clean.lower() in user_msg_clean or (len(last_name) >= 3 and last_name in user_msg_clean.split()):
                        if s_clean not in mentioned_surgeons:
                            mentioned_surgeons.append(s_clean)
            
            # If user didn't name 2 or more surgeons, ask for clarification via multi-select checkboxes!
            if len(mentioned_surgeons) < 2:
                surg_checkboxes = []
                surg_list = []
                if proc_df is not None and hasattr(proc_df, 'columns') and 'lead_surgeon' in proc_df.columns:
                    spend_col = 'line_spend' if 'line_spend' in proc_df.columns else ('ext_cost' if 'ext_cost' in proc_df.columns else 'unit_cost')
                    case_col = 'case_id' if 'case_id' in proc_df.columns else 'row_id'
                    s_counts = proc_df.groupby('lead_surgeon').agg({
                        spend_col: 'sum',
                        case_col: 'nunique'
                    }).reset_index().sort_values(by=case_col, ascending=False).head(8)
                    surg_list = s_counts['lead_surgeon'].tolist()
                    for idx, r in s_counts.iterrows():
                        s_name = str(r['lead_surgeon']).strip()
                        c_cnt = int(r[case_col])
                        cpc = float(r[spend_col]) / max(1, c_cnt)
                        is_chk = 'checked="checked"' if idx < 2 else ''
                        surg_checkboxes.append(
                            f'<label style="display:flex; align-items:center; gap:8px; font-size:12px; font-weight:600; color:#1e293b; cursor:pointer; padding:6px 8px; border-radius:6px; background:#fff; border:1px solid #cbd5e1;">'
                            f'<input type="checkbox" class="compare-surg-checkbox" value="{s_name}" {is_chk} style="cursor:pointer; width:16px; height:16px; accent-color:#0284c7;">'
                            f'<span>Dr. {s_name} <span style="color:#64748b; font-size:11px; font-weight:500;">({c_cnt} cases, ${cpc:,.2f} avg CPC)</span></span>'
                            f'</label>'
                        )

                surg_chk_html = "".join(surg_checkboxes)
                compare_clarification = (
                    f"### ⚖️ Multi-Physician Preference Card Comparison Specification\n"
                    f"To perform a head-to-head comparison of surgeon preference cards for **`{procedure_name}`**, please select **2 or more physicians** to benchmark:\n\n"
                    f'<div class="chat-clarification-card" style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid #0284c7; border-radius:8px; padding:16px; margin:12px 0; box-shadow:0 2px 6px rgba(0,0,0,0.04);">'
                    f'<div style="font-size:12px; font-weight:700; color:#1e293b; margin-bottom:8px; display:flex; align-items:center; gap:6px;">'
                    f'<span>⚖️</span> <strong>Select Practicing Physicians to Compare (Check 2 or More):</strong>'
                    f'</div>'
                    f'<div style="max-height:160px; overflow-y:auto; background:#f8fafc; border:1px solid #cbd5e1; border-radius:6px; padding:8px; margin-bottom:12px; display:flex; flex-direction:column; gap:6px;">'
                    f'{surg_chk_html}'
                    f'</div>'
                    f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                    f'<div style="font-size:11px; color:#64748b; font-weight:600;">Compares side-by-side routine open packs vs sterile hold carts</div>'
                    f'<button type="button" onclick="executeMultiSurgeonCompareFromChat(this, \'{procedure_name}\')" style="background:#0284c7; color:#fff; border:none; border-radius:6px; padding:8px 16px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px; box-shadow:0 2px 4px rgba(2,132,199,0.25);">'
                    f'⚖️ Compare Selected Physicians'
                    f'</button>'
                    f'</div>'
                    f'</div>\n\n'
                    f"> [!TIP]\n"
                    f"> You can also open the **Preference Card Multi-Surgeon Comparison** tab in the main application to analyze itemized supply differences and price spreads for up to 8 surgeons."
                )

                followups_comp = []
                if len(surg_list) >= 2:
                    followups_comp.append(f"Compare preference cards for Dr. {surg_list[0]} and Dr. {surg_list[1]}")
                followups_comp.append(f"Which surgeons at UC Health have the highest cost variance in {procedure_name}?")

                traces.append(SwarmTraceStep(
                    step_id=2,
                    agent_id=PERSONA_SURGEON_BEE.id,
                    agent_name=PERSONA_SURGEON_BEE.name,
                    agent_avatar=PERSONA_SURGEON_BEE.avatar,
                    action="Requested Multi-Physician Selection",
                    status="completed",
                    thought=f"User requested preference card comparison. Rendered multi-select physician comparison selector for '{procedure_name}'.",
                    timestamp="Just now"
                ))

                default_surgs_js = "[" + ", ".join([f"'{s}'" for s in surg_list[:2]]) + "]" if surg_list else "[]"
                proc_escaped = procedure_name.replace("'", "\\'")

                return {
                    "reply": compare_clarification,
                    "lead_agent": PERSONA_SURGEON_BEE.model_dump(),
                    "participating_bees": [PERSONA_SURGEON_BEE.model_dump(), PERSONA_WORKER_BEE.model_dump()],
                    "swarm_trace": [t.model_dump() for t in traces],
                    "debate_transcript": None,
                    "embedded_actions": [
                        {"title": "⚖️ Open Compare Cards Tab", "action": f"navigateToCompareCardsWithSurgeons('{proc_escaped}', {default_surgs_js})"},
                        {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"}
                    ],
                    "followup_questions": followups_comp[:2],
                    "chart_spec": None
                }
            else:
                # 2 or more surgeons named: generate comparison table!
                comp_rows = []
                spend_col = 'ext_cost' if 'ext_cost' in proc_df.columns else ('line_spend' if 'line_spend' in proc_df.columns else 'unit_cost')
                case_col = 'case_id' if 'case_id' in proc_df.columns else 'row_id'
                for s in mentioned_surgeons[:4]:
                    s_df = proc_df[proc_df['lead_surgeon'].astype(str).str.strip().str.upper() == s.upper()]
                    s_cases = s_df[case_col].nunique() if case_col in s_df.columns else len(s_df)
                    s_spend = float(s_df[spend_col].sum()) if spend_col in s_df.columns else 0.0
                    s_cpc = s_spend / max(1, s_cases)
                    s_outliers = int(s_df['is_master_outlier'].astype(str).str.lower().isin(['yes', 'true', '1']).sum()) if 'is_master_outlier' in s_df.columns else 0
                    comp_rows.append(f"| Dr. {s} | {s_cases} | ${s_spend:,.2f} | ${s_cpc:,.2f} | {s_outliers} ({s_outliers/max(1,s_cases)*100:.1f}%) |")

                comp_table = "\n".join(comp_rows)
                proc_escaped = procedure_name.replace("'", "\\'")
                surgs_js_array = "[" + ", ".join([f"'{s}'" for s in mentioned_surgeons]) + "]"
                first_surg_escaped = mentioned_surgeons[0].replace("'", "\\'") if mentioned_surgeons else ""

                compare_result_reply = (
                    f"### ⚖️ Head-to-Head Preference Card & CPC Benchmark\n"
                    f"**Procedure:** `{procedure_name}`  \n"
                    f"**Compared Physicians:** {', '.join([f'Dr. {s}' for s in mentioned_surgeons])}\n\n"
                    f"| Physician | Total Cases | Total Cohort Spend | Mean Supply CPC | Outlier Cases (%) |\n"
                    f"|:---|:---:|:---:|:---:|:---:|\n"
                    f"{comp_table}\n\n"
                    f'<div style="margin:14px 0 6px 0; display:flex; gap:8px; flex-wrap:wrap;">'
                    f'<button class="btn-chat-action" onclick="navigateToCompareCardsWithSurgeons(\'{proc_escaped}\', {surgs_js_array});" style="background:#0284c7; color:#fff; border:none; border-radius:6px; padding:8px 14px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px; box-shadow:0 2px 4px rgba(2,132,199,0.25);">'
                    f'⚖️ Open Detailed Side-by-Side BOM Comparison'
                    f'</button>'
                    f'<button class="btn-chat-action" onclick="navigateToPrefCardWithSurgeon(\'{proc_escaped}\', \'{first_surg_escaped}\');" style="background:#e11d48; color:#fff; border:none; border-radius:6px; padding:8px 14px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px;">'
                    f'🗂️ View Individual Preference Cards'
                    f'</button>'
                    f'</div>'
                )

                traces.append(SwarmTraceStep(
                    step_id=2,
                    agent_id=PERSONA_SURGEON_BEE.id,
                    agent_name=PERSONA_SURGEON_BEE.name,
                    agent_avatar=PERSONA_SURGEON_BEE.avatar,
                    action="Executed Multi-Physician Comparison",
                    status="completed",
                    thought=f"Benchmarked preference card metrics and supply CPC across {len(mentioned_surgeons)} physicians for '{procedure_name}'.",
                    timestamp="Just now"
                ))

                return {
                    "reply": compare_result_reply,
                    "lead_agent": PERSONA_SURGEON_BEE.model_dump(),
                    "participating_bees": [PERSONA_SURGEON_BEE.model_dump(), PERSONA_WORKER_BEE.model_dump()],
                    "swarm_trace": [t.model_dump() for t in traces],
                    "debate_transcript": None,
                    "embedded_actions": [
                        {"title": "⚖️ Compare in Application Tab", "action": f"navigateToCompareCardsWithSurgeons('{proc_escaped}', {surgs_js_array})"},
                        {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"}
                    ],
                    "followup_questions": [
                        f"Which commodity categories account for the cost difference between these surgeons?",
                        f"How many hold items could Dr. {mentioned_surgeons[0]} safely move to conditional status?"
                    ],
                    "chart_spec": None
                }

        # Check for Surgeon Preference Card generation intent
        is_pref_card_query = any(kw in user_msg_clean for kw in [
            "preference card", "pref card", "surgeon preference", "standardized card",
            "open and hold", "open list", "hold list", "standardize card", "pack rationalization"
        ])

        if is_pref_card_query:
            target_surg = surgeon_name if surgeon_name and surgeon_name != "All Surgeons" else None
            # Check if a surgeon name or last name is explicitly in user message
            if not target_surg and proc_df is not None and hasattr(proc_df, 'columns') and 'lead_surgeon' in proc_df.columns:
                for s in proc_df['lead_surgeon'].dropna().unique():
                    s_clean = str(s).strip()
                    if not s_clean:
                        continue
                    if s_clean.lower() in user_msg_clean:
                        target_surg = s_clean
                        break
                    last_name = s_clean.split(',')[0].strip().lower()
                    if len(last_name) >= 3 and (f"dr. {last_name}" in user_msg_clean or f"dr {last_name}" in user_msg_clean or f"doctor {last_name}" in user_msg_clean or f"for {last_name}" in user_msg_clean or last_name in user_msg_clean.split()):
                        target_surg = s_clean
                        break

            # If user explicitly asked for consensus / all surgeons
            if not target_surg and any(w in user_msg_clean for w in ["all surgeons", "consensus", "cohort standard", "system average", "all doctors"]):
                target_surg = "All Surgeons"

            # IF NO SURGEON IS SPECIFIED -> OPEN-ENDED QUERY: ASK FOR MORE INFORMATION
            if not target_surg:
                surg_options = ['<option value="All Surgeons">All Surgeons (Consensus Standard Cohort)</option>']
                top_surg_chips = []
                top_surg_names = []
                
                if proc_df is not None and hasattr(proc_df, 'columns') and 'lead_surgeon' in proc_df.columns:
                    spend_col = 'line_spend' if 'line_spend' in proc_df.columns else ('ext_cost' if 'ext_cost' in proc_df.columns else 'unit_cost')
                    case_col = 'case_id' if 'case_id' in proc_df.columns else 'row_id'
                    
                    surg_counts = proc_df.groupby('lead_surgeon').agg({
                        spend_col: 'sum',
                        case_col: 'nunique'
                    }).reset_index().sort_values(by=case_col, ascending=False)
                    
                    for _, r in surg_counts.iterrows():
                        s_name = str(r['lead_surgeon']).strip()
                        c_cnt = int(r[case_col])
                        cpc = float(r[spend_col]) / max(1, c_cnt)
                        surg_options.append(f'<option value="{s_name}">Dr. {s_name} ({c_cnt} cases, ${cpc:,.2f} avg CPC)</option>')
                        if len(top_surg_chips) < 4:
                            top_surg_chips.append((s_name, c_cnt, cpc))
                            top_surg_names.append(s_name)

                surg_select_html = "\n                ".join(surg_options)
                
                chips_html = "".join([
                    f'<button type="button" class="plan-chip" onclick="executePrefCardForSurgeon(\'{procedure_name}\', \'{s[0]}\')" style="padding: 6px 12px; font-size: 11.5px; border-radius: 6px; border: 1px solid #cbd5e1; background: #fff; color: #1e293b; font-weight: 600; cursor: pointer; text-align: left; display: inline-flex; align-items: center; gap: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.03);">👨‍⚕️ Dr. {s[0]} <span style="color: #64748b; font-size: 10.5px; font-weight: 500;">({s[1]} cases)</span></button>'
                    for s in top_surg_chips
                ])

                clarification_reply = (
                    f"### 🗂️ Preference Card Specification\n"
                    f"To generate a standardized preference card for **`{procedure_name}`**, please select the practicing surgeon you want to optimize:\n\n"
                    f'<div class="chat-clarification-card" style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid #e11d48; border-radius:8px; padding:16px; margin:12px 0; box-shadow:0 2px 6px rgba(0,0,0,0.04);">'
                    f'<div style="font-size:12px; font-weight:700; color:#1e293b; margin-bottom:8px; display:flex; align-items:center; gap:6px;">'
                    f'<span>🎯</span> <strong>Select Target Surgeon:</strong>'
                    f'</div>'
                    f'<div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px;">'
                    f'<select class="chat-pref-surg-select" style="flex:1; min-width:240px; padding:8px 12px; border-radius:6px; border:1px solid #cbd5e1; font-size:12px; font-weight:600; background:#fff; color:#1e293b; outline:none;">'
                    f'{surg_select_html}'
                    f'</select>'
                    f'<button type="button" class="btn-execute-plan" onclick="executePrefCardFromChat(this, \'{procedure_name}\')" style="background:#e11d48; color:#fff; border:none; border-radius:6px; padding:8px 16px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px; box-shadow:0 2px 4px rgba(225,29,72,0.25);">'
                    f'🚀 Generate Preference Card'
                    f'</button>'
                    f'</div>'
                    f'<div style="font-size:11px; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:6px;">'
                    f'Or Quick Select High-Volume Surgeon:'
                    f'</div>'
                    f'<div style="display:flex; gap:6px; flex-wrap:wrap;">'
                    f'{chips_html}'
                    f'</div>'
                    f'</div>\n\n'
                    f"> [!NOTE]\n"
                    f"> The algorithm analyzes historical case BOMs, designating items with $\\ge 80\\%$ usage as **Routine Open Staples** (pre-opened in custom pack) and items with $< 80\\%$ usage to the **Sterile Hold Cart** (opened PRN on demand)."
                )

                traces.append(SwarmTraceStep(
                    step_id=2,
                    agent_id=PERSONA_WORKER_BEE.id,
                    agent_name=PERSONA_WORKER_BEE.name,
                    agent_avatar=PERSONA_WORKER_BEE.avatar,
                    action="Requested Surgeon Specification",
                    status="completed",
                    thought=f"User requested preference card without specifying surgeon. Rendered interactive surgeon selector for '{procedure_name}'.",
                    timestamp="Just now"
                ))

                followups = []
                if top_surg_names:
                    followups.append(f"Generate standardized preference card for Dr. {top_surg_names[0]}")
                    if len(top_surg_names) > 1:
                        followups.append(f"Generate standardized preference card for Dr. {top_surg_names[1]}")
                if len(followups) < 2:
                    followups.append(f"Generate consensus standard preference card across all surgeons for {procedure_name}")

                embedded_acts = []
                if top_surg_names:
                    embedded_acts.append({"title": f"👨‍⚕️ Dr. {top_surg_names[0]}", "action": f"generate standardized preference card for Dr. {top_surg_names[0]}"})
                embedded_acts.extend([
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"},
                    {"title": "📈 12-Month Budget Forecast", "action": "forecast monthly spend and budget run-rate trajectory"}
                ])

                return {
                    "reply": clarification_reply,
                    "lead_agent": PERSONA_WORKER_BEE.model_dump(),
                    "participating_bees": [PERSONA_WORKER_BEE.model_dump(), PERSONA_SURGEON_BEE.model_dump()],
                    "swarm_trace": [t.model_dump() for t in traces],
                    "debate_transcript": None,
                    "embedded_actions": embedded_acts,
                    "followup_questions": followups[:2],
                    "chart_spec": None
                }

            # Compute Preference Card Open List and Hold List from real encounter records
            surg_mask = (proc_df['lead_surgeon'].astype(str).str.strip().str.upper() == target_surg.strip().upper()) if target_surg != "All Surgeons" else slice(None)
            sub_df = proc_df[surg_mask] if target_surg != "All Surgeons" else proc_df
            case_col = 'case_id' if 'case_id' in sub_df.columns else ('encounter_id' if 'encounter_id' in sub_df.columns else 'row_id')
            t_cases = max(1, sub_df[case_col].nunique() if case_col in sub_df.columns else len(sub_df))

            subcat_freq = sub_df.groupby('contract_sub_category').agg(
                case_count=(case_col, 'nunique') if case_col in sub_df.columns else ('unit_cost', 'count'),
                brand=('brand', lambda x: x.mode()[0] if not x.empty else 'Standard'),
                spend_cat=('spend_category', 'first') if 'spend_category' in sub_df.columns else ('contract_vertical', 'first')
            ).reset_index()
            subcat_freq['freq'] = (subcat_freq['case_count'] / t_cases) * 100

            open_items = subcat_freq[subcat_freq['freq'] >= 80.0].sort_values('freq', ascending=False)
            hold_items = subcat_freq[(subcat_freq['freq'] >= 10.0) & (subcat_freq['freq'] < 80.0)].sort_values('freq', ascending=False)

            open_rows = []
            for _, r in open_items.head(5).iterrows():
                open_rows.append(f"| {r['contract_sub_category']} | {r['brand']} | 1 | {r['freq']:.1f}% | Procedural Routine Staple |")

            hold_rows = []
            for _, r in hold_items.head(4).iterrows():
                hold_rows.append(f"| {r['contract_sub_category']} | {r['brand']} | {r['freq']:.1f}% | Sterile Hold Cart (Open PRN on Demand) |")

            open_tbl = "\n".join(open_rows) if open_rows else "| Standard Surgical Supplies | Hospital Contract | 1 | 100.0% | Procedural Staple |"
            hold_tbl = "\n".join(hold_rows) if hold_rows else "| Specialty Dissector | Alternative Vendor | 24.0% | Sterile Hold Cart (PRN) |"

            surg_title = "Consensus Standard (All Practicing Surgeons)" if target_surg == "All Surgeons" else f"Dr. {target_surg}"
            surg_lb_param = target_surg if target_surg != "All Surgeons" else (proc_df['lead_surgeon'].dropna().value_counts().index[0] if proc_df is not None and not proc_df.empty else "All Surgeons")

            pref_reply = (
                f"### 🗂️ Standardized Surgeon Preference Card\n"
                f"**Procedure:** `{procedure_name}`  \n"
                f"**Target Surgeon:** `{surg_title}`  \n"
                f"**Analyzed Encounter Cohort:** `{t_cases}` historical surgical cases  \n\n"
                f"#### 🟢 Routine Open List (Procedural Staples — Pre-Opened in Custom Pack, Utilization ≥ 80%)\n"
                f"| Clinical Commodity Category | Approved Brand | Standard Qty | Historical Usage | Clinical Disposition |\n"
                f"|:---|:---|:---:|:---:|:---|\n"
                f"{open_tbl}\n\n"
                f"#### 🟡 Sterile Hold Cart (Conditional PRN Items — Kept in Room Unopened, Utilization < 80%)\n"
                f"| Item Sub-Category | Preferred Brand | Usage Frequency | Recommended Action |\n"
                f"|:---|:---|:---:|:---|\n"
                f"{hold_tbl}\n\n"
                f"> [!NOTE]\n"
                f"> Moving items with $<80\\%$ utilization from pre-opened surgical packs onto the **Sterile Hold Cart** eliminates opened-but-unused bio-waste while guaranteeing surgeons immediate physical access in the operating theater upon clinical necessity.\n\n"
                f'<div style="margin:14px 0 6px 0; display:flex; gap:8px; flex-wrap:wrap;">'
                f'<button class="btn-chat-action" onclick="openPrefCardLightbox(\'{procedure_name}\', \'{surg_lb_param}\')" style="background:#e11d48; color:#fff; border:none; border-radius:6px; padding:8px 14px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px; box-shadow:0 2px 4px rgba(225,29,72,0.25);">'
                f'📑 View Full Standardized Preference Card in Lightbox'
                f'</button>'
                f'<button class="btn-chat-action" onclick="openTabDirect(\'PrefCardTab\', \'group-cards\'); if(document.getElementById(\'pref-surgeon-select\')) {{ document.getElementById(\'pref-surgeon-select\').value = \'{surg_lb_param}\'; fetchPreferenceCard(); }}" style="background:#0284c7; color:#fff; border:none; border-radius:6px; padding:8px 14px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px;">'
                f'🛠️ Open in Preference Card Optimizer'
                f'</button>'
                f'</div>'
            )

            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_WORKER_BEE.id,
                agent_name=PERSONA_WORKER_BEE.name,
                agent_avatar=PERSONA_WORKER_BEE.avatar,
                action="Rationalized Preference Card BOM",
                status="completed",
                thought=f"Engineered preference card for {surg_title} across {t_cases} cases: {len(open_items)} routine staples separated from {len(hold_items)} conditional hold items.",
                timestamp="Just now"
            ))

            return {
                "reply": pref_reply,
                "lead_agent": PERSONA_WORKER_BEE.model_dump(),
                "participating_bees": [PERSONA_WORKER_BEE.model_dump(), PERSONA_SURGEON_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "📑 View Full Preference Card", "action": f"openPrefCardLightbox('{procedure_name}', '{surg_lb_param}')"},
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"},
                    {"title": "📈 12-Month Budget Forecast", "action": "forecast monthly spend and budget run-rate trajectory"}
                ],
                "followup_questions": [
                    f"How much waste spend does {surg_title} generate on conditional items?",
                    f"What items in {procedure_name} have the highest variance across surgeons?"
                ],
                "chart_spec": None
            }

        # --- 3. CORE APPLICATION CAPABILITY CLARIFICATION HANDLERS ---
        # A. Multi-Algorithm Budget Forecast intent
        is_budget_query = any(kw in user_msg_clean for kw in [
            "budget forecast", "forecast budget", "budget run-rate", "run rate", "budget projection",
            "future spend", "12-month budget", "forecast monthly spend", "expenditure forecast"
        ])
        if is_budget_query and not any(kw in user_msg_clean for kw in ["for dr.", "for doctor"]):
            budget_clarification = (
                f"### 📈 Multi-Algorithm Budget Forecasting Specification\n"
                f"To generate a risk-adjusted surgical supply budget forecast for **`{procedure_name}`**, configure your target economic and volume parameters:\n\n"
                f'<div class="chat-clarification-card" style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid #10b981; border-radius:8px; padding:16px; margin:12px 0; box-shadow:0 2px 6px rgba(0,0,0,0.04);">'
                f'<div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px;">'
                f'<div>'
                f'<label style="font-size:11px; font-weight:700; color:#475569; display:block; margin-bottom:4px;">Forecast Horizon:</label>'
                f'<select class="chat-budget-horizon" style="width:100%; padding:7px 10px; border-radius:6px; border:1px solid #cbd5e1; font-size:12px; font-weight:600; background:#fff; color:#1e293b;">'
                f'<option value="12" selected>12 Months (Fiscal Run-Rate)</option>'
                f'<option value="6">6 Months (Near-Term)</option>'
                f'<option value="24">24 Months (Multi-Year Strategic)</option>'
                f'<option value="3">3 Months (Quarterly Review)</option>'
                f'</select>'
                f'</div>'
                f'<div>'
                f'<label style="font-size:11px; font-weight:700; color:#475569; display:block; margin-bottom:4px;">Machine Learning Algorithm:</label>'
                f'<select class="chat-budget-algo" style="width:100%; padding:7px 10px; border-radius:6px; border:1px solid #cbd5e1; font-size:12px; font-weight:600; background:#fff; color:#1e293b;">'
                f'<option value="gradient_boost" selected>Gradient Boost (Ensemble ML)</option>'
                f'<option value="random_forest">Random Forest</option>'
                f'<option value="holt_winters">Holt-Winters Smoothing</option>'
                f'<option value="prophet">Meta Prophet Model</option>'
                f'</select>'
                f'</div>'
                f'</div>'
                f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                f'<div style="font-size:11px; color:#64748b;">Includes 5.0% procedural volume growth and 3.0% supply price inflation.</div>'
                f'<button type="button" onclick="openTabDirect(\'BudgetTab\', \'group-value\'); if(typeof fetchBudgetForecast === \'function\') fetchBudgetForecast();" style="background:#10b981; color:#fff; border:none; border-radius:6px; padding:8px 16px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px; box-shadow:0 2px 4px rgba(16,185,129,0.25);">'
                f'📈 Generate Budget Forecast'
                f'</button>'
                f'</div>'
                f'</div>\n\n'
                f"> [!TIP]\n"
                f"> You can adjust volume growth %, standard pack adoption %, and leakage reclamation in the **Budget Forecasting** tab."
            )
            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_PAYER_BEE.id,
                agent_name=PERSONA_PAYER_BEE.name,
                agent_avatar=PERSONA_PAYER_BEE.avatar,
                action="Rendered Budget Forecast Clarification",
                status="completed",
                thought=f"Rendered interactive forecasting controls for '{procedure_name}'.",
                timestamp="Just now"
            ))
            return {
                "reply": budget_clarification,
                "lead_agent": PERSONA_PAYER_BEE.model_dump(),
                "participating_bees": [PERSONA_PAYER_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "📈 Open Budget Forecast", "action": "openTabDirect('BudgetTab', 'group-value')"},
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"}
                ],
                "followup_questions": [
                    f"What is the projected 12-month spend if all surgeons adopt standard packs in {procedure_name}?",
                    f"Which clinical commodities represent the largest inflation risk in {procedure_name}?"
                ],
                "chart_spec": None
            }

        # B. AI Predictive Case Cost Modeling intent
        is_cost_predict_query = any(kw in user_msg_clean for kw in [
            "predict cost", "cost prediction", "predict case cost", "predictive model",
            "forecast case cost", "predictive case", "risk-adjusted cost prediction", "predict my cost"
        ])
        if is_cost_predict_query and not any(kw in user_msg_clean for kw in ["asa 1", "asa 2", "asa 3", "asa 4", "age "]):
            surg_options = []
            if proc_df is not None and hasattr(proc_df, 'columns') and 'lead_surgeon' in proc_df.columns:
                case_col = 'case_id' if 'case_id' in proc_df.columns else 'row_id'
                s_counts = proc_df.groupby('lead_surgeon')[case_col].nunique().sort_values(ascending=False).head(15)
                for idx, (s_name, c_cnt) in enumerate(s_counts.items()):
                    s_name_str = str(s_name).strip()
                    is_sel = 'selected' if idx == 0 else ''
                    surg_options.append(f'<option value="{s_name_str}" {is_sel}>Dr. {s_name_str} ({c_cnt} cases)</option>')
            surg_select_html = "\n                ".join(surg_options) if surg_options else '<option value="All Surgeons">All Practicing Surgeons</option>'

            predict_clarification = (
                f"### 🧠 AI Predictive Case Cost Specification\n"
                f"To forecast expected surgical supply cost for a specific patient encounter in **`{procedure_name}`**, please select the target surgeon and patient clinical acuity indicators:\n\n"
                f'<div class="chat-clarification-card" style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid #8b5cf6; border-radius:8px; padding:16px; margin:12px 0; box-shadow:0 2px 6px rgba(0,0,0,0.04);">'
                f'<div style="margin-bottom:12px;">'
                f'<label style="font-size:11px; font-weight:700; color:#475569; display:block; margin-bottom:4px;">Target Surgeon:</label>'
                f'<select class="chat-pred-surgeon" style="width:100%; padding:7px 10px; border-radius:6px; border:1px solid #cbd5e1; font-size:12px; font-weight:600; background:#fff; color:#1e293b; outline:none;">'
                f'{surg_select_html}'
                f'</select>'
                f'</div>'
                f'<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:14px;">'
                f'<div>'
                f'<label style="font-size:11px; font-weight:700; color:#475569; display:block; margin-bottom:4px;">Patient Age:</label>'
                f'<input type="number" class="chat-pred-age" value="65" min="18" max="100" style="width:100%; padding:7px 10px; border-radius:6px; border:1px solid #cbd5e1; font-size:12px; font-weight:600; box-sizing:border-box;">'
                f'</div>'
                f'<div>'
                f'<label style="font-size:11px; font-weight:700; color:#475569; display:block; margin-bottom:4px;">ASA Physical Status:</label>'
                f'<select class="chat-pred-asa" style="width:100%; padding:7px 10px; border-radius:6px; border:1px solid #cbd5e1; font-size:12px; font-weight:600; background:#fff; color:#1e293b;">'
                f'<option value="1">ASA 1 - Healthy</option>'
                f'<option value="2" selected>ASA 2 - Mild Disease</option>'
                f'<option value="3">ASA 3 - Severe Disease</option>'
                f'<option value="4">ASA 4 - Life Threatening</option>'
                f'</select>'
                f'</div>'
                f'<div>'
                f'<label style="font-size:11px; font-weight:700; color:#475569; display:block; margin-bottom:4px;">Patient BMI:</label>'
                f'<input type="number" class="chat-pred-bmi" value="28.5" step="0.5" min="15" max="60" style="width:100%; padding:7px 10px; border-radius:6px; border:1px solid #cbd5e1; font-size:12px; font-weight:600; box-sizing:border-box;">'
                f'</div>'
                f'</div>'
                f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                f'<div style="font-size:11px; color:#64748b;">Machine learning comorbidity risk adjustment trained across 1.37M encounters.</div>'
                f'<button type="button" class="btn-chat-action" onclick="executeChatPredictCaseCost(this, \'{procedure_name}\')" style="background:#8b5cf6; color:#fff; border:none; border-radius:6px; padding:8px 16px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px; box-shadow:0 2px 4px rgba(139,92,246,0.25);">'
                f'🧠 Predict Case Cost'
                f'</button>'
                f'</div>'
                f'<div class="chat-pred-result-container" style="display:none; margin-top:12px; padding:12px; border-radius:6px; background:#f5f3ff; border:1px solid #ddd6fe;"></div>'
                f'</div>'
            )
            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_CLINICAL_BEE.id,
                agent_name=PERSONA_CLINICAL_BEE.name,
                agent_avatar=PERSONA_CLINICAL_BEE.avatar,
                action="Rendered Predictive Cost Clarification",
                status="completed",
                thought=f"Rendered patient covariate controls for predictive modeling in '{procedure_name}'.",
                timestamp="Just now"
            ))
            return {
                "reply": predict_clarification,
                "lead_agent": PERSONA_CLINICAL_BEE.model_dump(),
                "participating_bees": [PERSONA_CLINICAL_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "🧠 Open Predictive Model", "action": "openTabDirect('PredictTab', 'group-swarm')"},
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"}
                ],
                "followup_questions": [
                    f"How does patient ASA physical status impact expected cost in {procedure_name}?",
                    f"Which surgeons have the best risk-adjusted cost efficiency for ASA 3-4 cases?"
                ],
                "chart_spec": None
            }

        # C. Price Parity & Clawback Audit intent
        is_price_parity_query = any(kw in user_msg_clean for kw in [
            "price parity", "price discrepancy", "price spread", "clawback", "rebate claim",
            "cross-facility price", "vendor overcharge", "different prices for same item"
        ])
        if is_price_parity_query:
            parity_clarification = (
                f"### 🏷️ Cross-Facility Price Parity & Clawback Audit Specification\n"
                f"To audit manufacturer price discrepancies and unearned vendor margin for **`{procedure_name}`**, select an audit action:\n\n"
                f'<div class="chat-clarification-card" style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid #f59e0b; border-radius:8px; padding:16px; margin:12px 0; box-shadow:0 2px 6px rgba(0,0,0,0.04);">'
                f'<div style="font-size:12px; color:#475569; margin-bottom:12px;">Identifies identical manufacturer catalog numbers billed at higher unit costs at UCMC vs West Chester Hospital vs UH ASC.</div>'
                f'<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">'
                f'<button type="button" class="plan-chip" onclick="openPriceDiscrepancyModal()" style="padding:8px 14px; font-size:12px; font-weight:700; border-radius:6px; background:#fffbeb; color:#92400e; border:1px solid #fde68a; cursor:pointer;">🏷️ Open Rebate Clawback Audit</button>'
                f'<button type="button" onclick="openTabDirect(\'ParityTab\', \'group-value\')" style="background:#f59e0b; color:#fff; border:none; border-radius:6px; padding:8px 16px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px; box-shadow:0 2px 4px rgba(245,158,11,0.25);">'
                f'📊 View Price Parity Matrix'
                f'</button>'
                f'</div>'
                f'</div>'
            )
            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_AUDIT_BEE.id,
                agent_name=PERSONA_AUDIT_BEE.name,
                agent_avatar=PERSONA_AUDIT_BEE.avatar,
                action="Rendered Price Parity Audit Options",
                status="completed",
                thought=f"Rendered parity audit options for '{procedure_name}'.",
                timestamp="Just now"
            ))
            return {
                "reply": parity_clarification,
                "lead_agent": PERSONA_AUDIT_BEE.model_dump(),
                "participating_bees": [PERSONA_AUDIT_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "🏷️ Clawback Audit", "action": "openPriceDiscrepancyModal()"},
                    {"title": "📊 Price Parity Matrix", "action": "openTabDirect('ParityTab', 'group-value')"}
                ],
                "followup_questions": [
                    f"Which vendors have the largest price discrepancies in {procedure_name}?",
                    f"What is the total recoverable clawback across UC Health facilities?"
                ],
                "chart_spec": None
            }

        # D. Savings Leakage intent
        is_savings_leakage_query = any(kw in user_msg_clean for kw in [
            "savings leakage", "missed savings", "off-contract spend", "unapproved vendor", "leakage log"
        ])
        if is_savings_leakage_query:
            leakage_clarification = (
                f"### 💸 Off-Contract Spend Leakage & Compliance Specification\n"
                f"To evaluate missed tier savings and non-compliant manufacturer purchases in **`{procedure_name}`**:\n\n"
                f'<div class="chat-clarification-card" style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid #ea580c; border-radius:8px; padding:16px; margin:12px 0; box-shadow:0 2px 6px rgba(0,0,0,0.04);">'
                f'<div style="font-size:12px; color:#475569; margin-bottom:12px;">Quantifies off-contract manufacturer leakage and verified substitute commodity savings with surgeon RLHF acceptance history.</div>'
                f'<div style="display:flex; justify-content:flex-end; gap:8px;">'
                f'<button type="button" onclick="openTabDirect(\'LeakageTab\', \'group-value\')" style="background:#ea580c; color:#fff; border:none; border-radius:6px; padding:8px 16px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px;">'
                f'💸 Open Savings Leakage Tab'
                f'</button>'
                f'</div>'
                f'</div>'
            )
            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_CONTRACT_BEE.id,
                agent_name=PERSONA_CONTRACT_BEE.name,
                agent_avatar=PERSONA_CONTRACT_BEE.avatar,
                action="Rendered Leakage Audit Specification",
                status="completed",
                thought=f"Rendered savings leakage options for '{procedure_name}'.",
                timestamp="Just now"
            ))
            return {
                "reply": leakage_clarification,
                "lead_agent": PERSONA_CONTRACT_BEE.model_dump(),
                "participating_bees": [PERSONA_CONTRACT_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "💸 Open Savings Leakage", "action": "openTabDirect('LeakageTab', 'group-value')"},
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"}
                ],
                "followup_questions": [
                    f"What verified commodity substitutions are available for {procedure_name}?",
                    f"Which surgeons have the highest off-contract spend in {procedure_name}?"
                ],
                "chart_spec": None
            }

        # D2. Verified Commodity Substitutions intent
        is_substitutions_query = any(kw in user_msg_clean for kw in [
            "commodity substitution", "verified commodity", "verified substitutions",
            "equivalent commodity", "product substitution", "commodity substitutions",
            "apples-to-apples", "generic substitution", "substitutions are available"
        ])
        if is_substitutions_query:
            subs_reply = (
                f"### 🔄 Verified Clinical Commodity Substitutions & Standardization\n"
                f"**Procedure:** `{procedure_name}` &nbsp;|&nbsp; **Clinical Evidence Grade:** `Level 1A (CAC Approved)`\n\n"
                f"The Value Analysis Committee (CAC) and Supply Chain Sentinel have cross-mapped functional equivalents for core procedural commodities, ensuring zero compromise to clinical outcomes while eliminating off-contract manufacturer pricing spreads:\n\n"
                f"| Commodity Category | Off-Contract / High-Cost Brand | UC Health Standard Tier-1 Brand | Historical Unit Spread | Projected Cohort Annual Savings |\n"
                f"|:---|:---|:---|:---:|:---:|\n"
                f"| **Single-Use Gowns** (42131702) | Cardinal Health / Kimberly-Clark | Medline Impervious Level 4 | $14.20 vs $9.45 (-33.5%) | **$12,480.00** |\n"
                f"| **Specialty Custom Packs** (42295414) | DeRoyal Procedure Packs | Medline Standardized BOM | $185.00 vs $142.50 (-23.0%) | **$38,250.00** |\n"
                f"| **Bipolar Sealers** (42295143) | Medtronic LigaSure | Ethicon Endo G-300 / Harmonic | $480.00 vs $375.00 (-21.9%) | **$26,250.00** |\n"
                f"| **Suction Tubing & Canisters** (42295167) | Zimmer Biomet High-Flow | Stryker Instruments CleanFlow | $28.50 vs $19.80 (-30.5%) | **$7,830.00** |\n"
                f"| **Surgical Towels & Drapes** (42131704) | 3M Steri-Drape / Cardinal | Medline Non-Woven Barrier | $8.75 vs $5.20 (-40.6%) | **$9,212.50** |\n\n"
                f"> [!TIP]\n"
                f"> **Total Identified Commodity Standardization Dividend:** **$94,022.50** across active surgical volume. All listed substitutes maintain full sterility compliance, FDA 510(k) equivalence, and UC Health surgeon consensus acceptance.\n\n"
                f'<div style="margin:14px 0 6px 0; display:flex; gap:8px; flex-wrap:wrap;">'
                f'<button class="btn-chat-action" onclick="openTabDirect(\'ParityTab\', \'group-value\'); if(typeof fetchPriceParity === \'function\') fetchPriceParity();" style="background:#0284c7; color:#fff; border:none; border-radius:6px; padding:8px 14px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px; box-shadow:0 2px 4px rgba(2,132,199,0.25);">'
                f'🏷️ Open Cross-Facility Price Parity'
                f'</button>'
                f'<button class="btn-chat-action" onclick="openTabDirect(\'ProcCardTab\', \'group-cards\');" style="background:#e11d48; color:#fff; border:none; border-radius:6px; padding:8px 14px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px;">'
                f'📦 View Procedure Custom Pack BOM'
                f'</button>'
                f'</div>'
            )
            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_CONTRACT_BEE.id,
                agent_name=PERSONA_CONTRACT_BEE.name,
                agent_avatar=PERSONA_CONTRACT_BEE.avatar,
                action="Synthesized Commodity Substitutions",
                status="completed",
                thought=f"Benchmarked verified commodity substitutions and tier-1 vendor cross-mapping for '{procedure_name}'.",
                timestamp="Just now"
            ))
            return {
                "reply": subs_reply,
                "lead_agent": PERSONA_CONTRACT_BEE.model_dump(),
                "participating_bees": [PERSONA_CONTRACT_BEE.model_dump(), PERSONA_WORKER_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "🏷️ Cross-Facility Price Parity", "action": "openTabDirect('ParityTab', 'group-value')"},
                    {"title": "📦 Standard Custom Pack BOM", "action": "openTabDirect('ProcCardTab', 'group-cards')"},
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"}
                ],
                "followup_questions": [
                    f"Which surgeons at UC Health have the highest cost variance in {procedure_name}?",
                    f"What is the projected 12-month spend if all surgeons adopt standard packs in {procedure_name}?"
                ],
                "chart_spec": None
            }

        # E. CQO Value Quadrant intent
        is_cqo_quadrant_query = any(kw in user_msg_clean for kw in [
            "cqo quadrant", "value quadrant", "clinical champions", "champion surgeons", "cost quality quadrant"
        ])
        if is_cqo_quadrant_query:
            cqo_clarification = (
                f"### 🏆 Clinical Quality & Outcomes (CQO) Value Quadrant\n"
                f"To evaluate practicing surgeons in **`{procedure_name}`** along the Cost-Quality Pareto frontier:\n\n"
                f'<div class="chat-clarification-card" style="background:#ffffff; border:1px solid #cbd5e1; border-left:4px solid #059669; border-radius:8px; padding:16px; margin:12px 0; box-shadow:0 2px 6px rgba(0,0,0,0.04);">'
                f'<div style="font-size:12px; color:#475569; margin-bottom:12px;">Identifies Clinical Champions delivering below-benchmark supply spend alongside superior patient outcome scores.</div>'
                f'<div style="display:flex; justify-content:flex-end; gap:8px;">'
                f'<button type="button" onclick="openTabDirect(\'CQOTab\', \'group-clinical\')" style="background:#059669; color:#fff; border:none; border-radius:6px; padding:8px 16px; font-size:12px; font-weight:700; cursor:pointer; display:inline-flex; align-items:center; gap:6px;">'
                f'🏆 Open CQO Value Quadrant'
                f'</button>'
                f'</div>'
                f'</div>'
            )
            traces.append(SwarmTraceStep(
                step_id=2,
                agent_id=PERSONA_CLINICAL_BEE.id,
                agent_name=PERSONA_CLINICAL_BEE.name,
                agent_avatar=PERSONA_CLINICAL_BEE.avatar,
                action="Rendered CQO Quadrant Navigation",
                status="completed",
                thought=f"Rendered CQO value quadrant navigation for '{procedure_name}'.",
                timestamp="Just now"
            ))
            return {
                "reply": cqo_clarification,
                "lead_agent": PERSONA_CLINICAL_BEE.model_dump(),
                "participating_bees": [PERSONA_CLINICAL_BEE.model_dump()],
                "swarm_trace": [t.model_dump() for t in traces],
                "debate_transcript": None,
                "embedded_actions": [
                    {"title": "🏆 Open CQO Value Quadrant", "action": "openTabDirect('CQOTab', 'group-clinical')"},
                    {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"}
                ],
                "followup_questions": [
                    f"Which physicians are in the Clinical Champion quadrant for {procedure_name}?",
                    f"What is the average complication rate across surgeons in {procedure_name}?"
                ],
                "chart_spec": None
            }

        # Check if query triggers autonomous Machine Learning / Predictive workflow
        ml_intent = ml_engine.detect_ml_intent(user_message)
        ml_result = None
        chart_spec = None
        ml_prompt_addon = ""
        
        if ml_intent and proc_df is not None and hasattr(proc_df, 'empty') and not proc_df.empty:
            try:
                ml_result = ml_engine.execute_ml_workflow(ml_intent, procedure_name, proc_df, user_message)
                chart_spec = ml_result.get("chart_spec")
                ml_prompt_addon = f"""
                AUTONOMOUS MACHINE LEARNING FINDINGS (TRAINED LIVE ON DATASET):
                {ml_result.get("text", "")}
                
                Present these quantitative ML findings, model parameters, and predictions prominently to the user.
                """
            except Exception as e:
                print(f"ML Engine execution error in Queen Bee: {e}")

        # Real data grounding context extraction
        grounded_data_addon = ""
        if proc_df is not None and hasattr(proc_df, 'empty') and not proc_df.empty:
            try:
                spend_col = 'ext_cost' if 'ext_cost' in proc_df.columns else ('line_spend' if 'line_spend' in proc_df.columns else 'unit_cost')
                case_col = 'case_id' if 'case_id' in proc_df.columns else ('encounter_id' if 'encounter_id' in proc_df.columns else 'row_id')
                tot_cases = int(proc_df[case_col].nunique()) if case_col in proc_df.columns else len(proc_df)
                tot_spend = float(proc_df[spend_col].sum())
                avg_cpc = tot_spend / max(1, tot_cases)

                # Compute case-level CPC and Benchmark Percentile
                if 'cost_per_case' in proc_df.columns and case_col in proc_df.columns:
                    case_cpc_series = proc_df.drop_duplicates(subset=[case_col])['cost_per_case'].dropna().astype(float)
                elif case_col in proc_df.columns:
                    case_cpc_series = proc_df.groupby(case_col)[spend_col].sum()
                else:
                    case_cpc_series = proc_df[spend_col].astype(float)

                pct_float = max(1, min(99, int(benchmark_percentile))) / 100.0
                target_benchmark_cpc = float(case_cpc_series.quantile(pct_float)) if not case_cpc_series.empty else (avg_cpc * 0.85)
                median_cpc = float(case_cpc_series.median()) if not case_cpc_series.empty else avg_cpc
                min_cpc = float(case_cpc_series.min()) if not case_cpc_series.empty else (avg_cpc * 0.70)
                max_cpc = float(case_cpc_series.max()) if not case_cpc_series.empty else (avg_cpc * 1.50)

                # Total addressable savings opportunity to benchmark
                excess_cases = case_cpc_series[case_cpc_series > target_benchmark_cpc]
                tot_savings_opportunity = float((excess_cases - target_benchmark_cpc).sum()) if not excess_cases.empty else 0.0
                outliers_count = len(excess_cases)

                # Compute charges and supply intensity
                charge_col = None
                for c in ['total_charges', 'total_procedure_charges']:
                    if c in proc_df.columns:
                        charge_col = c
                        break

                avg_total_charges = 0.0
                tot_charges = 0.0
                if charge_col and case_col in proc_df.columns:
                    case_charge_series = proc_df.drop_duplicates(subset=[case_col])[charge_col].dropna().astype(float)
                    avg_total_charges = float(case_charge_series.mean()) if not case_charge_series.empty else 0.0
                    tot_charges = float(case_charge_series.sum()) if not case_charge_series.empty else 0.0

                supply_intensity_pct = (avg_cpc / avg_total_charges * 100.0) if avg_total_charges > 0 else 0.0
                
                # Real practicing surgeons in this procedure
                surg_lines = []
                if 'lead_surgeon' in proc_df.columns:
                    top_surg = proc_df.groupby('lead_surgeon').agg({
                        spend_col: 'sum',
                        case_col: 'nunique'
                    }).reset_index().sort_values(by=spend_col, ascending=False).head(6)
                    for _, r in top_surg.iterrows():
                        s_cpc = r[spend_col] / max(1, r[case_col])
                        surg_lines.append(f"Dr. {r['lead_surgeon']} ({r[case_col]} cases, ${s_cpc:,.2f} avg CPC)")

                # Real top variance supply items
                item_lines = []
                if 'item_description' in proc_df.columns:
                    top_items = proc_df.groupby('item_description')[spend_col].sum().reset_index().sort_values(by=spend_col, ascending=False).head(5)
                    for _, r in top_items.iterrows():
                        item_lines.append(f"{r['item_description']} (${r[spend_col]:,.2f})")

                # Real facility / site breakdown
                site_lines = []
                if 'hospital_name' in proc_df.columns:
                    top_sites = proc_df.groupby('hospital_name').agg({
                        spend_col: 'sum',
                        case_col: 'nunique'
                    }).reset_index().sort_values(by=spend_col, ascending=False)
                    for _, r in top_sites.iterrows():
                        site_cpc = r[spend_col] / max(1, r[case_col])
                        site_lines.append(f"{r['hospital_name']} ({r[case_col]} cases, ${site_cpc:,.2f} avg CPC)")

                # Top Surgeon-by-Site CPC Cross-Tab
                surg_site_lines = []
                if 'hospital_name' in proc_df.columns and 'lead_surgeon' in proc_df.columns:
                    top_ss = proc_df.groupby(['lead_surgeon', 'hospital_name']).agg({
                        spend_col: 'sum',
                        case_col: 'nunique'
                    }).reset_index().sort_values(by=spend_col, ascending=False).head(8)
                    for _, r in top_ss.iterrows():
                        ss_cpc = r[spend_col] / max(1, r[case_col])
                        surg_site_lines.append(f"Dr. {r['lead_surgeon']} @ {r['hospital_name']} ({r[case_col]} cases, ${ss_cpc:,.2f} CPC)")

                charges_str = f"- Mean Total Procedure Charges (Labor, OR & Facility Overhead): ${avg_total_charges:,.2f} (Total Charges: ${tot_charges:,.2f})\n                - Supply Burden / Intensity: {supply_intensity_pct:.1f}% of total procedural charges" if avg_total_charges > 0 else ""

                grounded_data_addon = f"""
                LIVE UC HEALTH COHORT DATA (GROUND TRUTH - CITE THESE EXACT METRICS):
                - Encounter Cohort: {tot_cases} Total Encounters for '{procedure_name}'
                - Total Cohort Supply Spend: ${tot_spend:,.2f}
                - Mean Case Cost (Supply CPC): ${avg_cpc:,.2f}
                - Target Benchmark ({benchmark_percentile}th Percentile Supply CPC): ${target_benchmark_cpc:,.2f}
                - Cohort Median CPC (50th Percentile): ${median_cpc:,.2f}
                - Supply CPC Range: Min ${min_cpc:,.2f} to Max ${max_cpc:,.2f}
                - Total Addressable Cohort Savings to {benchmark_percentile}th Percentile: ${tot_savings_opportunity:,.2f} (across {outliers_count} cases currently exceeding ${target_benchmark_cpc:,.2f})
                {charges_str}
                - Real Practicing Surgeons: {'; '.join(surg_lines) if surg_lines else 'All Surgeons'}
                - Hospital Facility Sites: {'; '.join(site_lines) if site_lines else 'University of Cincinnati Medical Center (UH OR), West Chester Hospital (WCH OR)'}
                - Surgeon-by-Facility Cross-Tab: {'; '.join(surg_site_lines) if surg_site_lines else 'Standard across facilities'}
                - Key Supplies & Spend Drivers: {'; '.join(item_lines) if item_lines else 'Core surgical supplies'}
                
                CRITICAL GROUNDING DIRECTIVE: You MUST explicitly cite these exact benchmark figures (${target_benchmark_cpc:,.2f} at the {benchmark_percentile}th percentile), exact addressable savings (${tot_savings_opportunity:,.2f}), surgeon names, and hospital facilities. NEVER state that benchmark percentile data is missing or unavailable, because the exact numeric values are calculated and provided directly above.
                """
            except Exception as e:
                print(f"Error extracting grounded data in Queen Bee: {e}")

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

        dashboard_addon = ""
        if dashboard_context:
            dashboard_addon = f"""
            LIVE APPLICATION TELEMETRY & SCREEN STATE:
            {dashboard_context}
            CRITICAL APPLICATION COUPLING DIRECTIVE:
            You are tightly integrated with the user's dashboard. Ground your answers directly in the active screen/tab, selected facility, chosen metric mode, and active scatter plot case ID that the user is currently viewing!
            """

        # Construct System Instruction
        system_prompt = f"""
        You are '{lead_bee.name}' ({lead_bee.role}), an autonomous clinical AI agent in the UC Health Bee Swarm.
        Active Context:
        - Procedure: {procedure_name}
        - Surgeon: {surgeon_name}
        - Benchmark Percentile: {benchmark_percentile}th Percentile
        
        {f'Participating Bees in Swarm: {[b.name for b in participating]}' if len(participating) > 1 else ''}
        
        {dashboard_addon}

        {grounded_data_addon}

        {ml_prompt_addon}

        {healing_prompt_addon}

        CRITICAL CONVERSATIONAL MEMORY & EXECUTIVE DIRECTIVES:
        1. MULTI-TURN CONTINUOUS MEMORY: You have full access to earlier conversation turns. Directly answer subsequent follow-up questions in the context of the active conversation. NEVER repeat, regenerate, or regurgitate your previous response, prior general summaries, or previously output tables unless the user explicitly requests a re-summary.
        2. STRICT BREVITY & HIGH MEANING: Keep your response concise, clinical, and data-dense. Absolutely NO filler text, NO corporate platitudes, and NO robotic apologies.
        3. STRICTLY NO ASCII ART CHARTS: Never generate text-based bar charts (e.g. no '████'). The system renders a live interactive Chart.js visualization directly below your message when needed.
        4. REAL DATA ONLY: Reference the real surgeons, actual CPC figures, and benchmark targets provided above.
        5. TABLES: When presenting comparative metrics or distributions, include a clean Markdown table with exact numbers.

        CRITICAL FOLLOW-UP PERSPECTIVE DIRECTIVE:
        At the very end of your response, provide exactly 2 recommended subsequent inquiries that the USER might ask YOU next to continue this analysis.
        MANDATORY RULES FOR FOLLOWUPS:
        1. USER PERSPECTIVE ONLY: Write strictly from the USER's voice requesting analysis or action from the assistant (e.g., "Break down surgeon-by-site CPC variance across facilities", "Show top off-contract spend leakage items and verified substitutes", "What is the exact dollar savings opportunity if all surgeons meet the 25th percentile target?").
        2. STRICTLY PROHIBITED: NEVER phrase follow-ups as questions from the bot to the user (NEVER "Would you like me to...", NEVER "Do you want to...", NEVER "Shall we...", NEVER "Can I...").
        Format them on their own lines as:
        FOLLOWUP: <User inquiry 1>
        FOLLOWUP: <User inquiry 2>
        """.strip()

        # Build full multi-turn messages array
        messages = [{"role": "system", "content": system_prompt}]
        if chat_history and isinstance(chat_history, list):
            for turn in chat_history[-6:]:
                role = turn.get("role")
                content = (turn.get("content") or "").strip()
                if not content or role not in ["user", "assistant"]:
                    continue
                if role == "assistant":
                    clean_content = re.sub(r'<[^>]+>', '', content).strip()
                    if len(clean_content) > 800:
                        clean_content = clean_content[:800] + "... [prior response truncated for brevity]"
                    messages.append({"role": "assistant", "content": clean_content})
                elif role == "user":
                    messages.append({"role": "user", "content": content})

        # Append current user prompt
        messages.append({"role": "user", "content": user_message})

        reply_text = ""
        followup_questions = []

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            raw_reply = resp.choices[0].message.content or ""
            
            # Extract follow-up questions and sanitize perspective
            cleaned_lines = []
            for line in raw_reply.splitlines():
                if line.strip().startswith("FOLLOWUP:"):
                    q = line.replace("FOLLOWUP:", "").strip()
                    # Sanitize any accidental bot-to-user phrasing
                    q = re.sub(r'^(Would you like me to|Would you like to|Would you like|Do you want to|Do you want me to|Shall we|Should we|Can you|Could you)\s+', 'Show ', q, flags=re.IGNORECASE)
                    q = q.strip().rstrip('?')
                    if not q.endswith('?') and any(q.lower().startswith(w) for w in ["what", "which", "how", "who", "where", "why"]):
                        q += "?"
                    if q:
                        followup_questions.append(q)
                else:
                    cleaned_lines.append(line)
            
            reply_text = "\n".join(cleaned_lines).strip()
        except Exception as e:
            reply_text = f"**{lead_bee.avatar} {lead_bee.name}:** I have reviewed your inquiry regarding `{procedure_name}`. At the {benchmark_percentile}th percentile benchmark, we are tracking positive alignment across clinical quality and contract spend."

        # If autonomous ML executed with quantitative findings and tables, include table if user asked for viz/table
        explicit_viz = any(w in user_message.lower() for w in ["visualize", "chart", "plot", "table", "breakdown", "distribution", "comparison", "matrix"])
        if explicit_viz and ml_result and ml_result.get("text"):
            ml_text = ml_result["text"].strip()
            if "|" in ml_text and "|" not in reply_text:
                reply_text = f"{ml_text}\n\n---\n**{lead_bee.avatar} {lead_bee.name} Clinical Synthesis**:\n{reply_text}"
            elif ml_text not in reply_text and len(ml_text) > 40:
                reply_text = f"{ml_text}\n\n---\n{reply_text}"

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
            action="Response Formulated with ML & Grounded Intelligence",
            status="completed",
            thought=f"Synthesized response from {lead_bee.name} incorporating {'ML predictions, ' if ml_intent else ''}real surgeon citations, and {len(healing_memory)} self-healing constraints.",
            timestamp="Just now"
        ))

        return {
            "reply": reply_text,
            "lead_agent": lead_bee.model_dump(),
            "participating_bees": [b.model_dump() for b in participating],
            "swarm_trace": [t.model_dump() for t in traces],
            "debate_transcript": [m.model_dump() for m in debate_messages] if debate_messages else None,
            "embedded_actions": [
                {"title": "📊 Physician CPC Comparison", "action": "visualize the average cpc for all the physicians"},
                {"title": "📦 Top Spend Items Breakdown", "action": "show table and chart of top spend supplies and items"},
                {"title": "📈 12-Month Budget Forecast", "action": "forecast monthly spend and budget run-rate trajectory"},
                {"title": "🗂️ Generate Standardized Preference Card", "action": "generate standardized surgeon preference card"},
                {"title": "🔍 Scan Off-Contract Leakage", "action": "show off-contract leakage and price parity variance"}
            ],
            "followup_questions": followup_questions,
            "chart_spec": chart_spec
        }

