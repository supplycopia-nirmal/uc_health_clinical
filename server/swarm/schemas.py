from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class AgentPersona(BaseModel):
    id: str
    name: str
    role: str
    avatar: str
    specialty: str
    description: str

class SwarmTraceStep(BaseModel):
    step_id: int
    agent_id: str
    agent_name: str
    agent_avatar: str
    action: str
    status: str  # "running", "completed", "warning", "info"
    thought: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str

class SwarmMessage(BaseModel):
    sender_id: str
    sender_name: str
    sender_avatar: str
    recipient_id: str
    content: str
    message_type: str = "analysis"  # "proposal", "critique", "consensus", "summary"

class SwarmAuditRequest(BaseModel):
    procedure_name: str
    surgeon_name: Optional[str] = "All Surgeons"
    benchmark_percentile: int = 25  # 10, 25, 50
    user_query: Optional[str] = None
    active_bees: Optional[List[str]] = None

class CQOMetrics(BaseModel):
    total_encounters: int
    outcome_score: float
    ssi_rate: float
    transfusion_rate: float
    readmission_rate: float
    mortality_rate: float
    alos_days: float
    med_supply_cost: float
    implant_cost: float
    contracted_spend: float
    off_contract_spend: float
    potential_savings: float
    savings_pct: float

class SwarmAuditResponse(BaseModel):
    procedure_name: str
    executive_summary: str
    cqo_metrics: CQOMetrics
    agent_traces: List[SwarmTraceStep]
    agent_findings: Dict[str, Any]
    verified_substitutions: List[Dict[str, Any]]
    off_contract_leakage: List[Dict[str, Any]]
    pack_recommendations: List[Dict[str, Any]]
    surgeon_scorecard: Dict[str, Any]
    suggested_actions: List[Dict[str, str]]

class SwarmChatRequest(BaseModel):
    message: str
    procedure_name: Optional[str] = "COLONOSCOPY WITH OR WITHOUT BIOPSY"
    surgeon_name: Optional[str] = "All Surgeons"
    benchmark_percentile: int = 25
    chat_history: Optional[List[Dict[str, str]]] = []
    preferred_bee: Optional[str] = "queen"  # "queen", "clinical", "scout", "contract", "worker", "surgeon"

class SwarmChatResponse(BaseModel):
    reply: str
    lead_agent: AgentPersona
    participating_bees: List[AgentPersona]
    swarm_trace: List[SwarmTraceStep]
    debate_transcript: Optional[List[SwarmMessage]] = None
    embedded_actions: Optional[List[Dict[str, str]]] = None
    followup_questions: Optional[List[str]] = None
