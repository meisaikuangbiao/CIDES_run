from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FlowNode(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    desc: str
    required_if: Optional[str] = None
    must_say: list[str] = Field(default_factory=list)
    must_not_say_before: list[str] = Field(default_factory=list)


class FlowEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")
    source: str
    target: str
    condition: str


class KnowledgePoint(BaseModel):
    model_config = ConfigDict(extra="ignore")
    topic: str
    triggers: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)


class HardConstraints(BaseModel):
    model_config = ConfigDict(extra="ignore")
    max_chars_per_reply: Optional[int] = None
    forbidden_words: list[str] = Field(default_factory=list)
    no_discount_promise: bool = False
    required_out_of_scope_reply: Optional[str] = None
    opening_keywords: list[str] = Field(default_factory=list)
    required_replies: list[dict[str, Any]] = Field(default_factory=list)


class InstructionConstraints(BaseModel):
    model_config = ConfigDict(extra="ignore")
    hard: HardConstraints = Field(default_factory=HardConstraints)
    soft: list[str] = Field(default_factory=list)
    termination: list[str] = Field(default_factory=list)


class InstructionSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    role: str
    task: str
    opening_line_template: str = ""
    variables: list[str] = Field(default_factory=list)
    flow_nodes: list[FlowNode] = Field(default_factory=list)
    flow_edges: list[FlowEdge] = Field(default_factory=list)
    knowledge: list[KnowledgePoint] = Field(default_factory=list)
    constraints: InstructionConstraints = Field(default_factory=InstructionConstraints)
    raw_markdown: str = ""


class ScenarioSpec(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    user_goal: str
    behaviour: str
    target_nodes: list[str] = Field(default_factory=list)
    target_constraints: list[str] = Field(default_factory=list)
    target_knowledge: list[str] = Field(default_factory=list)
    adversarial_events: list[str] = Field(default_factory=list)
    required_user_signals: list[str] = Field(default_factory=list)
    expected_termination: Optional[str] = None
    forbid_reveal: bool = True


class TurnRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: int
    role: Literal["user", "assistant", "system"]
    content: str
    chars: int
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    model: Optional[str] = None
    temperature: Optional[float] = None
    error: Optional[str] = None


class DialogueTrace(BaseModel):
    model_config = ConfigDict(extra="ignore")
    run_id: str
    case_id: str
    instruction_id: str
    scenario_id: str
    variables: dict[str, str] = Field(default_factory=dict)
    turns: list[TurnRecord] = Field(default_factory=list)
    terminated_by: Optional[str] = None
    seed: Optional[int] = None
    sut_model: Optional[str] = None
    user_sim_model: Optional[str] = None
    created_at: Optional[str] = None
    session_index: int = 1
    max_sessions: int = 1
    prior_memory: Optional[str] = None
    prior_sessions: list[dict[str, Any]] = Field(default_factory=list)


class ScoreDetail(BaseModel):
    model_config = ConfigDict(extra="ignore")
    criterion_id: str
    label: str
    passed: bool
    deduction: float = 0.0
    turn_ids: list[int] = Field(default_factory=list)
    evidence_quote: str = ""
    rationale: str = ""
    confidence: Optional[float] = None
    disagreement: Optional[float] = None


class DimensionScore(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    score: float
    raw_score: Optional[float] = None
    weight: float = 0.0
    confidence: Optional[float] = None
    source: str = "unknown"
    warnings: list[str] = Field(default_factory=list)
    details: list[ScoreDetail] = Field(default_factory=list)


class CaseReport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    trace: DialogueTrace
    dimensions: list[DimensionScore] = Field(default_factory=list)
    weighted_total: float = 0.0
    passed: bool = False
    fail_reasons: list[str] = Field(default_factory=list)
    sessions_used: int = 1
    confidence_interval: Optional[tuple[float, float]] = None
    failure_attribution: list[dict[str, Any]] = Field(default_factory=list)


class RunReport(BaseModel):
    model_config = ConfigDict(extra="ignore")
    run_id: str
    created_at: str
    config: dict[str, Any] = Field(default_factory=dict)
    cases: list[CaseReport] = Field(default_factory=list)
    aggregate: dict[str, Any] = Field(default_factory=dict)
