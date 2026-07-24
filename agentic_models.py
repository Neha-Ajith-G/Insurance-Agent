"""
agentic_models: Pydantic models for the AgenticIE (Planner, Executor, Responder) system.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class PlannerToolName(str, Enum):
    RUN_OCR = "RUN_OCR"
    CLASSIFY_DOCUMENT = "CLASSIFY_DOCUMENT"
    EXTRACT_FIXED_FIELDS = "EXTRACT_FIXED_FIELDS"
    EXTRACT_FIXED_SCHEMA = "EXTRACT_FIXED_SCHEMA"  # Legacy combined alias
    EXTRACT_OPEN_SCHEMA = "EXTRACT_OPEN_SCHEMA"
    RETRIEVE_CHUNKS = "RETRIEVE_CHUNKS"
    ANSWER_QUESTION = "ANSWER_QUESTION"
    FINISH = "FINISH"


class PlannerAction(BaseModel):
    tool: PlannerToolName
    tool_args: dict = Field(default_factory=dict)
    rationale: str = ""


class ExecutionStep(BaseModel):
    step_number: int
    action: PlannerAction
    result_summary: str
    token_cost: int = 0
    inference_time: float = 0.0
    timestamp: str = ""


class OpenSchemaResult(BaseModel):
    summary: str = ""
    discovered_keys: dict = Field(default_factory=dict)
    key_value_pairs: dict = Field(default_factory=dict)
    tables: list[dict] = Field(default_factory=list)


class VerificationResult(BaseModel):
    passed: bool = True
    score: float = 1.0
    reasoning: str = ""
    evidence_quotes: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)


class AgentState(BaseModel):
    file_path: str = ""
    filename: str = ""
    file_type: str = ""
    needs_ocr: bool = False
    raw_text: str = ""
    chunks: list[str] = Field(default_factory=list)
    deterministic_baseline: dict | None = None
    category: str | None = None
    document_type: str | None = None
    fixed_schema: dict | None = None
    open_schema: OpenSchemaResult | None = None
    user_query: str | None = None
    qa_answer: str | None = None
    history: list[ExecutionStep] = Field(default_factory=list)
    verification: VerificationResult | None = None
    is_complete: bool = False
    total_tokens: int = 0
    total_inference_time: float = 0.0
    total_elapsed: float = 0.0
