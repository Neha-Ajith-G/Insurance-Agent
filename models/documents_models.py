"""
Pydantic response models for the document storage endpoints (GET /documents, GET /documents/{id}).
"""
from __future__ import annotations

from pydantic import BaseModel


class DocumentSummary(BaseModel):
    id: int
    filename: str
    category: str
    document_type: str
    confidence: float | None = None
    total_tokens: int
    created_at: str


class DocumentDetail(DocumentSummary):
    reasoning: str | None = None
    extraction_fields: dict
    token_stats: list[dict]
    total_input_tokens: int
    total_output_tokens: int
    total_inference_time: float
    pipeline_elapsed: float
