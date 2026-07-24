"""
Pydantic response models for the document storage endpoints
(GET /documents, GET /documents/{id}). Kept separate from docmodels.py,
which owns the LLM pipeline's own shapes (ClassificationResult,
ExtractionResult, etc.) -- these are specifically what the API returns
to a caller listing/inspecting their stored documents.
"""
from __future__ import annotations

from pydantic import BaseModel


class DocumentSummary(BaseModel):
    """One row in the GET /documents list -- enough to identify and
    pick a document, without the full extraction payload."""

    id: int
    filename: str
    category: str
    document_type: str
    confidence: float | None = None
    total_tokens: int
    created_at: str


class DocumentDetail(DocumentSummary):
    """Full stored detail for GET /documents/{id}."""

    reasoning: str | None = None
    extraction_fields: dict
    token_stats: list[dict]
    total_input_tokens: int
    total_output_tokens: int
    total_inference_time: float
    pipeline_elapsed: float