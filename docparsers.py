"""
DocParsers: turns raw LLM text replies into validated Pydantic models.
"""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

from docmodels import (
    DEFAULT_EXTRACTION_MODEL,
    EXTRACTION_MODEL_MAP,
    ClassificationResult,
    DocumentCategory,
    DocumentType,
    ExtractionResult,
)


class DocParserError(Exception):
    """Raised when an LLM reply can't be parsed into the expected model."""


def _snippet(text: str, limit: int = 200) -> str:
    text = text.strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _extract_json(raw: str) -> dict:
    """
    Pull a JSON object out of a raw LLM reply.

    Handles the common cases: fenced ```json blocks, and stray prose or
    reasoning the model tacked on before/after the object despite being
    told to return JSON only.
    """
    text = raw.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    if not text.startswith("{"):
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DocParserError(
            f"Invalid JSON in LLM reply ({e}). First 200 chars: {_snippet(raw)}"
        ) from e

    if not isinstance(data, dict):
        raise DocParserError(
            f"Expected a JSON object, got {type(data).__name__}. "
            f"First 200 chars: {_snippet(raw)}"
        )

    return data


def parse_classification(raw: str) -> ClassificationResult:
    data = _extract_json(raw)
    try:
        return ClassificationResult.model_validate(data)
    except ValidationError as e:
        raise DocParserError(f"Classification did not match expected schema: {e}") from e


def parse_extraction(
    raw: str, category: DocumentCategory, document_type: DocumentType
) -> ExtractionResult:
    data = _extract_json(raw)
    model_cls = EXTRACTION_MODEL_MAP.get(document_type, DEFAULT_EXTRACTION_MODEL)
    try:
        fields = model_cls.model_validate(data)
    except ValidationError as e:
        raise DocParserError(
            f"Extraction did not match {model_cls.__name__} schema: {e}"
        ) from e
    return ExtractionResult(category=category, document_type=document_type, fields=fields)