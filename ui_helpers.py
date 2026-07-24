"""
ui_helpers: pure functions that turn a /classify API response into
display-ready structures. Deliberately separate from streamlit_app.py's
st.* calls so this logic can be unit tested directly -- Streamlit apps
can't easily be exercised outside their own runtime, but this can.
"""
from __future__ import annotations


def normalize_document_detail(doc: dict) -> dict:
    """
    Reshapes a GET /documents/{id} response (flat: category,
    document_type, extraction_fields, ...) into the same shape as a
    POST /classify PipelineResult response (nested: classification.*,
    extraction.fields), so the UI's single results-rendering function
    can display either a fresh classification or a previously stored
    one without duplicating the rendering code.
    """
    return {
        "filename": doc["filename"],
        "classification": {
            "category": doc["category"],
            "document_type": doc["document_type"],
            "confidence": doc.get("confidence"),
            "reasoning": doc.get("reasoning"),
        },
        "extraction": {"fields": doc["extraction_fields"]},
        "token_stats": doc["token_stats"],
        "total_input_tokens": doc["total_input_tokens"],
        "total_output_tokens": doc["total_output_tokens"],
        "total_tokens": doc["total_tokens"],
        "total_inference_time": doc["total_inference_time"],
        "pipeline_elapsed": doc["pipeline_elapsed"],
    }


def format_extraction_fields(fields: dict) -> list[tuple[str, str]]:
    """
    Turns the extraction.fields dict into a list of (label, display_value)
    pairs for a clean table: converts field_name -> "Field Name", renders
    None as "-- not found --" rather than a bare "None", and formats list
    values (e.g. key_points, line_items) as a joined, readable string
    instead of a raw Python repr.
    """
    rows = []
    for key, value in fields.items():
        label = key.replace("_", " ").title()
        if value is None:
            display = "-- not found --"
        elif isinstance(value, list):
            if not value:
                display = "-- none --"
            elif all(isinstance(v, (str, int, float)) for v in value):
                display = ", ".join(str(v) for v in value)
            else:
                # list of dicts (e.g. line_items) -- summarize count, full
                # detail can still be inspected via the raw JSON expander
                display = f"{len(value)} item(s) -- see raw JSON below"
        elif isinstance(value, float):
            display = f"{value:,.2f}"
        else:
            display = str(value)
        rows.append((label, display))
    return rows


def format_token_summary(result: dict) -> dict:
    """
    Pulls the top-level token/timing totals out of a PipelineResult JSON
    body into a flat dict for st.metric() display. Falls back to 0 for
    any field that's unexpectedly missing rather than raising, since this
    feeds direct UI display -- a missing field should degrade gracefully
    to "0", not crash the results page.
    """
    return {
        "total_input_tokens": result.get("total_input_tokens", 0),
        "total_output_tokens": result.get("total_output_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "total_inference_time": result.get("total_inference_time", 0.0),
        "pipeline_elapsed": result.get("pipeline_elapsed", 0.0),
    }


def format_turn_breakdown(token_stats: list[dict]) -> list[dict]:
    """
    Per-turn token/timing rows for a detail table -- one row per LLM call
    (classify, and one per extraction chunk if the document was chunked).
    """
    rows = []
    for i, stat in enumerate(token_stats, start=1):
        rows.append({
            "Turn": f"{i}. {stat.get('turn', 'unknown').title()}",
            "Input Tokens": stat.get("input_tokens", 0),
            "Output Tokens": stat.get("output_tokens", 0),
            "Total Tokens": stat.get("total_tokens", 0),
            "Time (s)": round(stat.get("inference_time", 0.0), 2),
        })
    return rows


def confidence_label(confidence: float | None) -> str:
    """Maps a raw 0-1 confidence score to a short human label, used
    alongside (not instead of) the raw number -- so a reader gets both
    the exact value and an at-a-glance sense of what it means."""
    if confidence is None:
        return "unknown"
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"