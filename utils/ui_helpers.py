"""
ui_helpers: pure functions for UI display formatting.
"""
from __future__ import annotations


def normalize_document_detail(doc: dict) -> dict:
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


def confidence_label(conf: float | str | None) -> str | None:
    if conf is None:
        return None
    try:
        val = float(conf)
    except (ValueError, TypeError):
        return None

    if val >= 0.8:
        return "High"
    if val >= 0.5:
        return "Medium"
    return "Low"



def format_extraction_fields(fields: dict) -> list[tuple[str, str]]:
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
                display = f"{len(value)} item(s) -- see raw JSON below"
        elif isinstance(value, float):
            display = f"{value:,.2f}"
        else:
            display = str(value)
        rows.append((label, display))
    return rows


def format_token_summary(result: dict) -> dict:
    return {
        "total_input_tokens": result.get("total_input_tokens", 0),
        "total_output_tokens": result.get("total_output_tokens", 0),
        "total_tokens": result.get("total_tokens", 0),
        "total_inference_time": result.get("total_inference_time", 0.0),
        "pipeline_elapsed": result.get("pipeline_elapsed", 0.0),
    }


def format_turn_breakdown(token_stats: list[dict]) -> list[dict]:
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
