"""
agentic_pipeline_logger: structured JSONL logging for the AgenticIE system.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

DEFAULT_AGENTIC_LOG_PATH = os.environ.get(
    "AGENTIC_PIPELINE_LOG_PATH", "agentic_pipeline_events.jsonl"
)


class AgenticPipelineEvent(BaseModel):
    timestamp: str
    event: str
    file_path: str | None = None
    filename: str | None = None
    step_number: int | None = None
    tool: str | None = None
    rationale: str | None = None
    result_summary: str | None = None
    verification_passed: bool | None = None
    verification_score: float | None = None
    verification_reasoning: str | None = None
    unsupported_claims: list[str] | None = None
    total_tokens: int | None = None
    total_elapsed: float | None = None
    token_cost: int | None = None
    inference_time: float | None = None
    error: str | None = None


class AgenticJsonlLogger:
    def __init__(self, path: str | Path = DEFAULT_AGENTIC_LOG_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, event: str, **fields) -> None:
        record = AgenticPipelineEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event=event,
            **fields,
        )
        line = record.model_dump_json(exclude_none=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
