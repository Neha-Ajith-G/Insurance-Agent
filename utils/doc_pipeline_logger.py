"""
JsonlPipelineLogger: structured logging of pipeline state transitions.
"""
from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

DEFAULT_LOG_PATH = os.environ.get("PIPELINE_LOG_PATH", "pipeline_events.jsonl")


class PipelineEvent(BaseModel):
    timestamp: str
    event: str
    file_path: str | None = None
    filename: str | None = None
    stage: str | None = None
    attempt: int | None = None
    category: str | None = None
    document_type: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    inference_time: float | None = None
    elapsed: float | None = None
    error: str | None = None


class JsonlPipelineLogger:
    def __init__(self, path: str | Path = DEFAULT_LOG_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()

    def log(self, event: str, **fields) -> None:
        record = PipelineEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event=event,
            **fields,
        )
        line = record.model_dump_json(exclude_none=True)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
