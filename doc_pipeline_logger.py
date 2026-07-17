"""
JsonlPipelineLogger: structured, machine-readable logging of pipeline
state transitions (one JSON object per line).

This is separate from the human-readable `logging.getLogger` calls
already in doc_orchestrator.py -- those are for a developer tailing
logs during a demo; this is for offline analysis: token usage over
time, retry rates per document type, error frequency, etc. Reading the
file back is just `[json.loads(line) for line in open(path)]`, or load
straight into pandas with `pd.read_json(path, lines=True)`.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel

DEFAULT_LOG_PATH = "pipeline_events.jsonl"


class PipelineEvent(BaseModel):
    """
    One row in the JSONL log. Every event type uses this same shape;
    fields that don't apply to a given event are simply omitted from
    the serialized line (via exclude_none) rather than forcing every
    event to carry every field.
    """

    timestamp: str
    event: str
    file_path: str | None = None
    filename: str | None = None
    stage: str | None = None  # "classification" | "extraction"
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
    """
    Appends one JSON object per line to `path`. A lock guards writes
    since a single DocOrchestrator instance is shared across concurrent
    FastAPI requests -- without it, interleaved writes from two
    in-flight requests could corrupt a line.
    """

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
