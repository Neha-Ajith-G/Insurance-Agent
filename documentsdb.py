"""
documents_db: SQLAlchemy-backed storage for processed documents,
keyed by the user who uploaded them.

Shares the same engine/Base as auth_db.py (one database file), so
user_id genuinely references the real users table via a foreign key,
rather than trusting an unchecked integer. Same "each db module owns
its own table(s)" pattern as auth_db.py -- this one owns documents,
nothing else.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from authdb import Base, SessionLocal, engine


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    category = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    # Extraction fields and per-turn token stats vary in shape depending
    # on document type / chunk count -- stored as JSON text rather than
    # a fixed set of columns, since a rigid schema here would fight the
    # same "5 detailed models + 1 generic fallback" flexibility already
    # built into docmodels.py.
    extraction_fields_json = Column(Text, nullable=False)
    token_stats_json = Column(Text, nullable=False)
    total_input_tokens = Column(Integer, nullable=False)
    total_output_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    total_inference_time = Column(Float, nullable=False)
    pipeline_elapsed = Column(Float, nullable=False)
    created_at = Column(String, nullable=False)

    owner = relationship("User")


def init_db() -> None:
    """Creates the documents table (and anything else registered on the
    shared Base -- e.g. users, from auth_db.py) if it doesn't exist yet.
    Safe to call repeatedly, and safe to call alongside auth_db.init_db()
    -- CREATE TABLE IF NOT EXISTS is idempotent per table."""
    Base.metadata.create_all(bind=engine)


def _to_dict(doc: Document) -> dict:
    return {
        "id": doc.id,
        "user_id": doc.user_id,
        "filename": doc.filename,
        "category": doc.category,
        "document_type": doc.document_type,
        "confidence": doc.confidence,
        "reasoning": doc.reasoning,
        "extraction_fields": json.loads(doc.extraction_fields_json),
        "token_stats": json.loads(doc.token_stats_json),
        "total_input_tokens": doc.total_input_tokens,
        "total_output_tokens": doc.total_output_tokens,
        "total_tokens": doc.total_tokens,
        "total_inference_time": doc.total_inference_time,
        "pipeline_elapsed": doc.pipeline_elapsed,
        "created_at": doc.created_at,
    }


def save_document(user_id: int, pipeline_result: dict) -> dict:
    """Persists a PipelineResult (as a plain dict, e.g. via
    PipelineResult.model_dump()) for the given user."""
    db = SessionLocal()
    try:
        classification = pipeline_result["classification"]
        extraction = pipeline_result["extraction"]
        doc = Document(
            user_id=user_id,
            filename=pipeline_result["filename"],
            category=classification["category"],
            document_type=classification["document_type"],
            confidence=classification.get("confidence"),
            reasoning=classification.get("reasoning"),
            extraction_fields_json=json.dumps(extraction["fields"]),
            token_stats_json=json.dumps(pipeline_result["token_stats"]),
            total_input_tokens=pipeline_result["total_input_tokens"],
            total_output_tokens=pipeline_result["total_output_tokens"],
            total_tokens=pipeline_result["total_tokens"],
            total_inference_time=pipeline_result["total_inference_time"],
            pipeline_elapsed=pipeline_result["pipeline_elapsed"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return _to_dict(doc)
    finally:
        db.close()


def list_documents_for_user(user_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        docs = (
            db.query(Document)
            .filter(Document.user_id == user_id)
            .order_by(Document.id.desc())
            .all()
        )
        return [_to_dict(d) for d in docs]
    finally:
        db.close()


def get_document_for_user(user_id: int, document_id: int) -> dict | None:
    """Returns None if the document doesn't exist OR belongs to a
    different user -- deliberately indistinguishable (via the caller's
    404), so a request can't be used to tell 'wrong id' apart from
    'someone else's document' -- an ownership check that doesn't leak
    existence."""
    db = SessionLocal()
    try:
        doc = (
            db.query(Document)
            .filter(Document.id == document_id, Document.user_id == user_id)
            .first()
        )
        return _to_dict(doc) if doc else None
    finally:
        db.close()