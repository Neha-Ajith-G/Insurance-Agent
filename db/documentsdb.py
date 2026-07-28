"""
documents_db: SQLAlchemy-backed storage for processed documents.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from db.authdb import Base, SessionLocal, engine


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    category = Column(String, nullable=False)
    document_type = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    reasoning = Column(Text, nullable=True)
    extraction_fields_json = Column(Text, nullable=False)
    token_stats_json = Column(Text, nullable=False)
    total_input_tokens = Column(Integer, nullable=False)
    total_output_tokens = Column(Integer, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    total_inference_time = Column(Float, nullable=False)
    pipeline_elapsed = Column(Float, nullable=False)
    created_at = Column(String, nullable=False)

    owner = relationship("User")


class AgenticExecution(Base):
    __tablename__ = "agentic_executions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    category = Column(String, nullable=True)
    document_type = Column(String, nullable=True)
    state_json = Column(Text, nullable=False)
    total_tokens = Column(Integer, nullable=False)
    total_inference_time = Column(Float, nullable=False)
    total_elapsed = Column(Float, nullable=False)
    created_at = Column(String, nullable=False)

    owner = relationship("User")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def _agentic_to_dict(rec: AgenticExecution) -> dict:
    return {
        "id": rec.id,
        "user_id": rec.user_id,
        "filename": rec.filename,
        "category": rec.category,
        "document_type": rec.document_type,
        "state": json.loads(rec.state_json),
        "total_tokens": rec.total_tokens,
        "total_inference_time": rec.total_inference_time,
        "total_elapsed": rec.total_elapsed,
        "created_at": rec.created_at,
    }


def save_agentic_state(user_id: int, state_dict: dict) -> dict:
    db = SessionLocal()
    try:
        rec = AgenticExecution(
            user_id=user_id,
            filename=state_dict.get("filename", "unknown"),
            category=state_dict.get("category"),
            document_type=state_dict.get("document_type"),
            state_json=json.dumps(state_dict),
            total_tokens=state_dict.get("total_tokens", 0),
            total_inference_time=state_dict.get("total_inference_time", 0.0),
            total_elapsed=state_dict.get("total_elapsed", 0.0),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        db.add(rec)

        fixed = state_dict.get("fixed_schema") or {}
        open_sc = state_dict.get("open_schema") or {}
        open_kv = open_sc.get("key_value_pairs", {}) if isinstance(open_sc, dict) else {}
        combined_fields = {**open_kv, **fixed}

        verif = state_dict.get("verification") or {}
        v_score = verif.get("score", 1.0) if isinstance(verif, dict) else 1.0
        v_reasoning = verif.get("reasoning", "AgenticIE execution") if isinstance(verif, dict) else "AgenticIE execution"

        history = state_dict.get("history", [])
        token_stats = [
            {
                "turn": h.get("action", {}).get("tool", f"step_{i}"),
                "input_tokens": 0,
                "output_tokens": h.get("token_cost", 0),
                "total_tokens": h.get("token_cost", 0),
                "inference_time": h.get("inference_time", 0.0),
            }
            for i, h in enumerate(history, start=1)
        ]

        doc_rec = Document(
            user_id=user_id,
            filename=state_dict.get("filename", "unknown"),
            category=state_dict.get("category") or "OTHER",
            document_type=state_dict.get("document_type") or "OTHER",
            confidence=v_score,
            reasoning=f"AgenticIE: {v_reasoning}",
            extraction_fields_json=json.dumps(combined_fields),
            token_stats_json=json.dumps(token_stats),
            total_input_tokens=0,
            total_output_tokens=state_dict.get("total_tokens", 0),
            total_tokens=state_dict.get("total_tokens", 0),
            total_inference_time=state_dict.get("total_inference_time", 0.0),
            pipeline_elapsed=state_dict.get("total_elapsed", 0.0),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        db.add(doc_rec)

        db.commit()
        db.refresh(rec)
        return _agentic_to_dict(rec)
    finally:
        db.close()


def list_agentic_executions_for_user(user_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        records = (
            db.query(AgenticExecution)
            .filter(AgenticExecution.user_id == user_id)
            .order_by(AgenticExecution.id.desc())
            .all()
        )
        return [_agentic_to_dict(r) for r in records]
    finally:
        db.close()


def get_agentic_execution_for_user(user_id: int, execution_id: int) -> dict | None:
    db = SessionLocal()
    try:
        rec = (
            db.query(AgenticExecution)
            .filter(AgenticExecution.id == execution_id, AgenticExecution.user_id == user_id)
            .first()
        )
        return _agentic_to_dict(rec) if rec else None
    finally:
        db.close()


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
