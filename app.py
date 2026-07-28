"""
FastAPI app exposing the Smart Insurance Document Router pipeline.

POST /classify -- upload a document, get back classification + extraction
                  + token stats (including total token count).
GET  /health    -- liveness check, including whether the Ollama backend
                  is reachable.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from db.auth import get_current_user
from models.authmodels import UserOut
from authrouter import router as auth_router
from pipeline.docloader import DocLoaderError
from agents.agentic_ie import AgenticIEOrchestrator
from models.agentic_models import AgentState
from models.docmodels import PipelineResult
from pipeline.docorchestrator import DocOrchestrator, DocOrchestratorError
from db.documentsdb import (
    get_agentic_execution_for_user,
    get_document_for_user,
    init_db as init_documents_db,
    list_agentic_executions_for_user,
    list_documents_for_user,
    save_agentic_state,
    save_document,
)
from models.documents_models import DocumentDetail, DocumentSummary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Insurance Document Router",
    description="Two-turn classify -> extract pipeline and AgenticIE (Planner, Executor, Responder) for insurance documents.",
    version="1.1.0",
)

# Auth endpoints: POST /register, POST /token, GET /users/me.
app.include_router(auth_router)

init_documents_db()

UPLOAD_DIR = os.environ.get("UPLOAD_DIR")
if UPLOAD_DIR:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

_default_orchestrator = DocOrchestrator()
_agentic_orchestrator = AgenticIEOrchestrator()


def get_orchestrator() -> DocOrchestrator:
    return _default_orchestrator


def get_agentic_orchestrator() -> AgenticIEOrchestrator:
    return _agentic_orchestrator


@app.get("/health")
def health(orchestrator: DocOrchestrator = Depends(get_orchestrator)) -> dict:
    health_check = getattr(orchestrator.client, "health", None)
    ollama_ok = health_check() if callable(health_check) else True
    return {"status": "ok" if ollama_ok else "degraded", "ollama_reachable": ollama_ok}


@app.post("/classify", response_model=PipelineResult)
async def classify(
    file: UploadFile = File(...),
    orchestrator: DocOrchestrator = Depends(get_orchestrator),
    current_user: UserOut = Depends(get_current_user),
) -> PipelineResult:
    safe_name = os.path.basename(file.filename or "upload")
    tmp_dir = tempfile.mkdtemp(prefix="doc_router_", dir=UPLOAD_DIR)
    tmp_path = os.path.join(tmp_dir, safe_name)

    try:
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        try:
            result = orchestrator.run(tmp_path)
        except DocLoaderError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except DocOrchestratorError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e

        try:
            save_document(current_user.id, result.model_dump())
        except Exception as e:
            logger.warning(
                "Failed to persist document result for user %s: %s", current_user.id, e
            )

        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/agentic/process", response_model=AgentState)
async def agentic_process(
    file: UploadFile = File(...),
    user_query: str | None = None,
    use_deterministic_baseline: bool = True,
    agentic_orch: AgenticIEOrchestrator = Depends(get_agentic_orchestrator),
    deterministic_orch: DocOrchestrator = Depends(get_orchestrator),
    current_user: UserOut = Depends(get_current_user),
) -> AgentState:
    safe_name = os.path.basename(file.filename or "upload")
    tmp_dir = tempfile.mkdtemp(prefix="agentic_router_", dir=UPLOAD_DIR)
    tmp_path = os.path.join(tmp_dir, safe_name)

    try:
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        deterministic_baseline = None
        if use_deterministic_baseline:
            try:
                det_result = deterministic_orch.run(tmp_path)
                deterministic_baseline = det_result.model_dump()
            except Exception as e:
                logger.info("Deterministic baseline skipped or failed: %s", e)

        try:
            state = agentic_orch.run(
                path=tmp_path,
                user_query=user_query,
                deterministic_baseline=deterministic_baseline,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AgenticIE execution failed: {e}") from e

        try:
            save_agentic_state(current_user.id, state.model_dump())
        except Exception as e:
            logger.warning("Failed to persist AgenticState for user %s: %s", current_user.id, e)

        return state
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/agentic/executions")
def list_agentic_executions(current_user: UserOut = Depends(get_current_user)) -> list[dict]:
    return list_agentic_executions_for_user(current_user.id)


@app.get("/agentic/executions/{execution_id}")
def get_agentic_execution(
    execution_id: int, current_user: UserOut = Depends(get_current_user)
) -> dict:
    rec = get_agentic_execution_for_user(current_user.id, execution_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="Agentic execution record not found")
    return rec


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents(current_user: UserOut = Depends(get_current_user)) -> list[dict]:
    return list_documents_for_user(current_user.id)


@app.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: int, current_user: UserOut = Depends(get_current_user)) -> dict:
    doc = get_document_for_user(current_user.id, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


class ChatRequest(BaseModel):
    filename: str
    query: str
    chat_history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[dict]
    tokens_used: int
    inference_time: float


@app.post("/agentic/chat", response_model=ChatResponse)
def agentic_chat(
    req: ChatRequest,
    agentic_orch: AgenticIEOrchestrator = Depends(get_agentic_orchestrator),
    current_user: UserOut = Depends(get_current_user),
) -> ChatResponse:
    """RAG Document Chatbot endpoint querying local ChromaDB vector store and Ollama model."""
    results = agentic_orch.vector_store.search(
        query=req.query, top_k=3, where_filter={"filename": req.filename} if req.filename else None
    )
    context_text = "\n\n".join([r["document"] for r in results]) if results else "No specific vector context found."

    history_str = ""
    for msg in req.chat_history[-6:]:
        role = msg.get("role", "user").capitalize()
        history_str += f"{role}: {msg.get('content', '')}\n"

    prompt = (
        f"You are an insurance document assistant chatbot. Answer the user's question accurately based strictly on the document text context.\n\n"
        f"Document Filename: {req.filename}\n\n"
        f"Retrieved Vector Context:\n{context_text}\n\n"
        f"Recent Conversation History:\n{history_str}\n"
        f"User Question: {req.query}\n"
        f"Grounded Answer:"
    )
    resp = agentic_orch.client.generate(prompt=prompt)
    return ChatResponse(
        answer=resp.text.strip(),
        retrieved_chunks=results,
        tokens_used=resp.total_tokens,
        inference_time=resp.inference_time,
    )
