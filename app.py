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

from auth import get_current_user
from authmodels import UserOut
from authrouter import router as auth_router
from docloader import DocLoaderError
from docmodels import PipelineResult
from docorchestrator import DocOrchestrator, DocOrchestratorError
from documentsdb import get_document_for_user, init_db as init_documents_db, list_documents_for_user, save_document
from documents_models import DocumentDetail, DocumentSummary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Insurance Document Router",
    description="Two-turn classify -> extract pipeline for insurance documents.",
    version="1.0.0",
)

# Auth endpoints: POST /register, POST /token, GET /users/me.
app.include_router(auth_router)

# Ensures the documents table exists (shares the same database/engine as
# auth_db.py's users table -- see documents_db.py). auth_router.py already
# triggers auth_db's init_db() at import time for the users table; this
# covers documents specifically and is safe to call regardless of import
# order, since CREATE TABLE IF NOT EXISTS is idempotent per table.
init_documents_db()

# Where uploaded files are staged before DocLoader reads them. Defaults to
# the system temp dir (unchanged local-dev behavior); in Docker, set
# UPLOAD_DIR (e.g. to /app/uploads, backed by a volume mount) so staged
# files land somewhere host-visible rather than the container's ephemeral
# temp space. Files are still deleted after each request either way (see
# the `finally: shutil.rmtree(...)` below) -- this only changes *where*
# they're staged during processing, not whether they persist afterward.
UPLOAD_DIR = os.environ.get("UPLOAD_DIR")
if UPLOAD_DIR:
    os.makedirs(UPLOAD_DIR, exist_ok=True)

# Single shared orchestrator instance. DocLoader/DocClient hold no
# per-request state, so this is safe to reuse across requests. Tests
# override get_orchestrator() via FastAPI's dependency_overrides rather
# than touching this instance directly.
_default_orchestrator = DocOrchestrator()


def get_orchestrator() -> DocOrchestrator:
    return _default_orchestrator


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
    # Write to a temp dir under the original filename (basename-only, to
    # avoid path traversal) so DocLoader sees the real suffix/name rather
    # than a random tempfile name.
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

        # Persist on success, keyed to whoever uploaded it. A storage
        # failure here logs a warning but doesn't fail the request --
        # the classification itself already succeeded, and losing that
        # result over a persistence-layer hiccup would be a worse
        # failure mode than just not having saved it this one time.
        try:
            save_document(current_user.id, result.model_dump())
        except Exception as e:
            logger.warning(
                "Failed to persist document result for user %s: %s", current_user.id, e
            )

        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents(current_user: UserOut = Depends(get_current_user)) -> list[dict]:
    return list_documents_for_user(current_user.id)


@app.get("/documents/{document_id}", response_model=DocumentDetail)
def get_document(document_id: int, current_user: UserOut = Depends(get_current_user)) -> dict:
    doc = get_document_for_user(current_user.id, document_id)
    if doc is None:
        # Same response whether the id doesn't exist at all or belongs
        # to someone else -- doesn't leak which case it is.
        raise HTTPException(status_code=404, detail="Document not found")
    return doc