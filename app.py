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

from docloader import DocLoaderError
from docmodels import PipelineResult
from docorchestrator import DocOrchestrator, DocOrchestratorError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
logging.getLogger("docorchestrator").addHandler(handler)
logging.getLogger("docorchestrator").setLevel(logging.DEBUG)

app = FastAPI(
    title="Smart Insurance Document Router",
    description="Two-turn classify -> extract pipeline for insurance documents.",
    version="1.0.0",
)

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
) -> PipelineResult:
    # Write to a temp dir under the original filename (basename-only, to
    # avoid path traversal) so DocLoader sees the real suffix/name rather
    # than a random tempfile name.
    safe_name = os.path.basename(file.filename or "upload")
    tmp_dir = tempfile.mkdtemp(prefix="doc_router_")
    tmp_path = os.path.join(tmp_dir, safe_name)

    try:
        with open(tmp_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        try:
            return orchestrator.run(tmp_path)
        except DocLoaderError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except DocOrchestratorError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)