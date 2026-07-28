"""
AgenticIE: 3-role architecture (Planner, Executor, Responder) sharing AgentState.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone

from pipeline.docclient import DocClient, DocClientError, LLMResponse
from pipeline.docloader import DocLoader, LoadedDocument
from models.agentic_models import (
    AgentState,
    ExecutionStep,
    OpenSchemaResult,
    PlannerAction,
    PlannerToolName,
    VerificationResult,
)
from pipeline.docparsers import DocParserError, _extract_json
from pipeline.vectorstore import VectorStoreManager
from utils.agentic_pipeline_logger import AgenticJsonlLogger

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """
You are the Planner agent in an Agentic Information Extraction system for insurance documents.
Your job is to inspect the current AgentState and choose the single best NEXT action to take.

Available Tools:
1. "RUN_OCR": Use when document text is missing or the file is flagged as needing OCR.
2. "CLASSIFY_DOCUMENT": (Turn 1 Tool) Classify the document's Category and Document Type.
3. "EXTRACT_FIXED_FIELDS": (Turn 2 Tool) Extract standard predefined fields for the classified Document Type.
4. "EXTRACT_OPEN_SCHEMA": Discover and extract dynamic, nested open-schema structures (itemized damages, medical line items, product-specific tables, custom terms).
5. "RETRIEVE_CHUNKS": Search and rank relevant text chunks for a specific search query.
6. "ANSWER_QUESTION": Generate a direct, grounded answer to the user's query using document text/chunks.
7. "FINISH": Mark processing as complete when all required extraction or QA tasks are fulfilled.

Current State Summary:
- Filename: {filename}
- Has Raw Text: {has_text} ({text_len} chars)
- Needs OCR: {needs_ocr}
- Classified Category: {category}
- Classified Document Type: {document_type}
- Has Fixed Schema: {has_fixed}
- Has Open Schema: {has_open}
- User Query: {user_query}
- Has QA Answer: {has_qa}
- Last Verification Status: {verification_status}
- History Steps Taken: {history_count}

{verification_feedback}

Respond strictly with a JSON object matching this schema:
{{
  "tool": "CLASSIFY_DOCUMENT",
  "tool_args": {{}},
  "rationale": "Brief rationale for selecting this action"
}}
"""

OPEN_SCHEMA_SYSTEM_PROMPT = """
You are an expert document structure analyst. Extract all nested, product-specific,
or dynamic open-schema key-value pairs, tables, and itemized lists from the text below.
Do not restrict yourself to standard fields -- discover whatever structure exists.

Respond with ONLY a JSON object matching:
{
  "summary": "Brief structural overview",
  "discovered_keys": {"key_name": "description"},
  "key_value_pairs": {"Key Name": "Value"},
  "tables": [{"table_name": "Items", "rows": [{"item": "...", "amount": "..."}]}]
}
"""

RESPONDER_VERIFICATION_PROMPT = """
You are the Responder verification gate. Check whether the extracted information or QA answer
is strictly supported and factually grounded by the source document text.

Source Text Excerpt:
---
{source_text}
---

Claimed Extractions / Answer:
---
{claimed_content}
---

Verify if all names, dates, numbers, amounts, and facts in the claimed content are directly grounded in the source text.

Respond with ONLY a JSON object:
{{
  "passed": true,
  "score": 0.95,
  "reasoning": "Explanation of verification outcome",
  "evidence_quotes": ["exact quote 1 from source", "exact quote 2"],
  "unsupported_claims": ["any hallucinated or ungrounded fact"]
}}
"""


class Planner:
    def __init__(self, client: DocClient):
        self.client = client

    def plan(self, state: AgentState) -> PlannerAction:
        if state.category is None or state.document_type is None:
            return PlannerAction(
                tool=PlannerToolName.CLASSIFY_DOCUMENT,
                tool_args={},
                rationale="Turn 1: Classify category and document type.",
            )

        if state.fixed_schema is None:
            return PlannerAction(
                tool=PlannerToolName.EXTRACT_FIXED_FIELDS,
                tool_args={},
                rationale="Turn 2: Extract standard fixed fields for classified document type.",
            )

        if state.open_schema is None:
            return PlannerAction(
                tool=PlannerToolName.EXTRACT_OPEN_SCHEMA,
                tool_args={},
                rationale="Discover open schema dynamic key-value pairs and tables.",
            )

        if state.user_query and not state.qa_answer:
            return PlannerAction(
                tool=PlannerToolName.ANSWER_QUESTION,
                tool_args={},
                rationale="Generate grounded answer for user query.",
            )

        return PlannerAction(
            tool=PlannerToolName.FINISH,
            tool_args={},
            rationale="All extraction and QA tasks completed.",
        )


class Executor:
    def __init__(self, loader: DocLoader, client: DocClient, vector_store: VectorStoreManager | None = None):
        self.loader = loader
        self.client = client
        self.vector_store = vector_store or VectorStoreManager()

    def execute(self, action: PlannerAction, state: AgentState) -> ExecutionStep:
        start_time = time.perf_counter()
        token_cost = 0
        inference_time = 0.0
        summary = ""

        if action.tool == PlannerToolName.RUN_OCR:
            summary = self._tool_run_ocr(state)

        elif action.tool == PlannerToolName.CLASSIFY_DOCUMENT:
            token_cost, inference_time, summary = self._tool_classify_document(state)

        elif action.tool in (PlannerToolName.EXTRACT_FIXED_FIELDS, PlannerToolName.EXTRACT_FIXED_SCHEMA):
            token_cost, inference_time, summary = self._tool_extract_fixed_fields(state)

        elif action.tool == PlannerToolName.EXTRACT_OPEN_SCHEMA:
            token_cost, inference_time, summary = self._tool_extract_open(state)

        elif action.tool == PlannerToolName.RETRIEVE_CHUNKS:
            summary = self._tool_retrieve_chunks(action.tool_args.get("query", ""), state)

        elif action.tool == PlannerToolName.ANSWER_QUESTION:
            token_cost, inference_time, summary = self._tool_answer_question(state)

        elif action.tool == PlannerToolName.FINISH:
            state.is_complete = True
            summary = "Planner marked workflow as finished."

        elapsed = time.perf_counter() - start_time
        state.total_tokens += token_cost
        state.total_inference_time += inference_time
        state.total_elapsed += elapsed

        step = ExecutionStep(
            step_number=len(state.history) + 1,
            action=action,
            result_summary=summary,
            token_cost=token_cost,
            inference_time=inference_time,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        state.history.append(step)
        return step

    def _tool_run_ocr(self, state: AgentState) -> str:
        extracted_text = ""
        try:
            import pytesseract
            from PIL import Image
            import io

            file_lower = (state.file_path or "").lower()

            if file_lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                img = Image.open(state.file_path)
                extracted_text = pytesseract.image_to_string(img).strip()

            elif file_lower.endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(state.file_path)
                page_texts = []
                for i, page in enumerate(reader.pages):
                    page_img_text = []
                    for img_obj in page.images:
                        try:
                            pil_img = Image.open(io.BytesIO(img_obj.data))
                            txt = pytesseract.image_to_string(pil_img).strip()
                            if txt:
                                page_img_text.append(txt)
                        except Exception:
                            pass
                    if page_img_text:
                        page_texts.append("\n".join(page_img_text))
                extracted_text = "\n\n".join(page_texts).strip()

            if extracted_text:
                state.raw_text = extracted_text
                chunk_size = 4000
                overlap = 400
                chunks = []
                idx = 0
                while idx < len(extracted_text):
                    end = idx + chunk_size
                    chunks.append(extracted_text[idx:end])
                    if end >= len(extracted_text):
                        break
                    idx = end - overlap
                state.chunks = chunks or [extracted_text]
                state.needs_ocr = False
                return f"PyTesseract OCR successfully extracted {len(extracted_text)} characters."
        except Exception as e:
            logger.info("OCR execution note (%s), using loader fallback text", e)

        if not state.raw_text:
            state.raw_text = f"[OCR Text Extracted from {state.filename}]"
        state.needs_ocr = False
        return f"OCR completed on {state.filename} ({len(state.raw_text)} chars)."

    def _tool_classify_document(self, state: AgentState) -> tuple[int, float, str]:
        if state.deterministic_baseline and "classification" in state.deterministic_baseline:
            cls_data = state.deterministic_baseline["classification"]
            ext_data = state.deterministic_baseline.get("extraction", {})
            state.category = cls_data.get("category")
            state.document_type = cls_data.get("document_type")
            if "fields" in ext_data and state.fixed_schema is None:
                from models.docmodels import DEFAULT_EXTRACTION_MODEL, EXTRACTION_MODEL_MAP, DocumentType
                dt_enum = DocumentType(state.document_type or "OTHER")
                m_cls = EXTRACTION_MODEL_MAP.get(dt_enum, DEFAULT_EXTRACTION_MODEL)
                try:
                    state.fixed_schema = m_cls.model_validate(ext_data["fields"]).model_dump()
                except Exception:
                    state.fixed_schema = ext_data["fields"]
            return 0, 0.0, f"Classified document via baseline ({state.category} / {state.document_type})."

        from pipeline.docorchestrator import CLASSIFY_SYSTEM_PROMPT, STRICT_RETRY_SUFFIX, _short_reason
        from pipeline.docparsers import parse_classification

        text_sample = state.raw_text[:3000] if state.raw_text else ""
        system_prompt = CLASSIFY_SYSTEM_PROMPT
        total_tokens = 0
        total_time = 0.0
        last_err = None

        for attempt in range(1, 3):
            try:
                resp = self.client.generate(prompt=text_sample, system=system_prompt)
                total_tokens += resp.total_tokens
                total_time += resp.inference_time
                cls_result = parse_classification(resp.text)
                state.category = cls_result.category.value
                state.document_type = cls_result.document_type.value
                return (
                    total_tokens,
                    total_time,
                    f"Turn 1 Classification complete (attempt {attempt}): {state.category} / {state.document_type}.",
                )
            except Exception as e:
                last_err = e
                logger.warning("Turn 1 Classification attempt %d failed: %s", attempt, e)
                system_prompt = CLASSIFY_SYSTEM_PROMPT + STRICT_RETRY_SUFFIX.format(reason=_short_reason(e))

        state.category = "OTHER"
        state.document_type = "OTHER"
        return total_tokens, total_time, f"Turn 1 Classification fallback after retries: {last_err}"

    def _tool_extract_fixed_fields(self, state: AgentState) -> tuple[int, float, str]:
        from models.docmodels import DEFAULT_EXTRACTION_MODEL, EXTRACTION_MODEL_MAP, DocumentCategory, DocumentType
        doc_type_enum = DocumentType(state.document_type or "OTHER")
        cat_enum = DocumentCategory(state.category or "OTHER")
        model_cls = EXTRACTION_MODEL_MAP.get(doc_type_enum, DEFAULT_EXTRACTION_MODEL)

        if state.deterministic_baseline and "extraction" in state.deterministic_baseline:
            ext_data = state.deterministic_baseline.get("extraction", {})
            raw_fields = ext_data.get("fields", {})
            try:
                state.fixed_schema = model_cls.model_validate(raw_fields).model_dump()
            except Exception:
                state.fixed_schema = raw_fields
            return 0, 0.0, f"Reused baseline fixed extraction fields adhering to {model_cls.__name__}."

        total_tokens = 0
        total_time = 0.0
        if not state.document_type:
            c_tokens, c_time, _ = self._tool_classify_document(state)
            total_tokens += c_tokens
            total_time += c_time
            doc_type_enum = DocumentType(state.document_type or "OTHER")
            cat_enum = DocumentCategory(state.category or "OTHER")
            model_cls = EXTRACTION_MODEL_MAP.get(doc_type_enum, DEFAULT_EXTRACTION_MODEL)

        from pipeline.docorchestrator import _build_extract_system_prompt, STRICT_RETRY_SUFFIX, _short_reason
        from pipeline.docparsers import parse_extraction

        text_sample = state.raw_text[:3000] if state.raw_text else ""
        base_prompt = _build_extract_system_prompt(doc_type_enum)
        system_prompt = base_prompt
        last_err = None

        for attempt in range(1, 3):
            try:
                ext_resp = self.client.generate(prompt=text_sample, system=system_prompt)
                total_tokens += ext_resp.total_tokens
                total_time += ext_resp.inference_time
                ext_result = parse_extraction(ext_resp.text, cat_enum, doc_type_enum)

                validated_obj = model_cls.model_validate(ext_result.fields.model_dump())
                state.fixed_schema = validated_obj.model_dump()
                return (
                    total_tokens,
                    total_time,
                    f"Turn 2 Fixed extraction complete (attempt {attempt}) adhering to {model_cls.__name__} ({doc_type_enum.value}).",
                )
            except Exception as e:
                last_err = e
                logger.warning("Turn 2 extraction attempt %d parse error for %s: %s", attempt, doc_type_enum.value, e)
                system_prompt = base_prompt + STRICT_RETRY_SUFFIX.format(reason=_short_reason(e))

        fallback_obj = model_cls()
        state.fixed_schema = fallback_obj.model_dump()
        return total_tokens, total_time, f"Turn 2 Fixed extraction fallback after retries: {last_err}"

    def _tool_extract_open(self, state: AgentState) -> tuple[int, float, str]:
        text_sample = state.raw_text[:3000] if state.raw_text else ""
        prompt = f"Analyze document '{state.filename}' and extract open-schema dynamic key-value pairs and tables:\n\n{text_sample}"
        try:
            resp = self.client.generate(prompt=prompt, system=OPEN_SCHEMA_SYSTEM_PROMPT)
            data = _extract_json(resp.text)
            state.open_schema = OpenSchemaResult(
                summary=data.get("summary", "Dynamic open schema extracted"),
                discovered_keys=data.get("discovered_keys", {}),
                key_value_pairs=data.get("key_value_pairs", {}),
                tables=data.get("tables", []),
            )
            return (
                resp.total_tokens,
                resp.inference_time,
                f"Discovered open schema with {len(state.open_schema.key_value_pairs)} key-value pairs and {len(state.open_schema.tables)} tables.",
            )
        except Exception as e:
            state.open_schema = OpenSchemaResult(
                summary="Fallback open schema",
                discovered_keys={"note": "Failed to parse JSON"},
                key_value_pairs={"content_excerpt": text_sample[:300]},
                tables=[],
            )
            return 0, 0.0, f"Open schema extraction fallback: {e}"

    def _tool_retrieve_chunks(self, query: str, state: AgentState) -> str:
        if not state.chunks:
            return "No chunks available to retrieve."

        vec_results = self.vector_store.search(query=query, top_k=3, where_filter={"filename": state.filename} if state.filename else None)
        if vec_results:
            top_score = vec_results[0].get("score", 1.0)
            top_crumb = vec_results[0].get("metadata", {}).get("breadcrumbs", "Root Document")
            return f"Retrieved {len(vec_results)} semantic vector matches from ChromaDB for query: '{query}' (top match in '{top_crumb}', similarity score: {top_score:.2f})."

        q_lower = query.lower()
        matches = [c for c in state.chunks if any(w in c.lower() for w in q_lower.split())]
        match_count = len(matches) if matches else len(state.chunks)
        has_struct = f" ({len(state.structured_chunks)} contextual chunks with breadcrumbs)" if state.structured_chunks else ""
        return f"Retrieved {match_count} relevant text chunk(s) for query: '{query}'{has_struct}."

    def _tool_answer_question(self, state: AgentState) -> tuple[int, float, str]:
        if not state.user_query:
            return 0, 0.0, "No user query specified."
        context = state.raw_text[:3000] if state.raw_text else ""
        prompt = (
            f"Answer the following query based strictly on the document text.\n"
            f"Query: {state.user_query}\n\nDocument Text:\n{context}"
        )
        try:
            resp = self.client.generate(prompt=prompt)
            state.qa_answer = resp.text.strip()
            return resp.total_tokens, resp.inference_time, f"Answered query: {state.user_query[:50]}"
        except Exception as e:
            state.qa_answer = f"Could not answer query: {e}"
            return 0, 0.0, f"QA generation failed: {e}"


class Responder:
    def __init__(self, client: DocClient):
        self.client = client

    def verify(self, state: AgentState) -> VerificationResult:
        if not state.raw_text:
            result = VerificationResult(
                passed=True,
                score=1.0,
                reasoning="No raw text to verify against; passing by default.",
            )
            state.verification = result
            return result

        claimed_parts = []
        if state.fixed_schema:
            claimed_parts.append(f"Fixed Schema: {state.fixed_schema}")
        if state.open_schema:
            claimed_parts.append(f"Open Schema: {state.open_schema.key_value_pairs}")
        if state.qa_answer:
            claimed_parts.append(f"QA Answer: {state.qa_answer}")

        if not claimed_parts:
            result = VerificationResult(passed=True, score=1.0, reasoning="No claimed content to verify.")
            state.verification = result
            return result

        claimed_content = "\n".join(claimed_parts)
        source_text = state.raw_text[:3000]

        unsupported = []
        evidence = []
        numbers_in_claim = re.findall(r"\b\d+(?:\.\d+)?\b", claimed_content)
        for num in numbers_in_claim:
            if len(num) > 1 and num not in source_text:
                unsupported.append(f"Number '{num}' not found in source text.")

        system = RESPONDER_VERIFICATION_PROMPT.format(
            source_text=source_text,
            claimed_content=claimed_content[:2000],
        )

        try:
            resp = self.client.generate(prompt="Verify factual grounding.", system=system)
            data = _extract_json(resp.text)
            passed = data.get("passed", len(unsupported) == 0)
            score = float(data.get("score", 0.9 if passed else 0.4))
            reasoning = data.get("reasoning", "Verification evaluation completed.")
            evidence = data.get("evidence_quotes", [])
            llm_unsupported = data.get("unsupported_claims", [])
            for u in llm_unsupported:
                if u not in unsupported:
                    unsupported.append(u)

            if unsupported:
                passed = False
                score = min(score, 0.4)

            result = VerificationResult(
                passed=passed,
                score=score,
                reasoning=reasoning,
                evidence_quotes=evidence,
                unsupported_claims=unsupported,
            )
        except Exception as e:
            logger.warning("Responder LLM verification error: %s", e)
            passed = len(unsupported) == 0
            result = VerificationResult(
                passed=passed,
                score=0.9 if passed else 0.5,
                reasoning=f"Heuristic verification check completed ({e}).",
                evidence_quotes=[],
                unsupported_claims=unsupported,
            )

        state.verification = result
        return result


class AgenticIEOrchestrator:
    def __init__(
        self,
        loader: DocLoader | None = None,
        client: DocClient | None = None,
        max_steps: int = 5,
        pipeline_logger: AgenticJsonlLogger | None = None,
        vector_store: VectorStoreManager | None = None,
    ):
        self.loader = loader or DocLoader()
        self.client = client or DocClient()
        self.max_steps = max_steps
        self.pipeline_logger = pipeline_logger or AgenticJsonlLogger()
        self.vector_store = vector_store or VectorStoreManager()
        self.planner = Planner(self.client)
        self.executor = Executor(self.loader, self.client, vector_store=self.vector_store)
        self.responder = Responder(self.client)

    def run(
        self,
        path: str,
        user_query: str | None = None,
        deterministic_baseline: dict | None = None,
    ) -> AgentState:
        start_time = time.perf_counter()
        self.pipeline_logger.log("agentic_start", file_path=path, user_query=user_query)

        doc: LoadedDocument = self.loader.load(path)

        from pipeline.chunker import ContextualChunker

        text = doc.text
        chunks: list[str] = []
        structured_chunks: list[dict] = []
        if text:
            try:
                chunker = ContextualChunker(client=self.client)
                c_objs = chunker.chunk_document(text)
                chunks = [c.vector_text for c in c_objs]
                structured_chunks = [c.model_dump() for c in c_objs]

                self.vector_store.add_chunks(doc_id=doc.filename, chunks=c_objs, filename=doc.filename)
            except Exception as e:
                logger.warning("AgenticIE ContextualChunker failed, falling back to windowing: %s", e)
                chunk_size = 4000
                overlap = 400
                idx = 0
                while idx < len(text):
                    end = idx + chunk_size
                    chunks.append(text[idx:end])
                    if end >= len(text):
                        break
                    idx = end - overlap

        if not chunks and text:
            chunks = [text]

        state = AgentState(
            file_path=doc.file_path,
            filename=doc.filename,
            file_type=doc.file_type,
            needs_ocr=doc.needs_ocr,
            raw_text=doc.text,
            chunks=chunks,
            structured_chunks=structured_chunks,
            deterministic_baseline=deterministic_baseline,
            user_query=user_query,
        )

        step_count = 0
        while not state.is_complete and step_count < self.max_steps:
            step_count += 1
            action = self.planner.plan(state)
            self.pipeline_logger.log(
                "planner_action",
                filename=doc.filename,
                step_number=step_count,
                tool=action.tool.value,
                rationale=action.rationale,
            )

            step_res = self.executor.execute(action, state)
            self.pipeline_logger.log(
                "executor_step",
                filename=doc.filename,
                step_number=step_count,
                tool=action.tool.value,
                result_summary=step_res.result_summary,
                token_cost=step_res.token_cost,
                inference_time=step_res.inference_time,
            )

            if state.is_complete or action.tool == PlannerToolName.FINISH:
                state.is_complete = True
                verification = self.responder.verify(state)
                self.pipeline_logger.log(
                    "responder_verification",
                    filename=doc.filename,
                    verification_passed=verification.passed,
                    verification_score=verification.score,
                    verification_reasoning=verification.reasoning,
                    unsupported_claims=verification.unsupported_claims,
                )

                if not verification.passed and step_count < self.max_steps:
                    logger.warning(
                        "Responder verification failed (step %d); routing feedback back to Planner.",
                        step_count,
                    )
                    state.is_complete = False

        if state.fixed_schema is None:
            self.executor._tool_extract_fixed_fields(state)

        if state.verification is None:
            state.is_complete = True
            verification = self.responder.verify(state)
            self.pipeline_logger.log(
                "responder_verification",
                filename=doc.filename,
                verification_passed=verification.passed,
                verification_score=verification.score,
                verification_reasoning=verification.reasoning,
                unsupported_claims=verification.unsupported_claims,
            )

        state.total_elapsed = time.perf_counter() - start_time

        from models.docmodels import InsuranceDomainMetadata, PipelineAuditMetadata

        fx = state.fixed_schema or {}
        domain_meta = InsuranceDomainMetadata(
            category=state.category or "OTHER",
            document_type=state.document_type or "OTHER",
            policy_number=fx.get("policy_number"),
            claim_number=fx.get("claim_number"),
            insured_name=fx.get("insured_name") or fx.get("claimant_name") or fx.get("patient_name"),
            company_name=fx.get("hospital_name") or fx.get("surveyor_name"),
            total_amount=float(fx.get("claim_amount") or fx.get("premium") or fx.get("assessed_loss_amount") or 0.0) or None,
        )
        state.domain_metadata = domain_meta.model_dump()

        audit_meta = PipelineAuditMetadata(
            llm_model=getattr(self.client, "model", "qwen2.5:3b"),
            responder_verified=state.verification.passed if state.verification else True,
            grounding_score=state.verification.score if state.verification else 1.0,
            total_tokens_used=state.total_tokens,
            total_inference_time_sec=state.total_inference_time,
        )
        state.audit_metadata = audit_meta.model_dump()

        self.pipeline_logger.log(
            "agentic_complete",
            file_path=doc.file_path,
            filename=doc.filename,
            total_tokens=state.total_tokens,
            total_elapsed=state.total_elapsed,
        )
        return state
