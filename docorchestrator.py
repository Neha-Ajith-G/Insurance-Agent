"""
DocOrchestrator: runs the two-turn classify -> extract pipeline,
retrying each turn on failure and tracking token usage/time per turn.
"""
from __future__ import annotations

import logging
import time

from docclient import DocClient, DocClientError
from docloader import DocLoader, DocLoaderError
from docmodels import (
    CATEGORY_TYPE_MAP,
    DEFAULT_EXTRACTION_MODEL,
    EXTRACTION_MODEL_MAP,
    ClassificationResult,
    DocumentType,
    ExtractionResult,
    PipelineResult,
    PipelineStage,
    TokenStats,
)
from docparsers import DocParserError, parse_classification, parse_extraction
from doc_pipeline_logger import JsonlPipelineLogger

logger = logging.getLogger(__name__)

# Backoff applied only before retrying after a client/connection failure --
# parse failures retry immediately since there's nothing transient to wait
# out, just a bad completion to correct.
CLIENT_RETRY_BACKOFF_SECONDS = 0.5

# Per-call char budget sent to the model. Now used as a *chunk size*
# rather than a hard truncation cutoff -- documents longer than this are
# split into multiple overlapping chunks (see _chunk_text) instead of
# having content silently discarded.
DEFAULT_MAX_DOCUMENT_CHARS = 8000

# Overlap between adjacent chunks, so a field that would otherwise land
# right on a chunk boundary has a reasonable chance of appearing whole
# in at least one chunk rather than being split across two.
CHUNK_OVERLAP_CHARS = 500


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping chunks of at most chunk_size chars.
    Returns [text] unchanged (as a single "chunk") if it already fits --
    callers don't need a special case for the common short-document path.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def _merge_extraction_fields(fields_list: list) -> object:
    """
    Merge extraction results from multiple chunks of the same document
    into one, field by field:
    - List fields (e.g. key_points, line_items): concatenate across
      chunks and drop exact duplicates, preserving first-seen order.
    - Scalar fields: first non-null value found across chunks wins --
      the field only needed to be correctly extracted from *a* chunk,
      not every chunk.
    Generic across whatever Pydantic model is passed in (works for any
    of the extraction models without hardcoding field names), same
    reasoning as the prompt-building functions already in this file.
    """
    if len(fields_list) == 1:
        return fields_list[0]

    model_cls = type(fields_list[0])
    merged: dict = {}
    for field_name in model_cls.model_fields:
        values = [getattr(f, field_name) for f in fields_list]
        if any(isinstance(v, list) for v in values):
            merged_list: list = []
            for v in values:
                if isinstance(v, list):
                    for item in v:
                        if item not in merged_list:
                            merged_list.append(item)
            merged[field_name] = merged_list
        else:
            merged[field_name] = next((v for v in values if v is not None), None)
    return model_cls.model_validate(merged)



# Appended to the system prompt on any retry attempt after a parse
# failure, so the model is told concretely what went wrong instead of
# just being asked the same question again. The reason is truncated to
# the first line / a short prefix -- a full Pydantic ValidationError can
# be a dozen lines long and would bloat the retry prompt considerably.
STRICT_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as valid JSON "
    "(reason: {reason}). Return ONLY a single valid JSON object. Do not "
    "use markdown code fences. Do not include any explanation, reasoning, "
    "or text outside the JSON object."
)


def _short_reason(err: Exception, limit: int = 150) -> str:
    """Collapse a (possibly multi-line) error into a short retry hint."""
    first_line = str(err).strip().split("\n", 1)[0]
    return first_line[:limit] + ("..." if len(first_line) > limit else "")


def _build_classify_system_prompt() -> str:
    """
    Build the classification system prompt from CATEGORY_TYPE_MAP so the
    category/type list stays in sync with doc_models.py automatically --
    no hand-maintained duplicate list to drift out of date.
    """
    lines = [
        "You are an insurance document classifier. Read the document text "
        "and assign it a category and a specific document type from the "
        "lists below. Only choose a document_type that is listed under "
        "its matching category.",
        "",
    ]
    for category, types in CATEGORY_TYPE_MAP.items():
        type_names = ", ".join(t.value for t in types)
        lines.append(f"{category.value}: {type_names}")
    lines.append("")
    lines.append(
        "Respond with ONLY a JSON object, no other text, matching this "
        'shape: {"category": "CLAIMS", "document_type": "CLAIM_FORM", '
        '"confidence": 0.9, "reasoning": "short reason"}'
    )
    return "\n".join(lines)


def _build_extract_system_prompt(document_type: DocumentType) -> str:
    """
    Build the extraction system prompt from the target Pydantic model's
    field names, rather than hand-writing one prompt per document type --
    scales to new types without editing this file.
    """
    model_cls = EXTRACTION_MODEL_MAP.get(document_type, DEFAULT_EXTRACTION_MODEL)
    field_names = ", ".join(model_cls.model_fields.keys())
    return (
        f"Extract ONLY the following fields: {field_names}. "
        f"Return exactly one JSON object matching field names exactly. "
        f"Do not invent values that aren't in the document. If a field is "
        f"missing, use null (or [] for list fields). Do not output markdown."
    )


CLASSIFY_SYSTEM_PROMPT = _build_classify_system_prompt()


class DocOrchestratorError(Exception):
    """Raised when a pipeline turn fails even after retries, or when a
    document can't be run through the pipeline at all (e.g. needs OCR)."""


class DocOrchestrator:
    def __init__(
        self,
        loader: DocLoader | None = None,
        client: DocClient | None = None,
        max_retries: int = 2,
        max_document_chars: int = DEFAULT_MAX_DOCUMENT_CHARS,
        pipeline_logger: JsonlPipelineLogger | None = None,
    ):
        self.loader = loader or DocLoader()
        self.client = client or DocClient()
        self.max_retries = max_retries
        self.max_document_chars = max_document_chars
        self.pipeline_logger = pipeline_logger or JsonlPipelineLogger()

    def run(self, path: str) -> PipelineResult:
        pipeline_start = time.perf_counter()
        self.pipeline_logger.log("pipeline_start", file_path=path)
        try:
            result = self._run(path, pipeline_start)
        except (DocOrchestratorError, DocLoaderError) as e:
            self.pipeline_logger.log("pipeline_error", file_path=path, error=str(e))
            raise
        except Exception as e:
            # Catch-all so any unexpected failure (a bug, a library
            # surprise) is logged consistently and surfaced as a
            # DocOrchestratorError rather than an opaque traceback.
            logger.exception("Unexpected pipeline failure for %s", path)
            self.pipeline_logger.log(
                "pipeline_error", file_path=path, error=f"Unexpected pipeline failure: {e}"
            )
            raise DocOrchestratorError(f"Unexpected pipeline failure: {e}") from e
        else:
            self.pipeline_logger.log(
                "pipeline_complete",
                file_path=result.file_path,
                filename=result.filename,
                total_tokens=result.total_tokens,
                elapsed=result.pipeline_elapsed,
            )
            return result

    def _run(self, path: str, pipeline_start: float) -> PipelineResult:
        health_check = getattr(self.client, "health", None)
        if callable(health_check) and not health_check():
            raise DocOrchestratorError(
                "LLM backend health check failed -- is Ollama running?"
            )

        logger.info("Pipeline starting: %s", path)
        doc = self.loader.load(path)
        self.pipeline_logger.log(
            "document_loaded",
            file_path=doc.file_path,
            filename=doc.filename,
        )

        if doc.needs_ocr:
            self.pipeline_logger.log(
                "ocr_required", file_path=doc.file_path, filename=doc.filename
            )
            raise DocOrchestratorError(
                f"{doc.filename} has no extractable text and needs OCR, "
                f"which is out of scope for this pipeline."
            )

        text = doc.text
        chunks = _chunk_text(text, self.max_document_chars, CHUNK_OVERLAP_CHARS)
        if len(chunks) > 1:
            logger.info(
                "%s: %d chars split into %d overlapping chunks (chunk_size=%d) "
                "-- classify runs on the first chunk, extract runs on every chunk",
                doc.filename, len(text), len(chunks), self.max_document_chars,
            )
            self.pipeline_logger.log(
                "document_chunked",
                file_path=doc.file_path,
                filename=doc.filename,
                error=f"{len(text)} chars split into {len(chunks)} chunks of {self.max_document_chars}",
            )

        # Classification only needs to determine a single type label, not
        # every field in the document -- runs on the first chunk rather
        # than being chunked+voted across all of them, which would double
        # the complexity of this change for comparatively little benefit.
        classification, classify_stats = self._classify(chunks[0], doc.filename)
        logger.info(
            "Classification complete for %s: %s / %s",
            doc.filename, classification.category.value, classification.document_type.value,
        )

        extraction, extract_stats_list = self._extract_chunks(chunks, classification, doc.filename)
        logger.info("Extraction complete for %s (%d chunk(s))", doc.filename, len(chunks))

        pipeline_elapsed = time.perf_counter() - pipeline_start
        result = PipelineResult(
            file_path=doc.file_path,
            filename=doc.filename,
            classification=classification,
            extraction=extraction,
            token_stats=[classify_stats, *extract_stats_list],
            pipeline_elapsed=pipeline_elapsed,
        )
        logger.info("Pipeline finished: %s (%.3fs)", path, pipeline_elapsed)
        return result

    def _classify(self, text: str, filename: str) -> tuple[ClassificationResult, TokenStats]:
        system_prompt = CLASSIFY_SYSTEM_PROMPT
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            self.pipeline_logger.log(
                "classify_attempt", filename=filename, stage="classification", attempt=attempt
            )
            try:
                resp = self.client.generate(prompt=text, system=system_prompt)
            except DocClientError as e:
                last_err = e
                logger.warning(
                    "Classify turn (%s): LLM call failed (attempt %d): %s", filename, attempt, e
                )
                self.pipeline_logger.log(
                    "classify_client_error", filename=filename, stage="classification",
                    attempt=attempt, error=str(e),
                )
                time.sleep(CLIENT_RETRY_BACKOFF_SECONDS * attempt)
                continue
            logger.debug("Classify turn (%s): raw reply: %s", filename, resp.text)
            try:
                result = parse_classification(resp.text)
            except DocParserError as e:
                last_err = e
                logger.warning(
                    "Classify turn (%s): parse failed (attempt %d): %s", filename, attempt, e
                )
                self.pipeline_logger.log(
                    "classify_parse_error", filename=filename, stage="classification",
                    attempt=attempt, error=str(e),
                    input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                    total_tokens=resp.total_tokens, inference_time=resp.inference_time,
                )
                system_prompt = CLASSIFY_SYSTEM_PROMPT + STRICT_RETRY_SUFFIX.format(
                    reason=_short_reason(e)
                )
                continue
            stats = TokenStats(
                turn=PipelineStage.CLASSIFICATION,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                total_tokens=resp.total_tokens,
                inference_time=resp.inference_time,
            )
            logger.info(
                "Classify turn (%s): %s / %s (confidence=%s) -- reasoning: %s",
                filename, result.category.value, result.document_type.value,
                result.confidence, result.reasoning,
            )
            self.pipeline_logger.log(
                "classify_success", filename=filename, stage="classification", attempt=attempt,
                category=result.category.value, document_type=result.document_type.value,
                confidence=result.confidence, reasoning=result.reasoning,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                total_tokens=resp.total_tokens, inference_time=resp.inference_time,
            )
            return result, stats

        self.pipeline_logger.log(
            "classify_exhausted", filename=filename, stage="classification", error=str(last_err)
        )
        raise DocOrchestratorError(
            f"Classification failed after {self.max_retries + 1} attempts: {last_err}"
        )

    def _extract_chunks(
        self, chunks: list[str], classification: ClassificationResult, filename: str
    ) -> tuple[ExtractionResult, list[TokenStats]]:
        """
        Runs extraction over every chunk (each with its own independent
        retry logic via _extract, unchanged) and merges the results into
        one ExtractionResult. For the common single-chunk case this is
        just a thin pass-through with no merge step.
        """
        if len(chunks) == 1:
            result, stats = self._extract(chunks[0], classification, filename)
            return result, [stats]

        results: list[ExtractionResult] = []
        stats_list: list[TokenStats] = []
        for i, chunk in enumerate(chunks, start=1):
            logger.info(
                "Extracting chunk %d/%d for %s", i, len(chunks), filename
            )
            result, stats = self._extract(chunk, classification, filename)
            results.append(result)
            stats_list.append(stats)

        merged_fields = _merge_extraction_fields([r.fields for r in results])
        merged_result = ExtractionResult(
            category=classification.category,
            document_type=classification.document_type,
            fields=merged_fields,
        )
        self.pipeline_logger.log(
            "extract_chunks_merged",
            filename=filename, stage="extraction",
            document_type=classification.document_type.value,
            error=f"merged {len(chunks)} chunk results",
        )
        return merged_result, stats_list

    def _extract(
        self, text: str, classification: ClassificationResult, filename: str
    ) -> tuple[ExtractionResult, TokenStats]:
        document_type = classification.document_type
        base_prompt = _build_extract_system_prompt(document_type)
        system_prompt = base_prompt
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            self.pipeline_logger.log(
                "extract_attempt", filename=filename, stage="extraction", attempt=attempt,
                document_type=document_type.value,
            )
            try:
                resp = self.client.generate(prompt=text, system=system_prompt)
            except DocClientError as e:
                last_err = e
                logger.warning(
                    "Extract turn (%s): LLM call failed (attempt %d): %s", filename, attempt, e
                )
                self.pipeline_logger.log(
                    "extract_client_error", filename=filename, stage="extraction",
                    attempt=attempt, document_type=document_type.value, error=str(e),
                )
                time.sleep(CLIENT_RETRY_BACKOFF_SECONDS * attempt)
                continue
            logger.debug("Extract turn (%s): raw reply: %s", filename, resp.text)
            try:
                result = parse_extraction(resp.text, classification.category, document_type)
            except DocParserError as e:
                last_err = e
                logger.warning(
                    "Extract turn (%s): parse failed (attempt %d): %s", filename, attempt, e
                )
                self.pipeline_logger.log(
                    "extract_parse_error", filename=filename, stage="extraction",
                    attempt=attempt, document_type=document_type.value, error=str(e),
                    input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                    total_tokens=resp.total_tokens, inference_time=resp.inference_time,
                )
                system_prompt = base_prompt + STRICT_RETRY_SUFFIX.format(reason=_short_reason(e))
                continue
            stats = TokenStats(
                turn=PipelineStage.EXTRACTION,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                total_tokens=resp.total_tokens,
                inference_time=resp.inference_time,
            )
            self.pipeline_logger.log(
                "extract_success", filename=filename, stage="extraction", attempt=attempt,
                category=classification.category.value, document_type=document_type.value,
                input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                total_tokens=resp.total_tokens, inference_time=resp.inference_time,
            )
            return result, stats

        self.pipeline_logger.log(
            "extract_exhausted", filename=filename, stage="extraction",
            document_type=document_type.value, error=str(last_err),
        )
        raise DocOrchestratorError(
            f"Extraction failed after {self.max_retries + 1} attempts: {last_err}"
        )