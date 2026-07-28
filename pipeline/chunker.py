"""
2-Stage Contextual & Structural Chunker.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pipeline.docclient import DocClient

logger = logging.getLogger(__name__)

try:
    import tiktoken

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TOKENIZER = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _TOKENIZER is not None:
        try:
            return len(_TOKENIZER.encode(text))
        except Exception:
            pass
    words = len(text.split())
    return max(1, int(words * 1.3))


class ChunkMetadata(BaseModel):
    doc_id: str | None = None
    filename: str | None = None
    company: str | None = None
    fiscal_year: str | None = None
    policy_number: str | None = None
    claim_number: str | None = None
    insured_name: str | None = None
    section_breadcrumbs: list[str] = Field(default_factory=list)
    page_number: int | None = None
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    is_table: bool = False
    table_title: str | None = None
    extra: dict = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str
    metadata: ChunkMetadata
    vector_text: str = ""
    context_summary: str = ""
    token_count: int = 0
    vector_token_count: int = 0

    @property
    def token_inflation_pct(self) -> float:
        if self.token_count <= 0:
            return 0.0
        return round(((self.vector_token_count - self.token_count) / self.token_count) * 100, 2)


class StructuralSplitter:
    def __init__(
        self,
        target_chunk_tokens: int = 400,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 80,
    ):
        self.target_chunk_tokens = target_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def _extract_doc_metadata(self, text: str) -> dict[str, str | None]:
        res: dict[str, str | None] = {
            "company": None,
            "fiscal_year": None,
            "policy_number": None,
            "claim_number": None,
            "insured_name": None,
        }

        company_match = re.search(
            r"(?i)\b([A-Z][A-Za-z0-9\s&]{2,30}?(?:Insurance|Assurance|Health|Corp|Ltd|Inc))\b",
            text[:2000],
        )
        if company_match:
            res["company"] = company_match.group(1).strip()

        fy_match = re.search(r"(?i)\b(?:FY\s*)?(20[1-2][0-9](?:-[0-9]{2,4})?)\b", text[:2000])
        if fy_match:
            res["fiscal_year"] = fy_match.group(1).strip()

        pol_match = re.search(r"(?i)\b(?:POL|Policy)[-\s#]*([A-Z0-9-]{4,20})\b", text[:3000])
        if pol_match:
            res["policy_number"] = pol_match.group(1).strip()

        claim_match = re.search(r"(?i)\b(?:CLM|Claim)[-\s#]*([A-Z0-9-]{4,20})\b", text[:3000])
        if claim_match:
            res["claim_number"] = claim_match.group(1).strip()

        name_match = re.search(r"(?i)\b(?:Insured|Claimant|Patient)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text[:3000])
        if name_match:
            res["insured_name"] = name_match.group(1).strip()

        return res

    def _split_by_tables_and_headers(self, text: str) -> list[tuple[list[str], str, bool]]:
        lines = text.split("\n")
        blocks: list[tuple[list[str], str, bool]] = []

        current_breadcrumbs: list[str] = []
        current_lines: list[str] = []
        in_table = False
        table_lines: list[str] = []

        header_md_re = re.compile(r"^(#{1,6})\s+(.+)$")
        header_plain_re = re.compile(r"^(?:SECTION|ARTICLE|PART|CHAPTER|\d+\.|\d+\))\s+[A-Za-z0-9\s:-]{3,60}$", re.IGNORECASE)
        table_row_re = re.compile(r"^\s*\|.*\|\s*$|^(?:[^\t\n]+\t){2,}[^\t\n]+$")

        for line in lines:
            stripped_line = line.strip()
            header_md_match = header_md_re.match(line)
            header_plain_match = bool(header_plain_re.match(stripped_line)) if not header_md_match else False
            is_table_row = bool(table_row_re.match(line))

            if is_table_row:
                if not in_table:
                    if current_lines:
                        block_str = "\n".join(current_lines).strip()
                        if block_str:
                            blocks.append((list(current_breadcrumbs), block_str, False))
                        current_lines = []
                    in_table = True
                table_lines.append(line)
                continue

            if in_table and not is_table_row:
                in_table = False
                tbl_str = "\n".join(table_lines).strip()
                if tbl_str:
                    blocks.append((list(current_breadcrumbs), tbl_str, True))
                table_lines = []

            if header_md_match or header_plain_match:
                if current_lines:
                    block_str = "\n".join(current_lines).strip()
                    if block_str:
                        blocks.append((list(current_breadcrumbs), block_str, False))
                    current_lines = []

                if header_md_match:
                    level = len(header_md_match.group(1))
                    title = header_md_match.group(2).strip()
                else:
                    level = 2
                    title = stripped_line

                if level <= len(current_breadcrumbs):
                    current_breadcrumbs = current_breadcrumbs[: max(0, level - 1)]
                current_breadcrumbs.append(title)
                current_lines.append(line)
            else:
                current_lines.append(line)

        if in_table and table_lines:
            tbl_str = "\n".join(table_lines).strip()
            if tbl_str:
                blocks.append((list(current_breadcrumbs), tbl_str, True))
        elif current_lines:
            block_str = "\n".join(current_lines).strip()
            if block_str:
                blocks.append((list(current_breadcrumbs), block_str, False))

        return blocks

    def _recursive_character_split(
        self, text: str, target_tokens: int
    ) -> list[str]:
        tokens = count_tokens(text)
        if tokens <= target_tokens:
            return [text]

        separators = ["\n\n", "\n", ". ", "; ", " ", ""]
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                sub_chunks: list[str] = []
                acc: list[str] = []
                acc_len = 0

                for part in parts:
                    part_with_sep = part + sep if sep != "" else part
                    part_tokens = count_tokens(part_with_sep)
                    if acc_len + part_tokens > target_tokens and acc:
                        sub_chunks.append("".join(acc).strip())
                        acc = [part_with_sep]
                        acc_len = part_tokens
                    else:
                        acc.append(part_with_sep)
                        acc_len += part_tokens

                if acc:
                    sub_chunks.append("".join(acc).strip())

                if len(sub_chunks) > 1:
                    return [c for c in sub_chunks if c.strip()]

        target_chars = target_tokens * 4
        return [text[i : i + target_chars] for i in range(0, len(text), target_chars)]

    def split(self, text: str) -> list[Chunk]:
        if not text or not text.strip():
            return []

        meta_dict = self._extract_doc_metadata(text)
        blocks = self._split_by_tables_and_headers(text)

        raw_chunks: list[Chunk] = []
        char_cursor = 0

        for breadcrumbs, block_text, is_table in blocks:
            b_tokens = count_tokens(block_text)
            if is_table or b_tokens <= self.target_chunk_tokens:
                sub_blocks = [block_text]
            else:
                sub_blocks = self._recursive_character_split(block_text, self.target_chunk_tokens)

            for sb in sub_blocks:
                sb_clean = sb.strip()
                if not sb_clean:
                    continue
                start_idx = text.find(sb_clean, char_cursor)
                if start_idx == -1:
                    start_idx = char_cursor
                end_idx = start_idx + len(sb_clean)
                char_cursor = end_idx

                t_cnt = count_tokens(sb_clean)
                meta = ChunkMetadata(
                    company=meta_dict["company"],
                    fiscal_year=meta_dict["fiscal_year"],
                    policy_number=meta_dict["policy_number"],
                    claim_number=meta_dict["claim_number"],
                    insured_name=meta_dict["insured_name"],
                    section_breadcrumbs=breadcrumbs,
                    start_char=start_idx,
                    end_char=end_idx,
                    token_count=t_cnt,
                    is_table=is_table,
                )
                raw_chunks.append(
                    Chunk(
                        content=sb_clean,
                        metadata=meta,
                        token_count=t_cnt,
                    )
                )

        merged_chunks: list[Chunk] = []
        for chk in raw_chunks:
            if (
                merged_chunks
                and not chk.metadata.is_table
                and not merged_chunks[-1].metadata.is_table
                and chk.metadata.section_breadcrumbs == merged_chunks[-1].metadata.section_breadcrumbs
                and (merged_chunks[-1].token_count < self.min_chunk_tokens or chk.token_count < self.min_chunk_tokens)
                and (merged_chunks[-1].token_count + chk.token_count <= self.target_chunk_tokens)
            ):
                prev = merged_chunks[-1]
                new_content = f"{prev.content}\n\n{chk.content}"
                new_tokens = count_tokens(new_content)
                prev.content = new_content
                prev.token_count = new_tokens
                prev.metadata.end_char = chk.metadata.end_char
                prev.metadata.token_count = new_tokens
            else:
                merged_chunks.append(chk)

        return merged_chunks


class ContextualEnricher:
    CONTEXT_PROMPT_TEMPLATE = (
        "<document_context>\n"
        "{doc_context}\n"
        "</document_context>\n\n"
        "<section_breadcrumbs>\n"
        "{breadcrumbs}\n"
        "</section_breadcrumbs>\n\n"
        "Here is a chunk from the document:\n"
        "<chunk>\n"
        "{chunk_content}\n"
        "</chunk>\n\n"
        "Please provide a 1-2 sentence context to situate this chunk within the overall "
        "document for retrieval. Answer ONLY with the succinct context text. Do not use code blocks."
    )

    def __init__(self, client: DocClient | None = None):
        self.client = client
        self._summary_cache: dict[str, str] = {}

    def _get_doc_summary(self, full_text: str) -> str:
        cache_key = str(hash(full_text[:3000]))
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        if not self.client:
            summary = full_text[:300].replace("\n", " ").strip()
            self._summary_cache[cache_key] = summary
            return summary

        prompt = (
            "Summarize the main subject, document type, and key entities of this text in 2 short sentences:\n\n"
            f"{full_text[:3000]}"
        )
        try:
            resp = self.client.generate(prompt=prompt)
            summary = resp.text.strip()
        except Exception as e:
            logger.warning("ContextualEnricher: failed to generate doc summary: %s", e)
            summary = full_text[:200].replace("\n", " ").strip()

        self._summary_cache[cache_key] = summary
        return summary

    def enrich_chunks(self, chunks: list[Chunk], full_text: str) -> list[Chunk]:
        if not chunks:
            return []

        doc_summary = self._get_doc_summary(full_text)

        for chk in chunks:
            breadcrumbs_str = " > ".join(chk.metadata.section_breadcrumbs) if chk.metadata.section_breadcrumbs else "General"

            context_summary = ""
            if self.client:
                prompt = self.CONTEXT_PROMPT_TEMPLATE.format(
                    doc_context=doc_summary,
                    breadcrumbs=breadcrumbs_str,
                    chunk_content=chk.content[:1500],
                )
                try:
                    resp = self.client.generate(prompt=prompt)
                    context_summary = resp.text.strip().replace("\n", " ")
                except Exception as e:
                    logger.warning("ContextualEnricher: chunk context generation failed: %s", e)

            if not context_summary:
                meta_parts = []
                if chk.metadata.company:
                    meta_parts.append(f"Company: {chk.metadata.company}")
                if chk.metadata.fiscal_year:
                    meta_parts.append(f"Year: {chk.metadata.fiscal_year}")
                meta_str = ", ".join(meta_parts) if meta_parts else "Insurance Document"
                context_summary = f"This chunk is from '{breadcrumbs_str}' section ({meta_str})."

            chk.context_summary = context_summary

            vector_parts = [
                f"[Context: {context_summary}]",
                f"[Section: {breadcrumbs_str}]",
            ]
            if chk.metadata.company:
                vector_parts.append(f"[Company: {chk.metadata.company}]")
            if chk.metadata.fiscal_year:
                vector_parts.append(f"[Fiscal Year: {chk.metadata.fiscal_year}]")

            vector_parts.append(chk.content)
            chk.vector_text = "\n\n".join(vector_parts)

            chk.token_count = count_tokens(chk.content)
            chk.vector_token_count = count_tokens(chk.vector_text)

        return chunks


class ContextualChunker:
    def __init__(
        self,
        client: DocClient | None = None,
        target_chunk_tokens: int = 400,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 80,
    ):
        self.client = client
        self.splitter = StructuralSplitter(
            target_chunk_tokens=target_chunk_tokens,
            overlap_tokens=overlap_tokens,
            min_chunk_tokens=min_chunk_tokens,
        )
        self.enricher = ContextualEnricher(client=client)

    def chunk_document(self, text: str) -> list[Chunk]:
        if not text or not text.strip():
            return []

        structural_chunks = self.splitter.split(text)
        enriched_chunks = self.enricher.enrich_chunks(structural_chunks, text)

        total_raw_tokens = sum(c.token_count for c in enriched_chunks)
        total_vec_tokens = sum(c.vector_token_count for c in enriched_chunks)
        inflation_pct = round(((total_vec_tokens - total_raw_tokens) / max(1, total_raw_tokens)) * 100, 2)

        logger.info(
            "ContextualChunker complete: %d chunks, %d raw tokens -> %d vector tokens (%.2f%% inflation)",
            len(enriched_chunks),
            total_raw_tokens,
            total_vec_tokens,
            inflation_pct,
        )

        return enriched_chunks
