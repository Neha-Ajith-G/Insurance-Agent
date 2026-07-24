"""
DocLoader: reads a source document from disk and returns a LoadedDocument
with extracted text plus metadata (file type, page count, OCR status).

"""
from __future__ import annotations

import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocLoaderError(Exception):
    """Raised when a document cannot be loaded."""


class LoadedDocument(BaseModel):
    """Result of loading a document: extracted text plus metadata."""

    file_path: str = ""
    filename: str
    file_type: str
    text: str
    page_count: int | None = None
    needs_ocr: bool = False


@runtime_checkable
class FileTypeLoader(Protocol):
    """Interface every file-type-specific loader must implement."""

    def can_load(self, path: Path) -> bool:
        """Return True if this loader knows how to handle `path`."""
        ...

    def load(self, path: Path) -> LoadedDocument:
        """Return the document's contents and metadata.

        Assumes `path` already exists and is a file -- existence/type
        checks are `DocLoader`'s job, not each individual loader's.
        """
        ...


class TextFileLoader:
    """Loads plain text files (.txt, .md) as-is."""

    SUPPORTED_SUFFIXES = {".txt", ".md"}

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def load(self, path: Path) -> LoadedDocument:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise DocLoaderError(f"Could not decode {path} as UTF-8: {e}") from e

        return LoadedDocument(
            filename=path.name,
            file_type="text",
            text=text,
            page_count=None,
        )


class PdfFileLoader:
    """
    Loads text from PDF files using pypdf's per-page extraction.

    Scope: text-layer PDFs only (the common case for invoices, contracts,
    reports exported from Word/most business tools). Scanned/image-only
    PDFs have no text layer to extract -- rather than failing outright,
    this loader flags `needs_ocr=True` so a downstream OCR step (or the
    orchestrator) can decide what to do with it.

   
    """

    SUPPORTED_SUFFIXES = {".pdf"}

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def load(self, path: Path) -> LoadedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise DocLoaderError(
                "pypdf is required for PDF loading. Install with: "
                "pip install pypdf"
            ) from e

        try:
            reader = PdfReader(str(path))
        except Exception as e:
            raise DocLoaderError(f"Could not open PDF {path}: {e}") from e

        pages_text = []
        for page in reader.pages:
            pages_text.append(page.extract_text() or "")
        text = "\n".join(pages_text)
        page_count = len(reader.pages)

        if not text.strip():
            # No text layer -- likely a scanned/image-only PDF. Flag it
            # instead of raising so callers can route it to OCR.
            logger.warning("No extractable text in %s; flagging needs_ocr", path)
            return LoadedDocument(
                filename=path.name,
                file_type="pdf",
                text="",
                page_count=page_count,
                needs_ocr=True,
            )

        return LoadedDocument(
            filename=path.name,
            file_type="pdf",
            text=_collapse_letter_spacing(text),
            page_count=page_count,
        )


class DocxFileLoader:
    """Loads text from Word documents (.docx), including table cells."""

    SUPPORTED_SUFFIXES = {".docx"}

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def load(self, path: Path) -> LoadedDocument:
        try:
            import docx
        except ImportError as e:
            raise DocLoaderError(
                "python-docx is required for DOCX loading. Install with: "
                "pip install python-docx"
            ) from e

        try:
            document = docx.Document(str(path))
        except Exception as e:
            raise DocLoaderError(f"Could not open DOCX {path}: {e}") from e

        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append("\t".join(cell.text for cell in row.cells))
        text = "\n".join(parts)

        # python-docx has no reliable page count without a rendering
        # engine (page breaks are layout-dependent, not stored as such).
        return LoadedDocument(
            filename=path.name,
            file_type="docx",
            text=text,
            page_count=None,
        )


def _collapse_letter_spacing(text: str) -> str:
    """
    Reconstruct words from PDFs that render with per-character spacing.

    Splits on double-spaces/newlines (real boundaries), then within each
    segment: if most tokens are single characters, treat the whole
    segment as one letter-spaced word and remove the internal spaces.
    Segments that are already normal words are left untouched, so this
    is safe to apply unconditionally rather than only on PDFs known to
    have the problem.
    """
    parts = re.split(r"(\n|  +)", text)
    out = []
    for part in parts:
        if part == "\n" or re.fullmatch(r" {2,}", part or ""):
            out.append(part)
            continue
        tokens = part.split(" ")
        single_char_tokens = sum(1 for t in tokens if len(t) == 1)
        if tokens and single_char_tokens / len(tokens) > 0.6:
            out.append("".join(tokens))
        else:
            out.append(part)
    return "".join(out)


def _normalize_text(text: str) -> str:
    """
    Clean up extracted text before it heads downstream to an LLM.

    - Unicode-normalizes (NFKC) so visually-identical characters from
      different sources (e.g. PDF ligatures, smart quotes) compare and
      tokenize consistently.
    - Converts tabs to single spaces (tabs from raw text extraction are
      usually stray, not intentional alignment).
    - Collapses runs of 3+ blank lines down to a single blank line.
    - Strips trailing whitespace from each line and the document as a
      whole.
    """
    if not text:
        return text

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


class ImageFileLoader:
    """Loads image files (.png, .jpg, .jpeg, .tiff, .bmp) flagged for OCR."""

    SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def load(self, path: Path) -> LoadedDocument:
        return LoadedDocument(
            filename=path.name,
            file_type="image",
            text="",
            needs_ocr=True,
        )


class DocLoader:
    """
    Loads a document from disk, delegating the actual read to whichever
    registered `FileTypeLoader` claims the file's type.
    """

    def __init__(self, loaders: list[FileTypeLoader] | None = None):
        # Default to text + PDF + DOCX + Image support
        self._loaders: list[FileTypeLoader] = (
            loaders
            if loaders is not None
            else [TextFileLoader(), PdfFileLoader(), DocxFileLoader(), ImageFileLoader()]
        )

    def load(self, path: str) -> LoadedDocument:
        p = Path(path)

        if not p.exists():
            raise DocLoaderError(f"No such file: {path}")
        if not p.is_file():
            raise DocLoaderError(f"Not a file: {path}")

        loader = next((l for l in self._loaders if l.can_load(p)), None)
        if loader is None:
            supported = sorted({s for l in self._loaders for s in getattr(l, "SUPPORTED_SUFFIXES", [])})
            raise DocLoaderError(
                f"No loader registered for file type {p.suffix!r}: {path}. "
                f"Supported: {supported}"
            )

        start = time.perf_counter()
        doc = loader.load(p)
        elapsed = time.perf_counter() - start

        doc.file_path = str(p)

        if doc.text:
            doc.text = _normalize_text(doc.text)

        # Scanned/image-only PDFs legitimately have empty text (they're
        # flagged via needs_ocr) -- everything else with empty text
        # after extraction is a real failure.
        if not doc.needs_ocr and not doc.text.strip():
            raise DocLoaderError(f"Document is empty: {path}")

        logger.info(
            "Loaded %s via %s: %d chars, page_count=%s, needs_ocr=%s, %.3fs",
            doc.file_path,
            type(loader).__name__,
            len(doc.text),
            doc.page_count,
            doc.needs_ocr,
            elapsed,
        )

        return doc