"""
DocLoader: reads a source document from disk and returns a LoadedDocument.
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
    file_path: str = ""
    filename: str
    file_type: str
    text: str
    page_count: int | None = None
    needs_ocr: bool = False


@runtime_checkable
class FileTypeLoader(Protocol):
    def can_load(self, path: Path) -> bool:
        ...

    def load(self, path: Path) -> LoadedDocument:
        ...


class TextFileLoader:
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


def _perform_ocr_on_bytes(image_bytes: bytes) -> str:
    try:
        import io
        import pytesseract
        from PIL import Image

        pil_img = Image.open(io.BytesIO(image_bytes))
        return pytesseract.image_to_string(pil_img).strip()
    except Exception as e:
        logger.debug("Automatic OCR on image bytes failed: %s", e)
        return ""


def _perform_ocr_on_file(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image

        pil_img = Image.open(path)
        return pytesseract.image_to_string(pil_img).strip()
    except Exception as e:
        logger.warning("Automatic OCR on image file %s failed: %s", path, e)
        return ""


class PdfFileLoader:
    SUPPORTED_SUFFIXES = {".pdf"}

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def load(self, path: Path) -> LoadedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise DocLoaderError(
                "pypdf is required for PDF loading. Install with: pip install pypdf"
            ) from e

        try:
            reader = PdfReader(str(path))
        except Exception as e:
            raise DocLoaderError(f"Could not open PDF {path}: {e}") from e

        pages_text = []
        for i, page in enumerate(reader.pages):
            p_text = (page.extract_text() or "").strip()
            ocr_text_parts = []

            if hasattr(page, "images") and page.images:
                for img_obj in page.images:
                    try:
                        ocr_res = _perform_ocr_on_bytes(img_obj.data)
                        if ocr_res and ocr_res not in p_text:
                            ocr_text_parts.append(ocr_res)
                    except Exception:
                        pass

            combined_page_text = p_text
            if ocr_text_parts:
                combined_page_text = (p_text + "\n" + "\n".join(ocr_text_parts)).strip()

            pages_text.append(combined_page_text)

        full_text = "\n\n".join(t for t in pages_text if t)
        page_count = len(reader.pages)

        return LoadedDocument(
            filename=path.name,
            file_type="pdf",
            text=_collapse_letter_spacing(full_text),
            page_count=page_count,
            needs_ocr=False,
        )


class DocxFileLoader:
    SUPPORTED_SUFFIXES = {".docx"}

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def load(self, path: Path) -> LoadedDocument:
        try:
            import docx
        except ImportError as e:
            raise DocLoaderError(
                "python-docx is required for DOCX loading. Install with: pip install python-docx"
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

        return LoadedDocument(
            filename=path.name,
            file_type="docx",
            text=text,
            page_count=None,
        )


def _collapse_letter_spacing(text: str) -> str:
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
    if not text:
        return text

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\t", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


class ImageFileLoader:
    SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

    def can_load(self, path: Path) -> bool:
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def load(self, path: Path) -> LoadedDocument:
        ocr_text = _perform_ocr_on_file(path)
        return LoadedDocument(
            filename=path.name,
            file_type="image",
            text=ocr_text,
            page_count=1,
            needs_ocr=False,
        )


class DocLoader:
    def __init__(self, loaders: list[FileTypeLoader] | None = None):
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
                f"No loader registered for file type {p.suffix!r}: {path}. Supported: {supported}"
            )

        start = time.perf_counter()
        doc = loader.load(p)
        elapsed = time.perf_counter() - start

        doc.file_path = str(p)

        if doc.text:
            doc.text = _normalize_text(doc.text)

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
