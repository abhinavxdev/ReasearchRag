"""PDF extraction with a parser fallback and explicit scan detection."""

from pathlib import Path

import pymupdf
from pypdf import PdfReader

from researchrag.core.exceptions import ResearchError


def extract(path: Path, max_pages: int = 500) -> tuple[str, list[str]]:
    """Return metadata title and page texts, preserving 1-based page provenance."""
    try:
        with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
            if doc.is_encrypted:
                raise ResearchError("Encrypted PDFs are not supported")
            if len(doc) > max_pages:
                raise ResearchError(f"PDF exceeds {max_pages} pages")
            title = str((doc.metadata or {}).get("title") or "").strip()
            if not title and len(doc):
                spans = [
                    span
                    for block in doc[0].get_text("dict")["blocks"]
                    for line in block.get("lines", [])
                    for span in line["spans"]
                    if len(span["text"].strip()) > 4
                ]
                if spans:
                    largest = max(span["size"] for span in spans)
                    title = " ".join(
                        span["text"].strip() for span in spans if span["size"] >= largest - 0.5
                    )[:240]
            pages = [page.get_text(sort=True) for page in doc]
    except ResearchError:
        raise
    except Exception:
        try:
            reader = PdfReader(path)
            if reader.is_encrypted or len(reader.pages) > max_pages:
                raise ResearchError("Encrypted PDF or page limit exceeded")
            title = str(reader.metadata.title or "") if reader.metadata else ""
            pages = [page.extract_text() or "" for page in reader.pages]
        except ResearchError:
            raise
        except Exception as exc:
            raise ResearchError("Invalid or unreadable PDF") from exc
    if sum(len(p.strip()) for p in pages) < 80:
        raise ResearchError("Image-only/scanned PDF detected; OCR is required before upload")
    return title, pages
