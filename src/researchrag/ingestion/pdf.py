"""PDF extraction with a parser fallback and explicit scan detection."""

from pathlib import Path

import pymupdf
from pypdf import PdfReader

from researchrag.core.exceptions import ResearchError


def page_text(page: pymupdf.Page) -> str:
    """Read common two-column layouts column-wise, keeping spanning block bands."""
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and b[4].strip()]  # type: ignore[no-untyped-call]
    middle = page.rect.width / 2
    left = [b for b in blocks if b[2] < middle + 12]
    right = [b for b in blocks if b[0] > middle - 12]
    if len(left) < 2 or len(right) < 2:
        return str(page.get_text(sort=True))  # type: ignore[no-untyped-call]
    spans = sorted([b for b in blocks if b not in left and b not in right], key=lambda b: b[1])
    remaining = left + right
    ordered = []
    for span in spans:
        band = [b for b in remaining if (b[1] + b[3]) / 2 < span[1]]
        ordered.extend(sorted(band, key=lambda b: (b[0] >= middle - 12, b[1])))
        remaining = [b for b in remaining if b not in band]
        ordered.append(span)
    ordered.extend(sorted(remaining, key=lambda b: (b[0] >= middle - 12, b[1])))
    return "\n".join(str(b[4]) for b in ordered)


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
                    if abs(line.get("dir", (1, 0))[1]) < 0.1
                    for span in line["spans"]
                    if len(span["text"].strip()) > 4
                ]
                if spans:
                    largest = max(span["size"] for span in spans)
                    title = " ".join(
                        span["text"].strip() for span in spans if span["size"] >= largest - 0.5
                    )[:240]
            pages = [page_text(page) for page in doc]
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
