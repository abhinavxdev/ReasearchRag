"""Prepare deterministic source chunks; persistence is coordinated by the library service."""

from pathlib import Path

from researchrag.config import Settings
from researchrag.core.models import Chunk
from researchrag.ingestion.chunker import chunk_sections
from researchrag.ingestion.cleaner import clean_pages
from researchrag.ingestion.pdf import extract
from researchrag.ingestion.sectioner import sections


def prepare(
    path: Path, paper_id: str, fallback_title: str, settings: Settings
) -> tuple[str, list[Chunk]]:
    """Extract, normalize, section and chunk a PDF without losing page provenance."""
    title, pages = extract(path, settings.max_pages)
    if not title:
        title = next(
            (line.strip() for line in pages[0].splitlines() if len(line.strip()) > 10),
            fallback_title,
        )[:240]
    chunks = chunk_sections(
        sections(clean_pages(pages)),
        paper_id,
        title,
        settings.chunk_tokens,
        settings.overlap_tokens,
    )
    return title, chunks
