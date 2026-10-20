"""Library lifecycle and background ingestion coordination."""

import hashlib
import logging
import re
import threading
import uuid
from pathlib import Path
from typing import BinaryIO

import httpx

from researchrag.config import Settings
from researchrag.core.exceptions import Conflict, ResearchError
from researchrag.core.models import Paper
from researchrag.db.repository import Repository
from researchrag.embeddings.base import Embedder
from researchrag.ingestion.arxiv import download, parse_id
from researchrag.ingestion.pipeline import prepare
from researchrag.retrieval.base import VectorStore

logger = logging.getLogger(__name__)


class LibraryService:
    """Single-process coordination with SQLite as authoritative visibility gate."""

    def __init__(
        self, repo: Repository, embedder: Embedder, store: VectorStore, settings: Settings
    ) -> None:
        self.repo, self.embedder, self.store, self.settings = repo, embedder, store, settings
        self.uploads = settings.data_dir / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ingest_lock = threading.Lock()

    def list(self) -> list[Paper]:
        """List library entries with current ingestion status."""
        return self.repo.papers()

    def get(self, paper_id: str) -> Paper:
        """Read current paper status."""
        return self.repo.get(paper_id)

    def stage(
        self, file: BinaryIO, filename: str, mime: str | None, arxiv_id: str | None = None
    ) -> Paper:
        """Validate bounded bytes and atomically stage a safely named document."""
        if mime not in {"application/pdf", "application/octet-stream"}:
            raise ResearchError("Upload must have PDF content type")
        safe = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(filename.replace("\\", "/")).name)[:160]
        if not safe.lower().endswith(".pdf"):
            raise ResearchError("Filename must end in .pdf")
        maximum = self.settings.max_upload_mb * 1024 * 1024
        data = file.read(maximum + 1)
        if len(data) > maximum:
            raise ResearchError(f"PDF exceeds {self.settings.max_upload_mb} MB size limit")
        if not data.startswith(b"%PDF-"):
            raise ResearchError("Invalid PDF signature")
        digest = hashlib.sha256(data).hexdigest()
        with self._lock:
            existing = self.repo.by_hash(digest)
            if existing:
                return existing
            paper_id = uuid.uuid4().hex
            path = self.uploads / f"{paper_id}.pdf"
            path.write_bytes(data)
            paper = Paper(
                id=paper_id, title=Path(safe).stem, filename=safe, sha256=digest, arxiv_id=arxiv_id
            )
            try:
                self.repo.add(paper)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            return paper

    def import_arxiv(self, value: str) -> Paper:
        """Fetch a restricted arXiv resource and stage it like a normal upload."""
        from io import BytesIO

        arxiv_id = parse_id(value)
        try:
            data = download(arxiv_id, self.settings.max_upload_mb * 1024 * 1024)
        except httpx.HTTPError as exc:
            raise ResearchError("arXiv unavailable after retries; please try again later") from exc
        return self.stage(
            BytesIO(data), arxiv_id.replace("/", "_") + ".pdf", "application/pdf", arxiv_id
        )

    def recover(self) -> None:
        """Mark interrupted jobs failed on startup; user may explicitly re-ingest."""
        for paper in self.repo.papers():
            if paper.status in {"pending", "processing"}:
                self.repo.status(
                    paper.id, "failed", "Server restarted during ingestion; re-ingest to retry"
                )

    def ingest(self, paper_id: str) -> None:
        """Run bounded parallelism, publish only fully indexed source records."""
        with self._ingest_lock:
            with self._lock:
                paper = self.repo.get(paper_id)
                if paper.status != "pending":
                    return
                self.repo.status(paper_id, "processing")
            try:
                title, chunks = prepare(
                    self.uploads / f"{paper_id}.pdf", paper_id, paper.title, self.settings
                )
                vectors = self.embedder.encode([chunk.text for chunk in chunks])
                with self._lock:
                    self.store.delete_paper(paper_id)
                    self.store.upsert(chunks, vectors)
                    self.repo.complete(paper_id, title, chunks)
                logger.info("ingestion_completed paper_id=%s chunks=%d", paper_id, len(chunks))
            except Exception as exc:
                logger.exception("ingestion_failed paper_id=%s", paper_id)
                try:
                    self.store.delete_paper(paper_id)
                except Exception:
                    logger.exception("vector_cleanup_failed paper_id=%s", paper_id)
                error = (
                    str(exc)
                    if isinstance(exc, ResearchError)
                    else "Ingestion failed; check server logs and model availability"
                )
                self.repo.status(paper_id, "failed", error)

    def reingest(self, paper_id: str) -> Paper:
        """Queue another attempt, immediately hiding previously indexed passages."""
        with self._lock:
            paper = self.repo.get(paper_id)
            if paper.status in {"pending", "processing"}:
                raise Conflict("Paper ingestion is already in progress")
            self.repo.status(paper_id, "pending")
            return self.repo.get(paper_id)

    def delete(self, paper_id: str) -> None:
        """Prevent races with active jobs and remove all persisted paper artifacts."""
        with self._lock:
            paper = self.repo.get(paper_id)
            if paper.status in {"pending", "processing"}:
                raise Conflict("Wait for ingestion to finish before deleting")
            self.repo.status(paper_id, "failed", "Deletion in progress")
            self.store.delete_paper(paper_id)
            self.repo.delete(paper_id)
            (self.uploads / f"{paper_id}.pdf").unlink(missing_ok=True)
