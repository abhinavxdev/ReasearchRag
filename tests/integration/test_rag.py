"""Offline full pipeline test with actual PDF, SQLite and persistent Chroma."""

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

from researchrag.config import Settings
from researchrag.core.models import Paper
from researchrag.db.repository import Repository
from researchrag.ingestion.pipeline import prepare
from researchrag.rag.service import AnswerService
from researchrag.retrieval.chroma import ChromaStore
from researchrag.retrieval.hybrid import HybridRetriever


class FakeEmbedder:
    """Deterministic topic vectors; no model downloads during tests."""

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        return [[1.0, 0.1 if "fusion" in text.lower() else 0.0, 0.0] for text in texts]


class FakeLLM:
    """Return a predictable cited claim from the supplied source mapping."""

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        source = re.search(r"SOURCE \[(S_[a-z0-9]+)\]", prompt)
        assert source
        yield f"The system combines BM25 and dense retrieval using reciprocal rank fusion [{source[1]}]."


def test_ingest_query_persist(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path)
    repo = Repository(tmp_path / "test.sqlite")
    embedder = FakeEmbedder()
    store = ChromaStore(tmp_path / "chroma", "test")
    path = Path("tests/fixtures/research.pdf")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    repo.add(Paper(id="fixture", title="Tiny Retrieval Study", filename=path.name, sha256=digest))
    title, chunks = prepare(path, "fixture", path.name, settings)
    store.upsert(chunks, embedder.encode([chunk.text for chunk in chunks]))
    repo.complete("fixture", title, chunks)
    retriever = HybridRetriever(repo, embedder, store)
    result = AnswerService(retriever, FakeLLM(), settings).answer("What is reciprocal rank fusion?")
    assert not result.refused
    assert result.citations
    assert "Tiny Retrieval Study" in result.answer
    assert all(c.snippet in [chunk.text for chunk in chunks] for c in result.citations)
    assert not retriever.search("fusion", paper_ids=[])
    reopened = ChromaStore(tmp_path / "chroma", "test")
    assert reopened.search([1, 0.1, 0], 1)[0][0] in {c.id for c in chunks}
