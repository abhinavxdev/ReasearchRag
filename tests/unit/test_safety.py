from pathlib import Path

from researchrag.api.dependencies import build_services
from researchrag.config import Settings
from researchrag.core.models import Chunk, Hit
from researchrag.db.repository import Repository
from researchrag.ingestion.library import LibraryService
from researchrag.rag.service import AnswerService
from researchrag.retrieval.chroma import ChromaStore


def test_low_confidence_never_calls_llm():
    class Retriever:
        def search(self, *args):
            return [
                Hit(
                    chunk=Chunk(id="x", paper_id="p", title="t", page=1, section="s", text="text"),
                    dense_score=0.1,
                )
            ]

    class LLM:
        def stream(self, *args):
            raise AssertionError("LLM must not be called for low confidence")

    assert AnswerService(Retriever(), LLM(), Settings(_env_file=None)).answer("question").refused


def test_recovery(tmp_path: Path):
    from io import BytesIO

    class Embedder:
        def encode(self, texts, *, query=False):
            return [[1, 0] for _ in texts]

    repo = Repository(tmp_path / "db.sqlite")
    settings = Settings(_env_file=None, data_dir=tmp_path)
    service = LibraryService(
        repo, Embedder(), ChromaStore(tmp_path / "vectors", "recovery"), settings
    )
    paper = service.stage(
        BytesIO(Path("tests/fixtures/research.pdf").read_bytes()),
        "../../safe.pdf",
        "application/pdf",
    )
    assert paper.filename == "safe.pdf"
    service.recover()
    assert repo.get(paper.id).status == "failed"
    assert "restarted" in repo.get(paper.id).error


def test_empty_library_refuses(tmp_path: Path):
    from unittest.mock import Mock

    services = build_services(
        Settings(_env_file=None, data_dir=tmp_path), embedder=Mock(), llm=Mock()
    )
    assert services.answers.answer("What is quantum gravity?").refused
