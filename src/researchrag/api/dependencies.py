"""Explicit dependency graph shared by API, tests and CLI."""

from dataclasses import dataclass

from fastapi import Request

from researchrag.config import Settings
from researchrag.db.repository import Repository
from researchrag.embeddings.base import Embedder
from researchrag.embeddings.local import SentenceEmbedder
from researchrag.ingestion.library import LibraryService
from researchrag.llm.anthropic import AnthropicClient
from researchrag.llm.base import LLMClient
from researchrag.llm.ollama import OllamaClient
from researchrag.rag.service import AnswerService
from researchrag.retrieval.base import VectorStore
from researchrag.retrieval.chroma import ChromaStore
from researchrag.retrieval.hybrid import HybridRetriever
from researchrag.retrieval.reranker import Reranker


@dataclass
class Services:
    """Application-scoped service graph, replaceable wholesale in integration tests."""

    library: LibraryService
    answers: AnswerService


def build_services(
    settings: Settings,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    llm: LLMClient | None = None,
) -> Services:
    """Build production adapters unless explicit test adapters are supplied."""
    repo = Repository(settings.data_dir / "researchrag.sqlite")
    embedder = embedder or SentenceEmbedder(settings.embedding_model)
    store = store or ChromaStore(settings.data_dir / "chroma", settings.embedding_model)
    if llm is None:
        llm = (
            AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
            if settings.llm_provider == "anthropic"
            else OllamaClient(
                settings.ollama_url,
                settings.ollama_model,
                settings.ollama_context_size,
                settings.ollama_batch_size,
            )
        )
    retriever = HybridRetriever(repo, embedder, store, Reranker() if settings.rerank else None)
    return Services(
        LibraryService(repo, embedder, store, settings), AnswerService(retriever, llm, settings)
    )


def get_services(request: Request) -> Services:
    """Resolve app-scoped dependency without hidden globals."""
    services: Services = request.app.state.services
    return services
