"""Developer CLI for inspecting real retrieval and provider answers."""

import argparse
import json

from researchrag.config import get_settings
from researchrag.db.repository import Repository
from researchrag.embeddings.local import SentenceEmbedder
from researchrag.llm.anthropic import AnthropicClient
from researchrag.llm.ollama import OllamaClient
from researchrag.rag.service import AnswerService
from researchrag.retrieval.chroma import ChromaStore
from researchrag.retrieval.hybrid import HybridRetriever


def main() -> None:
    """Ask one question against the persisted paper library."""
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    args = parser.parse_args()
    settings = get_settings()
    repo = Repository(settings.data_dir / "researchrag.sqlite")
    embedder = SentenceEmbedder(settings.embedding_model)
    store = ChromaStore(settings.data_dir / "chroma", settings.embedding_model)
    llm = (
        AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
        if settings.llm_provider == "anthropic"
        else OllamaClient(settings.ollama_url, settings.ollama_model)
    )
    service = AnswerService(HybridRetriever(repo, embedder, store), llm, settings)
    print(json.dumps(service.answer(args.question).model_dump(), indent=2))


if __name__ == "__main__":
    main()
