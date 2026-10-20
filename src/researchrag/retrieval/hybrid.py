"""Reciprocal rank fusion over dense and keyword retrieval."""

from collections import defaultdict

from researchrag.core.models import Hit
from researchrag.db.repository import Repository
from researchrag.embeddings.base import Embedder
from researchrag.retrieval.base import VectorStore
from researchrag.retrieval.bm25 import keyword_search
from researchrag.retrieval.reranker import Reranker


def rrf(rankings: list[list[str]], constant: int = 60) -> list[tuple[str, float]]:
    """Fuse unique IDs per ranking, independent of incompatible raw scores."""
    if constant < 1:
        raise ValueError("RRF constant must be positive")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] += 1 / (constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridRetriever:
    """SQLite gates visibility, including during partial ingestion failures."""

    def __init__(
        self,
        repo: Repository,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None = None,
    ) -> None:
        self.repo, self.embedder, self.store, self.reranker = repo, embedder, store, reranker

    def search(self, query: str, top_k: int = 6, paper_ids: list[str] | None = None) -> list[Hit]:
        """Retrieve candidates with mandatory paper selection in both search paths."""
        chunks = self.repo.chunks(paper_ids)
        if not chunks:
            return []
        available = {chunk.id: chunk for chunk in chunks}
        selected = sorted({chunk.paper_id for chunk in chunks})
        vector = self.embedder.encode([query], query=True)[0]
        dense = self.store.search(vector, max(30, top_k * 4), selected)
        dense_scores = {key: score for key, score in dense if key in available}
        keywords = keyword_search(query, chunks, max(30, top_k * 4))
        fused = rrf([list(dense_scores), keywords])
        candidates = [
            Hit(chunk=available[key], score=score, dense_score=dense_scores.get(key, 0))
            for key, score in fused
            if key in available
        ]
        if self.reranker:
            candidates = self.reranker.rank(query, candidates[: max(30, top_k * 4)])
        return candidates[:top_k]
