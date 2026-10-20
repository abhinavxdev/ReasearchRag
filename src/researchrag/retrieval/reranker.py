"""Optional lazy CPU cross encoder; never used to infer confidence thresholds."""

from typing import Any

from researchrag.core.models import Hit


class Reranker:
    """Rerank fused candidates by cross-encoder relevance."""

    def __init__(self) -> None:
        self._model: Any = None

    def rank(self, query: str, hits: list[Hit]) -> list[Hit]:
        """Reorder candidates without changing source provenance."""
        if not hits:
            return []
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        scores = self._model.predict([(query, hit.chunk.text) for hit in hits])
        return [
            hit
            for _, hit in sorted(zip(scores, hits, strict=True), key=lambda pair: -float(pair[0]))
        ]
