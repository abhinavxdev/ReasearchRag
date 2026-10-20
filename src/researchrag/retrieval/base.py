"""Vector storage port independent of Chroma."""

from typing import Protocol

from researchrag.core.models import Chunk


class VectorStore(Protocol):
    """Store stable chunk IDs and return cosine similarities."""

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Insert or replace a batch of vectors."""
        ...

    def search(
        self, vector: list[float], limit: int, paper_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Search optionally restricted paper IDs."""
        ...

    def delete_paper(self, paper_id: str) -> None:
        """Remove all vectors for one paper."""
        ...
