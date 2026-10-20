"""Persistent cosine Chroma adapter, one collection per embedding model."""

import hashlib
from pathlib import Path
from typing import cast

import chromadb
import numpy as np
from chromadb.api.types import Where
from chromadb.config import Settings as ChromaSettings

from researchrag.core.models import Chunk


class ChromaStore:
    """Use model-specific collections to prevent dimension/model mixing."""

    def __init__(self, path: Path, model: str) -> None:
        self.client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        name = "papers-" + hashlib.sha256(model.encode()).hexdigest()[:16]
        self.collection = self.client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Write bounded batches of vectors and paper filters."""
        for start in range(0, len(chunks), 128):
            batch = chunks[start : start + 128]
            self.collection.upsert(
                ids=[c.id for c in batch],
                embeddings=np.asarray(vectors[start : start + 128], dtype=np.float32),
                metadatas=[{"paper_id": c.paper_id} for c in batch],
            )

    def search(
        self, vector: list[float], limit: int, paper_ids: list[str] | None = None
    ) -> list[tuple[str, float]]:
        """Return cosine similarity rather than Chroma distance."""
        if paper_ids == [] or self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=np.asarray([vector], dtype=np.float32),
            n_results=min(limit, self.collection.count()),
            where=cast(Where, {"paper_id": {"$in": paper_ids}}) if paper_ids else None,
            include=["distances"],
        )
        distances = result["distances"]
        if distances is None:
            return []
        return [
            (key, max(-1.0, min(1.0, 1 - distance)))
            for key, distance in zip(result["ids"][0], distances[0], strict=True)
        ]

    def delete_paper(self, paper_id: str) -> None:
        """Idempotently remove a paper's indexed embeddings."""
        self.collection.delete(where={"paper_id": paper_id})
