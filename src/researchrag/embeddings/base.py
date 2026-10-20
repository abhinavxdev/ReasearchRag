"""Embedding port for model implementations and deterministic test doubles."""

from typing import Protocol


class Embedder(Protocol):
    """Normalized vectors in a shared document/query space."""

    def encode(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        """Embed a batch, distinguishing queries from document passages."""
        ...
