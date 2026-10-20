"""Local keyword ranking over the authoritative searchable source set."""

import re

from rank_bm25 import BM25Okapi

from researchrag.core.models import Chunk


def tokenize(text: str) -> list[str]:
    """Case-insensitive word tokens, retaining scientific acronyms and numbers."""
    return re.findall(r"\b\w+\b", text.lower())


def keyword_search(query: str, chunks: list[Chunk], limit: int) -> list[str]:
    """Return positive-scoring BM25 matches only."""
    if not chunks:
        return []
    model = BM25Okapi([tokenize(chunk.text) or [""] for chunk in chunks])
    scores = model.get_scores(tokenize(query))
    return [
        chunks[i].id
        for i in sorted(range(len(chunks)), key=lambda i: -scores[i])[:limit]
        if scores[i] > 0
    ]
