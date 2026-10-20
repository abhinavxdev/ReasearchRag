"""Section-aware chunks using conservative approximate token budgets."""

import hashlib
import re

from researchrag.core.models import Chunk
from researchrag.ingestion.sectioner import Section


def chunk_sections(
    parts: list[Section], paper_id: str, title: str, tokens: int = 600, overlap: int = 80
) -> list[Chunk]:
    """Chunk within page/section boundaries; stable IDs derive from provenance and content.

    A token is approximated by up to four non-whitespace characters. This is not
    the embedding tokenizer; the embedder separately windows long model inputs.
    """
    if tokens <= overlap or overlap < 0:
        raise ValueError("Chunk budget must exceed nonnegative overlap")
    result = []
    for part in parts:
        words = list(re.finditer(r"\S+", part.text))
        start = 0
        while start < len(words):
            end = start
            used = 0
            while end < len(words):
                cost = max(1, (len(words[end][0]) + 3) // 4)
                if end > start and used + cost > tokens:
                    break
                used += cost
                end += 1
            text = part.text[words[start].start() : words[end - 1].end()]
            key = f"{paper_id}:{part.page}:{part.name}:{start}:{text}"
            result.append(
                Chunk(
                    id=hashlib.sha256(key.encode()).hexdigest()[:24],
                    paper_id=paper_id,
                    title=title,
                    page=part.page,
                    section=part.name,
                    text=text,
                )
            )
            if end == len(words):
                break
            back = end
            used = 0
            while back > start + 1 and used < overlap:
                back -= 1
                used += max(1, (len(words[back][0]) + 3) // 4)
            start = max(start + 1, back)
    return result
