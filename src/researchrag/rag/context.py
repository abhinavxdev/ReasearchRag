"""Source labels derive from stable chunk IDs, not retrieval rank."""

from researchrag.core.models import Citation, Hit


def build_context(hits: list[Hit], max_chars: int = 22000) -> tuple[str, dict[str, Citation]]:
    """Construct bounded evidence with an exact snippet-to-ID mapping."""
    blocks: list[str] = []
    sources: dict[str, Citation] = {}
    used = 0
    for hit in hits:
        chunk = hit.chunk
        key = f"S_{chunk.id}"
        block = f"SOURCE [{key}] | {chunk.title} | p.{chunk.page} | {chunk.section}\n{chunk.text}\nEND SOURCE"
        if used + len(block) > max_chars:
            continue
        sources[key] = Citation(
            source_id=key,
            paper_id=chunk.paper_id,
            title=chunk.title,
            page=chunk.page,
            section=chunk.section,
            snippet=chunk.text,
        )
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks), sources
