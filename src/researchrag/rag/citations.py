"""Fail-closed syntactic citation validation and human-readable formatting."""

import re

from researchrag.core.models import Answer, Citation
from researchrag.llm.prompts import REFUSAL

BRACKET = re.compile(r"\[([^\[\]\n]+)\]")


def validate_answer(text: str, sources: dict[str, Citation]) -> Answer:
    """Reject unknown IDs or uncited sentences rather than displaying unsafe partial text.

    This validates provenance and citation coverage, not semantic entailment.
    """
    text = text.strip()
    if text.rstrip(".").lower() == REFUSAL.rstrip(".").lower():
        return Answer(answer=REFUSAL, refused=True)
    ids = BRACKET.findall(text)
    unknown = [key for key in ids if key not in sources]
    if unknown or not ids:
        return Answer(
            answer=REFUSAL,
            refused=True,
            warnings=["Unknown or missing source IDs; answer withheld"],
        )
    # Move citations before sentence-ending punctuation for a consistent splitter.
    normalized = re.sub(r"([.!?])\s*((?:\[S_[^\]]+\]\s*)+)", r" \2\1 ", text)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", normalized) if s.strip()]
    if any(not BRACKET.search(sentence) for sentence in sentences):
        return Answer(answer=REFUSAL, refused=True, warnings=["Uncited sentence; answer withheld"])
    citations = [sources[key] for key in dict.fromkeys(ids)]

    def format_citation(match: re.Match[str]) -> str:
        source = sources[match[1]]
        # Avoid bracket injection from untrusted PDF metadata.
        title = source.title.replace("[", "(").replace("]", ")")
        section = source.section.replace("[", "(").replace("]", ")")
        return f"[{title}, p.{source.page}, {section}]"

    return Answer(answer=BRACKET.sub(format_citation, text), citations=citations)
