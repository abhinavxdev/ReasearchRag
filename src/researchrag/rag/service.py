"""Grounded answer orchestration independent of HTTP and UI frameworks."""

from collections.abc import Iterator

from researchrag.config import Settings
from researchrag.core.models import Answer
from researchrag.llm.base import LLMClient
from researchrag.llm.prompts import REFUSAL, SYSTEM
from researchrag.rag.citations import validate_answer
from researchrag.rag.context import build_context
from researchrag.retrieval.hybrid import HybridRetriever


class AnswerService:
    """Confidence-gate retrieval and publish only validated model completions."""

    def __init__(self, retriever: HybridRetriever, llm: LLMClient, settings: Settings) -> None:
        self.retriever, self.llm, self.settings = retriever, llm, settings

    def answer(
        self, question: str, paper_ids: list[str] | None = None, top_k: int | None = None
    ) -> Answer:
        """Generate a source-backed answer, or an explicit refusal."""
        hits = self.retriever.search(question, top_k or self.settings.top_k, paper_ids)
        # RRF is rank-only and cannot be used as a calibrated confidence score.
        confident = [hit for hit in hits if hit.dense_score >= self.settings.min_dense_score]
        if not confident:
            return Answer(answer=REFUSAL, refused=True)
        context, sources = build_context(confident, self.settings.context_chars)
        if not sources:
            return Answer(answer=REFUSAL, refused=True)
        prompt = f"CONTEXT\n{context}\n\nQUESTION\n{question}\n\nAnswer in one or two sentences. End EVERY sentence with a source ID copied from CONTEXT in square brackets. Answer:"
        parts: list[str] = []
        size = 0
        for fragment in self.llm.stream(SYSTEM, prompt):
            size += len(fragment)
            if size > 24000:
                return Answer(
                    answer=REFUSAL, refused=True, warnings=["Model output limit exceeded"]
                )
            parts.append(fragment)
        return validate_answer("".join(parts), sources)

    def events(
        self, question: str, paper_ids: list[str] | None = None, top_k: int | None = None
    ) -> Iterator[dict[str, object]]:
        """Stream progress then validated text; never leak unchecked provider tokens."""
        yield {
            "event": "status",
            "data": {"message": "Retrieving evidence and validating citations"},
        }
        result = self.answer(question, paper_ids, top_k)
        text = result.answer
        for start in range(0, len(text), 48):
            yield {"event": "token", "data": {"text": text[start : start + 48]}}
        yield {"event": "done", "data": result.model_dump()}
