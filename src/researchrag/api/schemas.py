"""Validated API request contracts."""

from pydantic import BaseModel, Field, field_validator


class ArxivRequest(BaseModel):
    """Restricted ID or arxiv.org URL."""

    arxiv: str = Field(min_length=1, max_length=300)


class QueryRequest(BaseModel):
    """None selects all papers; an explicit empty list selects none."""

    question: str = Field(min_length=1, max_length=2000)
    paper_ids: list[str] | None = Field(default=None, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        """Reject whitespace-only questions."""
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()
