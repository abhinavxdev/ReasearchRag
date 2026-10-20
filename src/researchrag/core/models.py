"""Serializable domain objects shared by services and transport layers."""

from typing import Literal

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Persistent paper and ingestion lifecycle."""

    id: str
    title: str
    filename: str
    sha256: str
    status: Literal["pending", "processing", "done", "failed"] = "pending"
    error: str | None = None
    arxiv_id: str | None = None
    chunk_count: int = 0


class Chunk(BaseModel):
    """A single-page, single-section citation unit."""

    id: str
    paper_id: str
    title: str
    page: int
    section: str
    text: str


class Hit(BaseModel):
    """Retrieved passage with dense similarity and fusion score."""

    chunk: Chunk
    score: float = 0
    dense_score: float = 0


class Citation(BaseModel):
    """Verifiable passage attached to an answer."""

    source_id: str
    paper_id: str
    title: str
    page: int
    section: str
    snippet: str


class Answer(BaseModel):
    """A validated answer and exactly its cited passages."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    warnings: list[str] = Field(default_factory=list)
