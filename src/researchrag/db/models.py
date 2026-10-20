"""SQLite schema. Document content is local and is never logged."""

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """SQLAlchemy declarative registry."""


class PaperRow(Base):
    """Paper lifecycle and deduplication key."""

    __tablename__ = "papers"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str]
    filename: Mapped[str]
    sha256: Mapped[str] = mapped_column(String, unique=True)
    status: Mapped[str]
    error: Mapped[str | None]
    arxiv_id: Mapped[str | None]
    chunk_count: Mapped[int] = mapped_column(default=0)


class ChunkRow(Base):
    """Authoritative source text and provenance."""

    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id"), index=True)
    title: Mapped[str]
    page: Mapped[int]
    section: Mapped[str]
    text: Mapped[str] = mapped_column(Text)
