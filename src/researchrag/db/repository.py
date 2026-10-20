"""Short-lived database sessions; only completed papers are searchable."""

from pathlib import Path

from sqlalchemy import create_engine, delete, event, select, update
from sqlalchemy.orm import sessionmaker

from researchrag.core.exceptions import NotFound
from researchrag.core.models import Chunk, Paper
from researchrag.db.models import Base, ChunkRow, PaperRow


class Repository:
    """Thread-safe session factory around a local SQLite database."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})

        @event.listens_for(self.engine, "connect")
        def pragmas(connection: object, _: object) -> None:
            # DBAPI connection is dynamically supplied by SQLAlchemy.
            connection.execute("PRAGMA journal_mode=WAL")  # type: ignore[attr-defined]
            connection.execute("PRAGMA busy_timeout=10000")  # type: ignore[attr-defined]
            connection.execute("PRAGMA foreign_keys=ON")  # type: ignore[attr-defined]

        Base.metadata.create_all(self.engine)
        self.sessions = sessionmaker(self.engine)

    @staticmethod
    def _paper(row: PaperRow) -> Paper:
        return Paper.model_validate({key: getattr(row, key) for key in Paper.model_fields})

    def add(self, paper: Paper) -> None:
        """Insert a new pending paper."""
        with self.sessions.begin() as session:
            session.add(PaperRow(**paper.model_dump()))

    def get(self, paper_id: str) -> Paper:
        """Load a paper or raise a domain 404."""
        with self.sessions() as session:
            row = session.get(PaperRow, paper_id)
            if row is None:
                raise NotFound("Paper not found")
            return self._paper(row)

    def by_hash(self, digest: str) -> Paper | None:
        """Return an existing byte-identical document."""
        with self.sessions() as session:
            row = session.scalar(select(PaperRow).where(PaperRow.sha256 == digest))
            return self._paper(row) if row else None

    def papers(self) -> list[Paper]:
        """List the full local library."""
        with self.sessions() as session:
            return [self._paper(row) for row in session.scalars(select(PaperRow))]

    def status(self, paper_id: str, status: str, error: str | None = None) -> None:
        """Transition ingestion state."""
        with self.sessions.begin() as session:
            session.execute(
                update(PaperRow).where(PaperRow.id == paper_id).values(status=status, error=error)
            )

    def complete(self, paper_id: str, title: str, chunks: list[Chunk]) -> None:
        """Atomically publish chunks and completed metadata after vector insertion."""
        with self.sessions.begin() as session:
            session.execute(delete(ChunkRow).where(ChunkRow.paper_id == paper_id))
            session.add_all(ChunkRow(**chunk.model_dump()) for chunk in chunks)
            session.execute(
                update(PaperRow)
                .where(PaperRow.id == paper_id)
                .values(title=title, status="done", error=None, chunk_count=len(chunks))
            )

    def chunks(self, paper_ids: list[str] | None = None) -> list[Chunk]:
        """Read searchable chunks, optionally restricted to selected papers."""
        query = select(ChunkRow).join(PaperRow).where(PaperRow.status == "done")
        if paper_ids is not None:
            query = query.where(ChunkRow.paper_id.in_(paper_ids))
        with self.sessions() as session:
            return [
                Chunk.model_validate({key: getattr(row, key) for key in Chunk.model_fields})
                for row in session.scalars(query)
            ]

    def delete(self, paper_id: str) -> None:
        """Remove source records; caller removes vector and file artifacts first."""
        with self.sessions.begin() as session:
            session.execute(delete(ChunkRow).where(ChunkRow.paper_id == paper_id))
            session.execute(delete(PaperRow).where(PaperRow.id == paper_id))
