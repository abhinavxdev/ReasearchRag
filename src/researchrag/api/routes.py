"""Thin HTTP adapters: all document and answer behavior lives in services."""

import json
from collections.abc import Iterator

from fastapi import APIRouter, BackgroundTasks, Depends, Response, UploadFile
from fastapi.responses import StreamingResponse

from researchrag.api.dependencies import Services, get_services
from researchrag.api.schemas import ArxivRequest, QueryRequest
from researchrag.core.exceptions import ResearchError
from researchrag.core.models import Answer, Paper

router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness only; intentionally does not download or load any models."""
    return {"status": "ok", "version": "0.1.0"}


@router.get("/papers", response_model=list[Paper])
def papers(services: Services = Depends(get_services)) -> list[Paper]:
    """List papers and ingestion status."""
    return services.library.list()


@router.get("/papers/{paper_id}", response_model=Paper)
def paper(paper_id: str, services: Services = Depends(get_services)) -> Paper:
    """Inspect one ingestion job."""
    return services.library.get(paper_id)


@router.post("/papers", response_model=Paper, status_code=202)
def upload(
    file: UploadFile, tasks: BackgroundTasks, services: Services = Depends(get_services)
) -> Paper:
    """Validate an upload, deduplicate, and schedule background parsing."""
    result = services.library.stage(file.file, file.filename or "upload.pdf", file.content_type)
    tasks.add_task(services.library.ingest, result.id)
    return result


@router.post("/papers/arxiv", response_model=Paper, status_code=202)
def import_arxiv(
    body: ArxivRequest, tasks: BackgroundTasks, services: Services = Depends(get_services)
) -> Paper:
    """Download from arXiv then schedule ingestion."""
    result = services.library.import_arxiv(body.arxiv)
    tasks.add_task(services.library.ingest, result.id)
    return result


@router.post("/papers/{paper_id}/reingest", response_model=Paper, status_code=202)
def reingest(
    paper_id: str, tasks: BackgroundTasks, services: Services = Depends(get_services)
) -> Paper:
    """Rebuild source chunks and vectors from the original upload."""
    result = services.library.reingest(paper_id)
    tasks.add_task(services.library.ingest, paper_id)
    return result


@router.delete("/papers/{paper_id}", status_code=204)
def delete_paper(paper_id: str, services: Services = Depends(get_services)) -> Response:
    """Delete a completed or failed paper."""
    services.library.delete(paper_id)
    return Response(status_code=204)


@router.post("/query", response_model=Answer)
def query(body: QueryRequest, services: Services = Depends(get_services)) -> Answer:
    """Return only validated citations and human-readable grounded text."""
    return services.answers.answer(body.question, body.paper_ids, body.top_k)


@router.post("/query/stream")
def stream(body: QueryRequest, services: Services = Depends(get_services)) -> StreamingResponse:
    """SSE protocol: status, token, done; error events after headers are sent."""

    def events() -> Iterator[str]:
        try:
            for event in services.answers.events(body.question, body.paper_ids, body.top_k):
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        except ResearchError as exc:
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"
        except Exception:
            yield 'event: error\ndata: {"detail": "Unexpected server error; check server logs"}\n\n'

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
