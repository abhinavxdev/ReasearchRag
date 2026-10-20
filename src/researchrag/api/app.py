"""FastAPI application factory with injectable dependencies."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from researchrag.api.dependencies import Services, build_services
from researchrag.api.routes import router
from researchrag.config import Settings, get_settings
from researchrag.core.exceptions import ResearchError
from researchrag.core.logging import configure_logging


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    """Create an isolated app; startup does not require LLM credentials or model files."""
    config = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        app.state.services = services or build_services(config)
        app.state.services.library.recover()
        yield

    app = FastAPI(title="ResearchRAG", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ResearchError)
    async def domain_error(_: Request, exc: ResearchError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    app.include_router(router)
    return app


app = create_app()
