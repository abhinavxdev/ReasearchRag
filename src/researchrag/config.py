"""Validated environment configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings; all environment names use RESEARCHRAG_."""

    model_config = SettingsConfigDict(env_prefix="RESEARCHRAG_", env_file=".env", extra="ignore")
    data_dir: Path = Path("data")
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    llm_provider: Literal["anthropic", "ollama"] = "anthropic"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-5"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_context_size: int = Field(default=8192, ge=2048, le=131072)
    ollama_batch_size: int = Field(default=64, ge=1, le=512)
    context_chars: int = Field(default=14000, ge=1000, le=50000)
    rerank: bool = False
    max_upload_mb: int = Field(default=25, ge=1, le=200)
    top_k: int = Field(default=6, ge=1, le=20)
    min_dense_score: float = Field(default=0.4, ge=0, le=1)
    chunk_tokens: int = Field(default=600, ge=100, le=1000)
    overlap_tokens: int = Field(default=80, ge=0, le=99)
    max_pages: int = Field(default=500, ge=1, le=2000)


@lru_cache
def get_settings() -> Settings:
    """Return process-wide validated settings."""
    return Settings()
