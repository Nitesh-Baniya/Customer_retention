from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LLMBackend(StrEnum):
    HUGGINGFACE = "huggingface"
    VLLM = "vllm"


class Settings(BaseSettings):
    """Typed application configuration loaded from environment variables."""

    app_name: str = "Engineering AI Assistant"
    app_version: str = "0.1.0"
    app_environment: Environment = Environment.DEVELOPMENT
    app_log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    app_api_v1_prefix: str = Field(default="/api/v1", pattern=r"^/")
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    database_url: SecretStr = SecretStr(
        "postgresql+asyncpg://postgres:postgres@localhost:5433/assistant"
    )
    database_echo: bool = False

    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    chat_rate_limit_requests: int = Field(default=10, ge=1)
    chat_rate_limit_window_seconds: int = Field(default=60, ge=1)

    google_client_id: str | None = None
    auth_cookie_name: str = "assistant_session"
    auth_session_days: int = Field(default=30, ge=1, le=365)
    auth_cookie_secure: bool = False
    auth_cookie_samesite: Literal["lax", "strict"] = "lax"

    llm_backend: LLMBackend = LLMBackend.HUGGINGFACE
    hf_token: SecretStr | None = None
    hf_base_url: AnyHttpUrl = AnyHttpUrl("https://router.huggingface.co/v1")
    hf_model: str = "openai/gpt-oss-20b:groq"
    hf_fallback_model: str = "deepseek-ai/DeepSeek-V4-Flash-0731:deepinfra"

    vllm_base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8001/v1")
    vllm_api_key: SecretStr = SecretStr("local-only")
    vllm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    llm_max_output_tokens: int = Field(default=1200, gt=0)
    llm_timeout_seconds: float = Field(default=60.0, gt=0.0)
    llm_max_tool_iterations: int = Field(default=5, ge=1, le=20)

    monid_api_key: SecretStr | None = None
    monid_base_url: AnyHttpUrl = AnyHttpUrl("https://api.monid.ai")
    monid_timeout_seconds: float = Field(default=30.0, gt=0.0)

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_dimension: int = Field(default=384, gt=0)

    qdrant_url: AnyHttpUrl = AnyHttpUrl("http://localhost:6333")
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "assistant_documents"
    qdrant_dense_model: str = "sentence-transformers/all-minilm-l6-v2"
    qdrant_sparse_model: str = "qdrant/bm25"
    qdrant_reranker_model: str = "answerdotai/answerai-colbert-small-v1"
    qdrant_reranker_dimension: int = Field(default=96, gt=0)

    rag_chunk_size: int = Field(default=800, gt=0)
    rag_chunk_overlap: int = Field(default=120, ge=0)
    rag_retrieval_top_k: int = Field(default=5, gt=0, le=50)
    rag_candidate_top_k: int = Field(default=20, gt=0, le=100)
    rag_score_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    documents_directory: Path = Path("data/documents")
    max_upload_size_mb: int = Field(default=10, gt=0)
    max_batch_upload_files: int = Field(default=10, gt=0, le=15)
    document_retention_days: int = Field(default=10, ge=1, le=365)
    document_cleanup_interval_seconds: int = Field(default=3_600, ge=0)
    document_cleanup_batch_size: int = Field(default=100, ge=1, le=1_000)

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_related_settings(self) -> Self:
        """Validate rules involving more than one environment variable."""

        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP must be smaller than RAG_CHUNK_SIZE")
        if self.rag_candidate_top_k < self.rag_retrieval_top_k:
            raise ValueError("RAG_CANDIDATE_TOP_K must be at least RAG_RETRIEVAL_TOP_K")

        return self


@lru_cache
def get_settings() -> Settings:
    """Build settings once per process and reuse the validated instance."""

    return Settings()  # type: ignore[call-arg]
