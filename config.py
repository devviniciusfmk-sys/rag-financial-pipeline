"""Configuracao central do pipeline, carregada de variaveis de ambiente / .env."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.logging import RichHandler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for _d in (DATA_DIR, RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Le o .env da raiz do projeto. Nomes de env var sao case-insensitive."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:54322/postgres",
        alias="DATABASE_URL",
    )
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    llm_model: str = Field(default="nvidia/nemotron-3-ultra:free", alias="LLM_MODEL")
    fallback_model: str = Field(
        default="meta-llama/llama-3.3-70b:free", alias="FALLBACK_MODEL"
    )
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    debug: bool = Field(default=True, alias="DEBUG")

    # Ajustes finos (nao obrigatorios no .env)
    embedding_dim: int = Field(default=384, alias="EMBEDDING_DIM")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    embedding_device: str | None = Field(default=None, alias="EMBEDDING_DEVICE")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=1024, alias="LLM_MAX_TOKENS")
    llm_timeout: float = Field(default=90.0, alias="LLM_TIMEOUT")
    db_pool_min: int = Field(default=1, alias="DB_POOL_MIN")
    db_pool_max: int = Field(default=8, alias="DB_POOL_MAX")
    demo_row_limit: int = Field(default=20_000, alias="DEMO_ROW_LIMIT")
    app_title: str = Field(default="RAG Financeiro BR", alias="APP_TITLE")
    app_referer: str = Field(
        default="https://github.com/rag-financial-pipeline", alias="APP_REFERER"
    )

    @field_validator("database_url")
    @classmethod
    def _normalize_dsn(cls, value: str) -> str:
        # psycopg2 nao entende os esquemas usados por SQLAlchemy/LangChain.
        for prefix in ("postgresql+psycopg2://", "postgresql+psycopg://", "postgres://"):
            if value.startswith(prefix):
                return "postgresql://" + value[len(prefix) :]
        return value

    @property
    def raw_dir(self) -> Path:
        return RAW_DIR

    @property
    def processed_dir(self) -> Path:
        return PROCESSED_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

_LOGGING_READY = False


def setup_logging(level: int | str | None = None) -> logging.Logger:
    """Configura logging estruturado com rich (idempotente)."""
    global _LOGGING_READY

    resolved = level or (logging.DEBUG if settings.debug else logging.INFO)
    if not _LOGGING_READY:
        logging.basicConfig(
            level=resolved,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(
                    rich_tracebacks=True,
                    markup=False,
                    show_path=settings.debug,
                    omit_repeated_times=False,
                )
            ],
        )
        # Bibliotecas verbosas demais em DEBUG
        for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "openai"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _LOGGING_READY = True
    else:
        logging.getLogger().setLevel(resolved)

    return logging.getLogger("rag")


__all__ = [
    "BASE_DIR",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "Settings",
    "get_settings",
    "settings",
    "setup_logging",
]
