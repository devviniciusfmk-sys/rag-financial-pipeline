"""Aplicacao FastAPI do pipeline RAG financeiro."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from rich.console import Console
from rich.panel import Panel

from api.routes import router
from config import settings, setup_logging
from embeddings.generator import get_embedding_generator
from embeddings.store import get_vector_store
from pipeline.chain import get_rag_chain
from retrieval.search import SemanticSearch

logger = setup_logging()
console = Console()

APP_VERSION = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: prepara banco, modelo e cadeia RAG. Shutdown: fecha o pool."""
    console.print(
        Panel.fit(
            f"[bold cyan]{settings.app_title}[/bold cyan] v{APP_VERSION}\n"
            f"LLM: [green]{settings.llm_model}[/green] "
            f"(fallback: {settings.fallback_model})\n"
            f"Embeddings: [green]{settings.embedding_model}[/green] "
            f"({settings.embedding_dim}d)\n"
            f"Docs: http://{settings.api_host}:{settings.api_port}/docs",
            title="startup",
            border_style="cyan",
        )
    )

    store = get_vector_store()
    generator = get_embedding_generator()

    app.state.started_at = time.time()
    app.state.store = store
    app.state.generator = generator
    app.state.db_ready = False
    app.state.model_ready = False

    try:
        store.ensure_schema()
        app.state.db_ready = True
        logger.info("banco pronto: %d embeddings indexados", store.count())
    except Exception as exc:  # noqa: BLE001 - a API sobe mesmo sem banco
        logger.error("banco indisponivel no startup: %s", exc)

    try:
        generator.warm_up()
        app.state.model_ready = True
    except Exception as exc:  # noqa: BLE001 - warm-up e best-effort
        logger.error("falha ao carregar o modelo de embeddings: %s", exc)

    app.state.search = SemanticSearch(generator=generator, store=store)
    app.state.chain = get_rag_chain()
    app.state.chain.searcher = app.state.search

    if not app.state.chain.health()["api_key_configured"]:
        logger.warning(
            "OPENROUTER_API_KEY ausente: /ask vai falhar ate a chave ser configurada"
        )

    logger.info("API pronta em http://%s:%s", settings.api_host, settings.api_port)
    try:
        yield
    finally:
        logger.info("encerrando aplicacao")
        try:
            store.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("erro ao fechar o pool: %s", exc)


app = FastAPI(
    title=settings.app_title,
    description=(
        "RAG sobre dados publicos financeiros brasileiros "
        "(Receita Federal, CVM e Open Finance Brasil)."
    ),
    version=APP_VERSION,
    debug=settings.debug,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,  # obrigatorio ser False junto de allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log estruturado de cada requisicao + header com o tempo de resposta."""
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
    logger.info(
        "%s %s -> %s (%.1f ms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("erro nao tratado em %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "erro interno", "error": str(exc)},
    )


@app.get("/", tags=["meta"], summary="Metadados da API")
def root() -> dict[str, Any]:
    return {
        "name": settings.app_title,
        "version": APP_VERSION,
        "docs": "/docs",
        "endpoints": ["/health", "/ask", "/search", "/companies", "/stats"],
    }


@app.get("/health", tags=["meta"], summary="Healthcheck (banco + modelo + LLM)")
def health(request: Request) -> JSONResponse:
    """Verifica conexao com o Postgres/pgvector e o estado dos componentes."""
    store = getattr(request.app.state, "store", None) or get_vector_store()
    db = store.health()

    chain = getattr(request.app.state, "chain", None)
    llm = chain.health() if chain else {"api_key_configured": False}

    healthy = bool(db.get("connected") and db.get("table_exists"))
    payload: dict[str, Any] = {
        "status": "ok" if healthy else "degraded",
        "version": APP_VERSION,
        "uptime_s": round(time.time() - getattr(request.app.state, "started_at", time.time()), 1),
        "database": {
            "connected": bool(db.get("connected")),
            "pgvector": db.get("pgvector"),
            "table_exists": bool(db.get("table_exists")),
            "rows": db.get("rows", 0),
            "server": db.get("version"),
            "error": db.get("error"),
        },
        "embeddings": {
            "model": settings.embedding_model,
            "dimension": settings.embedding_dim,
            "loaded": bool(getattr(request.app.state, "model_ready", False)),
        },
        "llm": llm,
    }
    code = status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(status_code=code, content=payload)


app.include_router(router)


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_config=None,  # deixa o rich cuidar do logging
    )
