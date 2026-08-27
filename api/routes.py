"""Endpoints da API: /ask, /search, /companies e /stats."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from config import setup_logging
from embeddings.store import VectorStore, get_vector_store
from pipeline.chain import RAGChain, get_rag_chain
from retrieval.search import SemanticSearch, get_semantic_search

logger = setup_logging()

router = APIRouter(tags=["rag"])


# --------------------------------------------------------------------------- #
# Dependencias (usam os objetos criados no lifespan, com fallback ao singleton)
# --------------------------------------------------------------------------- #
def get_store(request: Request) -> VectorStore:
    return getattr(request.app.state, "store", None) or get_vector_store()


def get_search(request: Request) -> SemanticSearch:
    return getattr(request.app.state, "search", None) or get_semantic_search()


def get_chain(request: Request) -> RAGChain:
    return getattr(request.app.state, "chain", None) or get_rag_chain()


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Pergunta em linguagem natural",
        examples=["Quais bancos participam do Open Finance?"],
    )
    top_k: int = Field(5, ge=1, le=20, description="Quantos chunks recuperar")
    uf: str | None = Field(None, max_length=2, description="Filtra o contexto por UF")
    porte: str | None = Field(None, max_length=40, description="Filtra por porte")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question nao pode ser vazia")
        return cleaned

    @field_validator("uf")
    @classmethod
    def _upper_uf(cls, value: str | None) -> str | None:
        return value.upper() if value else value


class SourceItem(BaseModel):
    cnpj: str = ""
    cnpj_formatado: str = ""
    razao_social: str = ""
    score: float = 0.0
    uf: str = ""
    source: str = ""
    text_chunk: str = ""


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    model: str = Field(..., description="Modelo que efetivamente respondeu")
    fallback_used: bool = False
    elapsed_ms: int = 0


class SearchItem(BaseModel):
    cnpj: str = ""
    cnpj_formatado: str = ""
    razao_social: str = ""
    score: float = 0.0
    uf: str = ""
    porte: str = ""
    situacao: str = ""
    cnae: str = ""
    source: str = ""
    text_chunk: str = ""


class CompanyItem(BaseModel):
    cnpj: str = ""
    cnpj_formatado: str = ""
    razao_social: str = ""
    uf: str = ""
    porte: str = ""
    porte_cod: str = ""
    situacao: str = ""
    cnae: str = ""
    capital_social: float = 0.0
    source: str = ""


class StatsResponse(BaseModel):
    total_companies: int = 0
    total_embeddings: int = 0
    sources: list[str] = Field(default_factory=list)
    by_source: dict[str, int] = Field(default_factory=dict)
    top_uf: dict[str, int] = Field(default_factory=dict)
    last_ingestion: str | None = None
    embedding_dim: int = 384


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Pergunta em linguagem natural sobre a base ingerida",
)
def ask(payload: AskRequest, chain: RAGChain = Depends(get_chain)) -> AskResponse:
    try:
        result = chain.ask(
            payload.question, top_k=payload.top_k, uf=payload.uf, porte=payload.porte
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except RuntimeError as exc:
        logger.error("falha no LLM: %s", exc)
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("erro inesperado em /ask")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"erro interno: {exc}"
        ) from exc

    return AskResponse(
        answer=result["answer"],
        sources=[SourceItem(**src) for src in result.get("sources", [])],
        model=result.get("model_used", "none"),
        fallback_used=bool(result.get("fallback_used")),
        elapsed_ms=int(result.get("elapsed_ms", 0)),
    )


@router.get(
    "/search",
    response_model=list[SearchItem],
    summary="Busca semantica por similaridade de cosseno",
)
def search(
    q: str = Query(..., min_length=2, max_length=500, description="Termo de busca"),
    limit: int = Query(5, ge=1, le=50, description="Numero de resultados"),
    uf: str | None = Query(None, max_length=2),
    porte: str | None = Query(None, max_length=40),
    source: str | None = Query(None, max_length=40),
    min_score: float = Query(0.0, ge=-1.0, le=1.0),
    engine: SemanticSearch = Depends(get_search),
) -> list[SearchItem]:
    try:
        hits = engine.search(
            q,
            top_k=limit,
            uf=uf.upper() if uf else None,
            porte=porte,
            source=source,
            min_score=min_score,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("erro em /search")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"falha na busca: {exc}"
        ) from exc

    return [
        SearchItem(
            cnpj=hit.get("cnpj", ""),
            cnpj_formatado=hit.get("cnpj_formatado", ""),
            razao_social=hit.get("razao_social", ""),
            score=round(float(hit.get("score", 0.0)), 4),
            uf=hit.get("uf", ""),
            porte=hit.get("porte", ""),
            situacao=hit.get("situacao", ""),
            cnae=hit.get("cnae", ""),
            source=hit.get("source", ""),
            text_chunk=hit.get("text_chunk", ""),
        )
        for hit in hits
    ]


@router.get(
    "/companies",
    response_model=list[CompanyItem],
    summary="Lista empresas com filtros estruturados (sem busca vetorial)",
)
def companies(
    uf: str | None = Query(None, max_length=2, description="Sigla da UF, ex.: RS"),
    porte: str | None = Query(
        None, max_length=40, description="Codigo (01, 03, 05) ou descricao do porte"
    ),
    source: str | None = Query(
        None, description="receita_federal | cvm | bacen", max_length=40
    ),
    razao_social: str | None = Query(
        None, max_length=200, description="Busca parcial por razao social"
    ),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    store: VectorStore = Depends(get_store),
) -> list[CompanyItem]:
    try:
        rows = store.list_companies(
            uf=uf.upper() if uf else None,
            porte=porte,
            source=source,
            search=razao_social,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("erro em /companies")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR, f"falha na listagem: {exc}"
        ) from exc

    items: list[CompanyItem] = []
    for row in rows:
        meta: dict[str, Any] = row.get("metadata") or {}
        items.append(
            CompanyItem(
                cnpj=row.get("cnpj", ""),
                cnpj_formatado=meta.get("cnpj_formatado", ""),
                razao_social=meta.get("razao_social", ""),
                uf=meta.get("uf", ""),
                porte=meta.get("porte", ""),
                porte_cod=str(meta.get("porte_cod", "")),
                situacao=meta.get("situacao", ""),
                cnae=meta.get("cnae", ""),
                capital_social=float(meta.get("capital_social", 0.0) or 0.0),
                source=meta.get("source", ""),
            )
        )
    return items


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Estatisticas agregadas da base vetorial",
)
def stats(store: VectorStore = Depends(get_store)) -> StatsResponse:
    try:
        data = store.stats()
    except Exception as exc:  # noqa: BLE001
        logger.exception("erro em /stats")
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"banco indisponivel: {exc}"
        ) from exc
    return StatsResponse(**data)
