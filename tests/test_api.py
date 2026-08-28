"""Testes dos endpoints FastAPI usando fakes de store, generator e chain."""

from __future__ import annotations

from typing import Any, Iterator

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_ROW, FakeGenerator, FakeStore

MODELO_PRIMARIO = "nvidia/nemotron-3-ultra:free"


class FakeChain:
    """Substituto de `RAGChain` que nao chama o OpenRouter."""

    model = MODELO_PRIMARIO

    def __init__(self, api_key_configured: bool = True) -> None:
        self.searcher: Any = None
        self.api_key_configured = api_key_configured
        self.perguntas: list[str] = []

    def health(self) -> dict[str, Any]:
        return {
            "base_url": "https://openrouter.ai/api/v1",
            "model": self.model,
            "fallback_model": "meta-llama/llama-3.3-70b:free",
            "api_key_configured": self.api_key_configured,
        }

    def ask(self, question: str, top_k: int = 5, uf=None, porte=None) -> dict[str, Any]:
        self.perguntas.append(question)
        documentos = self.searcher.search(question, top_k=top_k, uf=uf, porte=porte)
        return {
            "answer": "Resposta de teste.",
            "model_used": self.model,
            "fallback_used": False,
            "elapsed_ms": 42,
            "sources": [
                {
                    "cnpj": doc["cnpj"],
                    "cnpj_formatado": doc["cnpj_formatado"],
                    "razao_social": doc["razao_social"],
                    "score": doc["score"],
                    "uf": doc["uf"],
                    "source": doc["source"],
                    "text_chunk": doc["text_chunk"],
                }
                for doc in documentos
            ],
        }


@pytest.fixture
def chain() -> FakeChain:
    return FakeChain()


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, fake_store: FakeStore, chain: FakeChain
) -> Iterator[TestClient]:
    """TestClient com o lifespan real, mas infraestrutura falsa."""
    import api.main as main

    monkeypatch.setattr(main, "get_vector_store", lambda: fake_store)
    monkeypatch.setattr(main, "get_embedding_generator", lambda: FakeGenerator())
    monkeypatch.setattr(main, "get_rag_chain", lambda: chain)

    with TestClient(main.app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# /health e metadados
# --------------------------------------------------------------------------- #
def test_health_ok(client: TestClient) -> None:
    resposta = client.get("/health")
    corpo = resposta.json()

    assert resposta.status_code == 200
    assert corpo["status"] == "ok"
    assert corpo["database"]["connected"] is True
    assert corpo["database"]["pgvector"] == "0.7.4"
    assert corpo["embeddings"]["dimension"] == 384
    assert corpo["llm"]["api_key_configured"] is True


def test_health_degradado_quando_banco_cai(
    monkeypatch: pytest.MonkeyPatch, client: TestClient, fake_store: FakeStore
) -> None:
    monkeypatch.setattr(
        fake_store,
        "health",
        lambda: {"connected": False, "table_exists": False, "error": "conexao recusada"},
    )
    resposta = client.get("/health")

    assert resposta.status_code == 503
    assert resposta.json()["status"] == "degraded"


def test_lifespan_prepara_schema_e_modelo(client: TestClient, fake_store: FakeStore) -> None:
    estado = client.app.state
    assert fake_store.schema_calls == 1
    assert estado.db_ready is True
    assert estado.model_ready is True


def test_root_lista_endpoints(client: TestClient) -> None:
    corpo = client.get("/").json()
    assert set(corpo["endpoints"]) >= {"/ask", "/search", "/companies", "/stats"}


def test_docs_e_openapi_expostos(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    caminhos = client.get("/openapi.json").json()["paths"]
    assert {"/ask", "/search", "/companies", "/stats", "/health"} <= set(caminhos)


def test_cors_e_header_de_tempo(client: TestClient) -> None:
    resposta = client.get("/stats", headers={"Origin": "https://exemplo.com.br"})
    assert resposta.headers["access-control-allow-origin"] == "*"
    assert resposta.headers["X-Process-Time"].endswith("ms")


# --------------------------------------------------------------------------- #
# /search
# --------------------------------------------------------------------------- #
def test_search_retorna_resultados_ordenados(client: TestClient) -> None:
    resposta = client.get("/search", params={"q": "nubank", "limit": 5})
    corpo = resposta.json()

    assert resposta.status_code == 200
    assert corpo[0]["cnpj"] == "18236120000158"
    assert corpo[0]["razao_social"] == "Nu Pagamentos S.A."
    assert corpo[0]["score"] == pytest.approx(0.9231)
    assert corpo[0]["uf"] == "SP"


def test_search_respeita_limit(client: TestClient, fake_store: FakeStore) -> None:
    fake_store.rows = [SAMPLE_ROW] * 5
    assert len(client.get("/search", params={"q": "banco", "limit": 2}).json()) == 2


def test_search_valida_parametros(client: TestClient) -> None:
    assert client.get("/search", params={"q": "x"}).status_code == 422
    assert client.get("/search").status_code == 422
    assert client.get("/search", params={"q": "banco", "limit": 999}).status_code == 422


# --------------------------------------------------------------------------- #
# /companies
# --------------------------------------------------------------------------- #
def test_companies_com_filtros(client: TestClient) -> None:
    resposta = client.get("/companies", params={"uf": "SP", "porte": "05", "limit": 20})
    corpo = resposta.json()

    assert resposta.status_code == 200
    assert corpo[0]["cnpj"] == "18236120000158"
    assert corpo[0]["cnpj_formatado"] == "18.236.120/0001-58"
    assert corpo[0]["porte_cod"] == "05"
    assert corpo[0]["source"] == "bacen"


def test_companies_repassa_filtros_ao_store(
    client: TestClient, fake_store: FakeStore
) -> None:
    recebidos: dict[str, Any] = {}
    fake_store.list_companies = lambda **kwargs: recebidos.update(kwargs) or []

    client.get("/companies", params={"uf": "rs", "porte": "01", "limit": 7, "offset": 3})

    assert recebidos["uf"] == "RS"  # normalizado para maiusculas
    assert recebidos["porte"] == "01"
    assert recebidos["limit"] == 7
    assert recebidos["offset"] == 3


def test_companies_erro_no_banco_vira_500(
    client: TestClient, fake_store: FakeStore
) -> None:
    def explode(**kwargs: Any):
        raise RuntimeError("conexao perdida")

    fake_store.list_companies = explode
    assert client.get("/companies").status_code == 500


# --------------------------------------------------------------------------- #
# /stats
# --------------------------------------------------------------------------- #
def test_stats_retorna_totais_e_fontes(client: TestClient) -> None:
    corpo = client.get("/stats").json()

    assert corpo["total_companies"] == 1234
    assert corpo["total_embeddings"] == 1234
    assert corpo["sources"] == ["bacen", "cvm", "receita_federal"]
    assert corpo["by_source"]["receita_federal"] == 1000
    assert corpo["embedding_dim"] == 384


def test_stats_banco_indisponivel_vira_503(
    client: TestClient, fake_store: FakeStore
) -> None:
    def explode():
        raise RuntimeError("banco fora do ar")

    fake_store.stats = explode
    assert client.get("/stats").status_code == 503


# --------------------------------------------------------------------------- #
# /ask
# --------------------------------------------------------------------------- #
def test_ask_retorna_answer_sources_e_model(client: TestClient, chain: FakeChain) -> None:
    resposta = client.post(
        "/ask", json={"question": "Quais bancos participam do Open Finance?"}
    )
    corpo = resposta.json()

    assert resposta.status_code == 200
    assert corpo["answer"] == "Resposta de teste."
    assert corpo["model"] == MODELO_PRIMARIO
    assert corpo["fallback_used"] is False
    assert corpo["sources"][0]["razao_social"] == "Nu Pagamentos S.A."
    assert chain.perguntas == ["Quais bancos participam do Open Finance?"]


def test_ask_repassa_top_k(client: TestClient, fake_store: FakeStore) -> None:
    fake_store.rows = [SAMPLE_ROW] * 10
    corpo = client.post("/ask", json={"question": "bancos digitais", "top_k": 3}).json()
    assert len(corpo["sources"]) == 3


def test_ask_valida_pergunta(client: TestClient) -> None:
    assert client.post("/ask", json={"question": "ab"}).status_code == 422
    assert client.post("/ask", json={"question": "   "}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": "ok?", "top_k": 99}).status_code == 422


def test_ask_falha_do_llm_vira_502(client: TestClient, chain: FakeChain) -> None:
    def explode(question: str, top_k: int = 5, uf=None, porte=None):
        raise RuntimeError("todos os modelos falharam")

    chain.ask = explode
    resposta = client.post("/ask", json={"question": "Quais bancos existem?"})

    assert resposta.status_code == 502
    assert "todos os modelos falharam" in resposta.json()["detail"]


def test_search_nao_corta_candidatos_por_padrao(client: TestClient) -> None:
    """min_score default -1.0: consulta ruim devolve os melhores, nao vazio."""
    resposta = client.get("/search", params={"q": "termo improvavel xyz", "limit": 3})
    assert resposta.status_code == 200
    assert len(resposta.json()) > 0


def test_search_expoe_componentes_do_score_hibrido(client: TestClient) -> None:
    item = client.get("/search", params={"q": "nubank"}).json()[0]
    assert "vector_score" in item and "name_score" in item
