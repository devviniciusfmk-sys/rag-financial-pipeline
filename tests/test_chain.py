"""Testes de `pipeline.chain`: fallback de modelo, ausencia de contexto e erros."""

from __future__ import annotations

from typing import Any

import httpx
import openai
import pytest

from pipeline.chain import NO_CONTEXT_ANSWER, RAGChain

DOCUMENTO: dict[str, Any] = {
    "cnpj": "18236120000158",
    "cnpj_formatado": "18.236.120/0001-58",
    "razao_social": "Nu Pagamentos S.A.",
    "score": 0.93,
    "uf": "SP",
    "source": "bacen",
    "text_chunk": "Empresa: Nu Pagamentos S.A. CNPJ: 18236120000158. UF: SP",
    "metadata": {"source": "bacen"},
}


class FakeSearcher:
    """Substituto de `SemanticSearch` com resultados fixos."""

    def __init__(self, documentos: list[dict[str, Any]]) -> None:
        self.documentos = documentos
        self.chamadas: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5, uf=None, porte=None, **kwargs: Any):
        self.chamadas.append((query, top_k))
        return self.documentos[:top_k]


def make_chain(documentos: list[dict[str, Any]] | None = None) -> RAGChain:
    docs = [DOCUMENTO] if documentos is None else documentos
    return RAGChain(searcher=FakeSearcher(docs), api_key="sk-test")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def test_build_context_numera_documentos() -> None:
    contexto = RAGChain.build_context([DOCUMENTO, DOCUMENTO])
    assert "[1]" in contexto and "[2]" in contexto
    assert "fonte: bacen" in contexto
    assert "similaridade: 0.930" in contexto


def test_build_messages_tem_system_e_user() -> None:
    chain = make_chain()
    mensagens = chain.build_messages("Quais bancos?", [DOCUMENTO])

    assert [m["role"] for m in mensagens] == ["system", "user"]
    assert "portugues do Brasil" in mensagens[0]["content"]
    assert "Quais bancos?" in mensagens[1]["content"]
    assert "18236120000158" in mensagens[1]["content"]


# --------------------------------------------------------------------------- #
# Fallback automatico
# --------------------------------------------------------------------------- #
def test_usa_modelo_primario_quando_funciona() -> None:
    chain = make_chain()
    tentativas: list[str] = []

    def fake_call(model: str, messages: list[dict[str, str]]) -> str:
        tentativas.append(model)
        return "Resposta do primario."

    chain._call_model = fake_call
    resultado = chain.ask("Quais bancos participam do Open Finance?")

    assert resultado["answer"] == "Resposta do primario."
    assert resultado["model_used"] == chain.model
    assert resultado["fallback_used"] is False
    assert resultado["errors"] == []
    assert tentativas == [chain.model]


def test_cai_para_o_fallback_quando_primario_falha() -> None:
    chain = make_chain()
    tentativas: list[str] = []

    def fake_call(model: str, messages: list[dict[str, str]]) -> str:
        tentativas.append(model)
        if model == chain.model:
            raise openai.APIConnectionError(request=None)
        return "Resposta do secundario."

    chain._call_model = fake_call
    resultado = chain.ask("Quais bancos participam do Open Finance?")

    assert resultado["answer"] == "Resposta do secundario."
    assert resultado["model_used"] == chain.fallback_model
    assert resultado["fallback_used"] is True
    assert tentativas == [chain.model, chain.fallback_model]
    assert "APIConnectionError" in resultado["errors"][0]


def test_nao_repete_modelo_quando_primario_e_igual_ao_fallback() -> None:
    chain = RAGChain(
        searcher=FakeSearcher([DOCUMENTO]),
        model="modelo-unico",
        fallback_model="modelo-unico",
        api_key="sk-test",
    )
    tentativas: list[str] = []

    def fake_call(model: str, messages: list[dict[str, str]]) -> str:
        tentativas.append(model)
        raise ValueError("indisponivel")

    chain._call_model = fake_call
    with pytest.raises(RuntimeError):
        chain.ask("pergunta")

    assert tentativas == ["modelo-unico"]


def test_chave_invalida_nao_tenta_o_fallback() -> None:
    chain = make_chain()
    tentativas: list[str] = []

    def fake_call(model: str, messages: list[dict[str, str]]) -> str:
        tentativas.append(model)
        resposta = httpx.Response(
            401, request=httpx.Request("POST", "https://openrouter.ai/api/v1/x")
        )
        raise openai.AuthenticationError("chave invalida", response=resposta, body=None)

    chain._call_model = fake_call
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        chain.ask("pergunta")

    assert tentativas == [chain.model]  # nao adianta trocar de modelo


def test_erro_quando_todos_os_modelos_falham() -> None:
    chain = make_chain()
    chain._call_model = lambda model, messages: (_ for _ in ()).throw(
        ValueError("boom")
    )

    with pytest.raises(RuntimeError, match="todos os modelos falharam"):
        chain.ask("pergunta")


# --------------------------------------------------------------------------- #
# Sem contexto e validacoes
# --------------------------------------------------------------------------- #
def test_sem_contexto_nao_chama_o_llm() -> None:
    chain = make_chain(documentos=[])
    chamou = False

    def fake_call(model: str, messages: list[dict[str, str]]) -> str:
        nonlocal chamou
        chamou = True
        return "nao deveria acontecer"

    chain._call_model = fake_call
    resultado = chain.ask("pergunta sobre base vazia")

    assert chamou is False
    assert resultado["answer"] == NO_CONTEXT_ANSWER
    assert resultado["sources"] == []
    assert resultado["model_used"] == "none"


def test_pergunta_vazia_levanta_value_error() -> None:
    chain = make_chain()
    with pytest.raises(ValueError):
        chain.ask("   ")


def test_ask_monta_sources_e_metricas() -> None:
    chain = make_chain()
    chain._call_model = lambda model, messages: "ok"
    resultado = chain.ask("Quais bancos?", top_k=3)

    assert resultado["question"] == "Quais bancos?"
    assert resultado["elapsed_ms"] >= 0
    assert chain.searcher.chamadas == [("Quais bancos?", 3)]
    fonte = resultado["sources"][0]
    assert fonte["cnpj"] == "18236120000158"
    assert fonte["cnpj_formatado"] == "18.236.120/0001-58"
    assert fonte["score"] == pytest.approx(0.93)


def test_health_reporta_configuracao() -> None:
    chain = make_chain()
    saude = chain.health()

    assert saude["model"] == chain.model
    assert saude["fallback_model"] == chain.fallback_model
    assert saude["api_key_configured"] is True
    assert saude["base_url"].startswith("https://")


def test_health_detecta_chave_placeholder() -> None:
    chain = RAGChain(searcher=FakeSearcher([]), api_key="your-key-here")
    assert chain.health()["api_key_configured"] is False
