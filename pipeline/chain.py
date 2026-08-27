"""Cadeia RAG: recupera contexto no pgvector e responde via OpenRouter."""

from __future__ import annotations

import logging
import time
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings, setup_logging
from retrieval.search import SemanticSearch, get_semantic_search

logger = setup_logging()

# Erros transitorios: vale a pena repetir no MESMO modelo.
TRANSIENT_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

SYSTEM_PROMPT = """Voce e um analista de dados financeiros brasileiro.
Responda SEMPRE em portugues do Brasil, de forma objetiva e tecnica.

Regras obrigatorias:
1. Use exclusivamente o CONTEXTO fornecido. Nao invente CNPJs, valores ou nomes.
2. Se o contexto nao contiver a resposta, diga explicitamente que a base
   ingerida nao possui essa informacao e sugira qual fonte poderia te-la
   (Receita Federal, CVM ou diretorio do Open Finance).
3. Cite os CNPJs (formatados) das empresas que embasam a resposta.
4. Nao repita o contexto cru; sintetize.
5. Seja conciso: no maximo 6 frases, salvo se o usuario pedir detalhamento."""

USER_TEMPLATE = """CONTEXTO RECUPERADO ({n} registros da base):
{context}

PERGUNTA: {question}

RESPOSTA:"""

NO_CONTEXT_ANSWER = (
    "Nao encontrei nenhum registro na base vetorial para essa pergunta. "
    "Rode a ingestao (`python -m scripts.build_index`) ou reformule a consulta."
)


class RAGChain:
    """Retrieval-Augmented Generation com fallback automatico de modelo."""

    def __init__(
        self,
        searcher: SemanticSearch | None = None,
        model: str | None = None,
        fallback_model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.searcher = searcher or get_semantic_search()
        self.model = model or settings.llm_model
        self.fallback_model = fallback_model or settings.fallback_model
        self.temperature = (
            settings.llm_temperature if temperature is None else temperature
        )
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.api_key = api_key or settings.openrouter_api_key
        self.base_url = base_url or settings.openrouter_base_url

        self.client = OpenAI(
            api_key=self.api_key or "missing-key",
            base_url=self.base_url,
            timeout=settings.llm_timeout,
            max_retries=0,  # o retry fica a cargo do tenacity
            default_headers={
                # Cabecalhos de atribuicao recomendados pelo OpenRouter
                "HTTP-Referer": settings.app_referer,
                "X-Title": settings.app_title,
            },
        )

    # ------------------------------------------------------------------ #
    # Prompt
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_context(documents: list[dict[str, Any]]) -> str:
        """Formata os documentos recuperados em blocos numerados."""
        blocks: list[str] = []
        for i, doc in enumerate(documents, start=1):
            meta = doc.get("metadata") or {}
            blocks.append(
                f"[{i}] {doc.get('text_chunk', '').strip()}\n"
                f"    (fonte: {meta.get('source', 'desconhecida')}, "
                f"similaridade: {doc.get('score', 0):.3f})"
            )
        return "\n\n".join(blocks)

    def build_messages(
        self, question: str, documents: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    n=len(documents),
                    context=self.build_context(documents),
                    question=question.strip(),
                ),
            },
        ]

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #
    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=20),
        retry=retry_if_exception_type(TRANSIENT_ERRORS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _call_model(self, model: str, messages: list[dict[str, str]]) -> str:
        """Uma chamada ao OpenRouter, com retry para erros transitorios."""
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if not completion.choices:
            raise ValueError(f"modelo {model} devolveu resposta sem choices")
        content = completion.choices[0].message.content or ""
        content = content.strip()
        if not content:
            raise ValueError(f"modelo {model} devolveu conteudo vazio")
        return content

    def _generate(self, messages: list[dict[str, str]]) -> tuple[str, str, list[str]]:
        """Tenta o modelo primario e, em caso de falha, o secundario."""
        errors: list[str] = []
        tried: set[str] = set()
        for attempt_model in (self.model, self.fallback_model):
            if not attempt_model or attempt_model in tried:
                continue
            tried.add(attempt_model)
            try:
                logger.info("chamando modelo %s", attempt_model)
                return self._call_model(attempt_model, messages), attempt_model, errors
            except AuthenticationError as exc:
                # Chave invalida: trocar de modelo nao adianta.
                raise RuntimeError(
                    "OPENROUTER_API_KEY invalida ou ausente: " + str(exc)
                ) from exc
            except Exception as exc:  # noqa: BLE001 - queremos cair no fallback
                message = f"{attempt_model}: {type(exc).__name__}: {exc}"
                logger.warning("falha no modelo %s -> %s", attempt_model, exc)
                errors.append(message)
        raise RuntimeError("todos os modelos falharam: " + " | ".join(errors))

    # ------------------------------------------------------------------ #
    # API publica
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        top_k: int = 5,
        uf: str | None = None,
        porte: str | None = None,
    ) -> dict[str, Any]:
        """Responde `question` usando o contexto recuperado da base vetorial."""
        started = time.perf_counter()
        clean_question = (question or "").strip()
        if not clean_question:
            raise ValueError("question nao pode ser vazia")

        documents = self.searcher.search(
            clean_question, top_k=top_k, uf=uf, porte=porte
        )
        sources = [
            {
                "cnpj": doc["cnpj"],
                "cnpj_formatado": doc.get("cnpj_formatado", ""),
                "razao_social": doc.get("razao_social", ""),
                "score": round(float(doc.get("score", 0.0)), 4),
                "uf": doc.get("uf", ""),
                "source": doc.get("source", ""),
                "text_chunk": doc.get("text_chunk", ""),
            }
            for doc in documents
        ]

        if not documents:
            return {
                "answer": NO_CONTEXT_ANSWER,
                "sources": [],
                "model_used": "none",
                "question": clean_question,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "fallback_used": False,
                "errors": [],
            }

        messages = self.build_messages(clean_question, documents)
        answer, model_used, errors = self._generate(messages)

        return {
            "answer": answer,
            "sources": sources,
            "model_used": model_used,
            "question": clean_question,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "fallback_used": model_used != self.model,
            "errors": errors,
        }

    def health(self) -> dict[str, Any]:
        """Diagnostico leve da configuracao do LLM (nao chama a API)."""
        return {
            "base_url": self.base_url,
            "model": self.model,
            "fallback_model": self.fallback_model,
            "api_key_configured": bool(self.api_key and self.api_key != "your-key-here"),
        }


_default_chain: RAGChain | None = None


def get_rag_chain() -> RAGChain:
    """Singleton usado pela API."""
    global _default_chain
    if _default_chain is None:
        _default_chain = RAGChain()
    return _default_chain


if __name__ == "__main__":  # pragma: no cover
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Pergunta ao pipeline RAG")
    parser.add_argument("question", nargs="+", help="pergunta em linguagem natural")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    chain = RAGChain()
    result = chain.ask(" ".join(args.question), top_k=args.top_k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
