"""Busca semantica: query -> embedding -> pgvector -> resultados enriquecidos."""

from __future__ import annotations

import re
from typing import Any

from config import setup_logging
from embeddings.generator import EmbeddingGenerator, get_embedding_generator
from embeddings.store import VectorStore, get_vector_store

logger = setup_logging()

CNPJ_PATTERN = re.compile(r"\d{2}[.\s]?\d{3}[.\s]?\d{3}[/\s]?\d{4}[-\s]?\d{2}|\b\d{14}\b")


class SemanticSearch:
    """Orquestra `EmbeddingGenerator` + `VectorStore` para busca por similaridade."""

    def __init__(
        self,
        generator: EmbeddingGenerator | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.generator = generator or get_embedding_generator()
        self.store = store or get_vector_store()

    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int = 5,
        uf: str | None = None,
        porte: str | None = None,
        source: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Retorna os `top_k` chunks mais similares a `query`.

        Cada item: cnpj, razao_social, score, text_chunk, uf, porte, situacao,
        cnae, capital_social, source e metadata completa.
        """
        cleaned = (query or "").strip()
        if not cleaned:
            return []

        # Atalho: se a query e um CNPJ, resolve por igualdade exata.
        exact = self._search_by_cnpj(cleaned, top_k)
        if exact:
            logger.debug("busca exata por CNPJ retornou %d resultado(s)", len(exact))
            return exact

        vector = self.generator.generate_one(cleaned)
        rows = self.store.search(
            vector,
            top_k=top_k,
            uf=uf,
            porte=porte,
            source=source,
            min_score=min_score,
        )
        results = [self._format(row) for row in rows]
        logger.debug("busca '%s' -> %d resultado(s)", cleaned[:60], len(results))
        return results

    def search_texts(self, query: str, top_k: int = 5) -> list[str]:
        """Somente os textos, no formato consumido pelo prompt do RAG."""
        return [item["text_chunk"] for item in self.search(query, top_k=top_k)]

    # ------------------------------------------------------------------ #
    def _search_by_cnpj(self, query: str, top_k: int) -> list[dict[str, Any]]:
        match = CNPJ_PATTERN.search(query)
        if not match:
            return []
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) != 14:
            return []
        rows = self.store.get_by_cnpj(digits)
        formatted = []
        for row in rows[:top_k]:
            item = self._format({**row, "score": 1.0, "id": row.get("id", 0)})
            item["match_type"] = "cnpj_exato"
            formatted.append(item)
        return formatted

    @staticmethod
    def _format(row: dict[str, Any]) -> dict[str, Any]:
        metadata = row.get("metadata") or {}
        return {
            "cnpj": row.get("cnpj", ""),
            "cnpj_formatado": metadata.get("cnpj_formatado", ""),
            "razao_social": metadata.get("razao_social", ""),
            "score": float(row.get("score") or 0.0),
            "uf": metadata.get("uf", ""),
            "porte": metadata.get("porte", ""),
            "situacao": metadata.get("situacao", ""),
            "cnae": metadata.get("cnae", ""),
            "capital_social": metadata.get("capital_social", 0.0),
            "source": metadata.get("source", ""),
            "text_chunk": row.get("text_chunk", ""),
            "metadata": metadata,
            "match_type": "semantico",
        }


_default_search: SemanticSearch | None = None


def get_semantic_search() -> SemanticSearch:
    """Singleton usado pela API."""
    global _default_search
    if _default_search is None:
        _default_search = SemanticSearch()
    return _default_search


if __name__ == "__main__":  # pragma: no cover
    engine = SemanticSearch()
    for hit in engine.search("bancos digitais em Sao Paulo", top_k=5):
        logger.info(
            "%.4f  %-14s %s", hit["score"], hit["cnpj"], hit["razao_social"]
        )
