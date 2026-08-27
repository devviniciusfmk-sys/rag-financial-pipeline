"""Geracao local de embeddings com sentence-transformers (all-MiniLM-L6-v2)."""

from __future__ import annotations

import threading
from typing import Iterable, Sequence

import numpy as np
from tqdm import tqdm

from config import settings, setup_logging

logger = setup_logging()


class EmbeddingGenerator:
    """Carrega o modelo uma unica vez e vetoriza textos em batches.

    O modelo e carregado de forma preguicosa (lazy) para nao penalizar o import
    do modulo -- util na API, onde o warm-up acontece no lifespan.
    """

    _lock = threading.Lock()

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embedding_batch_size
        self.device = device or settings.embedding_device
        self.normalize = normalize
        self._model = None
        self._dimension: int | None = None

    # ------------------------------------------------------------------ #
    # Ciclo de vida do modelo
    # ------------------------------------------------------------------ #
    @property
    def model(self):
        """Instancia de SentenceTransformer (thread-safe, carregada uma vez)."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._load_model()
        return self._model

    def _load_model(self):
        from sentence_transformers import SentenceTransformer  # import tardio

        logger.info(
            "carregando modelo de embeddings '%s' (device=%s)",
            self.model_name,
            self.device or "auto",
        )
        model = SentenceTransformer(self.model_name, device=self.device)
        # o nome do metodo mudou nas versoes recentes do sentence-transformers
        dim_fn = getattr(model, "get_sentence_embedding_dimension", None) or getattr(
            model, "get_embedding_dimension"
        )
        self._dimension = int(dim_fn())
        logger.info(
            "modelo pronto: %s (dim=%d)", self.model_name, self._dimension
        )
        if self._dimension != settings.embedding_dim:
            logger.warning(
                "dimensao do modelo (%d) difere de EMBEDDING_DIM (%d): ajuste a "
                "coluna vector() do banco antes de indexar",
                self._dimension,
                settings.embedding_dim,
            )
        return model

    def warm_up(self) -> None:
        """Forca o download/carregamento do modelo antecipadamente."""
        self.generate(["warm up"], show_progress=False)

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            _ = self.model
        return int(self._dimension or settings.embedding_dim)

    # ------------------------------------------------------------------ #
    # Geracao
    # ------------------------------------------------------------------ #
    def generate(
        self,
        texts: Sequence[str] | Iterable[str],
        show_progress: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Vetoriza uma lista de textos.

        Retorna um `np.ndarray` float32 de shape (len(texts), dimension).
        Use `.tolist()` para obter `list[list[float]]`.
        """
        items = [self._sanitize(t) for t in texts]
        if not items:
            return np.empty((0, self.dimension), dtype=np.float32)

        size = batch_size or self.batch_size
        vectors: list[np.ndarray] = []
        bar = tqdm(
            total=len(items),
            desc="embeddings",
            unit="doc",
            disable=not show_progress or len(items) <= size,
        )
        try:
            for start in range(0, len(items), size):
                batch = items[start : start + size]
                encoded = self.model.encode(
                    batch,
                    batch_size=size,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                )
                vectors.append(np.asarray(encoded, dtype=np.float32))
                bar.update(len(batch))
        finally:
            bar.close()

        return np.vstack(vectors).astype(np.float32, copy=False)

    def generate_one(self, text: str) -> np.ndarray:
        """Vetoriza um unico texto e devolve um vetor 1-D."""
        return self.generate([text], show_progress=False)[0]

    def generate_list(
        self, texts: Sequence[str], show_progress: bool = True
    ) -> list[list[float]]:
        """Mesma geracao, porem serializavel em JSON (`list[list[float]]`)."""
        return self.generate(texts, show_progress=show_progress).tolist()

    @staticmethod
    def _sanitize(text: object) -> str:
        if text is None:
            return " "
        value = str(text).replace("\x00", " ").strip()
        return value or " "

    def __repr__(self) -> str:  # pragma: no cover
        loaded = self._model is not None
        return (
            f"EmbeddingGenerator(model={self.model_name!r}, "
            f"batch_size={self.batch_size}, loaded={loaded})"
        )


_default_generator: EmbeddingGenerator | None = None
_default_lock = threading.Lock()


def get_embedding_generator() -> EmbeddingGenerator:
    """Singleton usado pela API e pelos scripts de ingestao."""
    global _default_generator
    if _default_generator is None:
        with _default_lock:
            if _default_generator is None:
                _default_generator = EmbeddingGenerator()
    return _default_generator


if __name__ == "__main__":  # pragma: no cover
    gen = EmbeddingGenerator()
    sample = [
        "Empresa: Banco do Brasil S.A. CNPJ: 00000000000191. UF: DF",
        "Empresa: Nu Pagamentos S.A. CNPJ: 18236120000158. UF: SP",
    ]
    vecs = gen.generate(sample)
    logger.info("shape=%s dtype=%s dim=%d", vecs.shape, vecs.dtype, gen.dimension)
    logger.info("similaridade=%.4f", float(vecs[0] @ vecs[1]))
