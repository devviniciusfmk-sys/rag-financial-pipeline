"""Persistencia dos embeddings no PostgreSQL + pgvector."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import Json, RealDictCursor, execute_values
from psycopg2.pool import ThreadedConnectionPool

from config import settings, setup_logging

logger = setup_logging()

TABLE_NAME = "company_embeddings"

DDL_STATEMENTS = (
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS {table} (
        id          BIGSERIAL PRIMARY KEY,
        cnpj        VARCHAR(14) NOT NULL,
        text_chunk  TEXT        NOT NULL,
        embedding   vector({dim}) NOT NULL,
        metadata    JSONB       NOT NULL DEFAULT '{{}}'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS {table}_cnpj_chunk_uidx "
    "ON {table} (cnpj, md5(text_chunk))",
    "CREATE INDEX IF NOT EXISTS {table}_cnpj_idx ON {table} (cnpj)",
    "CREATE INDEX IF NOT EXISTS {table}_metadata_gin "
    "ON {table} USING gin (metadata jsonb_path_ops)",
    "CREATE INDEX IF NOT EXISTS {table}_vec_idx "
    "ON {table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
)


def to_vector_literal(vector: Sequence[float] | np.ndarray) -> str:
    """Converte um vetor em literal aceito por pgvector: '[0.1,0.2,...]'."""
    array = np.asarray(vector, dtype=np.float32).ravel()
    return "[" + ",".join(f"{float(x):.7g}" for x in array) + "]"


class VectorStore:
    """CRUD e busca vetorial sobre a tabela `company_embeddings`."""

    def __init__(
        self,
        dsn: str | None = None,
        dimension: int | None = None,
        minconn: int | None = None,
        maxconn: int | None = None,
    ) -> None:
        self.dsn = dsn or settings.database_url
        self.dimension = dimension or settings.embedding_dim
        self._minconn = minconn or settings.db_pool_min
        self._maxconn = maxconn or settings.db_pool_max
        self._pool: ThreadedConnectionPool | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Conexao
    # ------------------------------------------------------------------ #
    @property
    def pool(self) -> ThreadedConnectionPool:
        if self._pool is None:
            with self._lock:
                if self._pool is None:
                    logger.info("abrindo pool de conexoes (%s..%s)", self._minconn, self._maxconn)
                    self._pool = ThreadedConnectionPool(
                        self._minconn, self._maxconn, dsn=self.dsn
                    )
        return self._pool

    @contextmanager
    def connection(self) -> Iterator[psycopg2.extensions.connection]:
        conn = self.pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self.pool.putconn(conn)

    @contextmanager
    def cursor(self, dict_rows: bool = True) -> Iterator[psycopg2.extensions.cursor]:
        with self.connection() as conn:
            factory = RealDictCursor if dict_rows else None
            with conn.cursor(cursor_factory=factory) as cur:
                yield cur

    def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            logger.info("pool de conexoes encerrado")

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def ensure_schema(self) -> None:
        """Cria extensao, tabela e indices caso ainda nao existam."""
        with self.cursor(dict_rows=False) as cur:
            for statement in DDL_STATEMENTS:
                cur.execute(statement.format(table=TABLE_NAME, dim=self.dimension))
        logger.info("schema pronto: %s (vector(%d))", TABLE_NAME, self.dimension)

    def health(self) -> dict[str, Any]:
        """Checagem usada pelo endpoint /health."""
        info: dict[str, Any] = {"connected": False, "table_exists": False}
        try:
            with self.cursor() as cur:
                cur.execute("SELECT version() AS version")
                info["version"] = cur.fetchone()["version"].split(",")[0]
                cur.execute(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
                )
                row = cur.fetchone()
                info["pgvector"] = row["extversion"] if row else None
                cur.execute("SELECT to_regclass(%s) AS reg", (TABLE_NAME,))
                info["table_exists"] = cur.fetchone()["reg"] is not None
                info["connected"] = True
                if info["table_exists"]:
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) AS n FROM {}").format(
                            sql.Identifier(TABLE_NAME)
                        )
                    )
                    info["rows"] = int(cur.fetchone()["n"])
        except psycopg2.Error as exc:
            info["error"] = str(exc).strip()
        return info

    # ------------------------------------------------------------------ #
    # Escrita
    # ------------------------------------------------------------------ #
    def save(
        self,
        cnpj: str,
        text: str,
        embedding: Sequence[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insere (ou atualiza) um chunk. Retorna o id da linha."""
        query = f"""
            INSERT INTO {TABLE_NAME} (cnpj, text_chunk, embedding, metadata)
            VALUES (%s, %s, %s::vector, %s)
            ON CONFLICT (cnpj, md5(text_chunk)) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata  = EXCLUDED.metadata
            RETURNING id
        """
        with self.cursor() as cur:
            cur.execute(
                query,
                (
                    str(cnpj)[:14],
                    text,
                    to_vector_literal(embedding),
                    Json(metadata or {}),
                ),
            )
            return int(cur.fetchone()["id"])

    def save_many(
        self,
        records: Iterable[tuple[str, str, Sequence[float] | np.ndarray, dict[str, Any]]],
        page_size: int = 200,
    ) -> int:
        """Insercao em lote. `records` = (cnpj, text, embedding, metadata)."""
        rows = [
            (str(cnpj)[:14], text, to_vector_literal(emb), Json(meta or {}))
            for cnpj, text, emb, meta in records
        ]
        if not rows:
            return 0

        query = f"""
            INSERT INTO {TABLE_NAME} (cnpj, text_chunk, embedding, metadata)
            VALUES %s
            ON CONFLICT (cnpj, md5(text_chunk)) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    metadata  = EXCLUDED.metadata
        """
        with self.cursor(dict_rows=False) as cur:
            execute_values(
                cur,
                query,
                rows,
                template="(%s, %s, %s::vector, %s)",
                page_size=page_size,
            )
        return len(rows)

    def delete_by_source(self, source: str) -> int:
        """Remove todos os chunks de uma fonte (util para reprocessar)."""
        with self.cursor(dict_rows=False) as cur:
            cur.execute(
                f"DELETE FROM {TABLE_NAME} WHERE metadata->>'source' = %s", (source,)
            )
            return cur.rowcount

    def truncate(self) -> None:
        with self.cursor(dict_rows=False) as cur:
            cur.execute(f"TRUNCATE TABLE {TABLE_NAME} RESTART IDENTITY")
        logger.warning("tabela %s truncada", TABLE_NAME)

    # ------------------------------------------------------------------ #
    # Leitura
    # ------------------------------------------------------------------ #
    def search(
        self,
        query_embedding: Sequence[float] | np.ndarray,
        top_k: int = 5,
        uf: str | None = None,
        porte: str | None = None,
        source: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Busca por similaridade de cosseno. Score = 1 - distancia (0..1)."""
        filters: list[str] = []
        params: list[Any] = [to_vector_literal(query_embedding)]

        if uf:
            filters.append("metadata->>'uf' = %s")
            params.append(uf.upper())
        if porte:
            filters.append("(metadata->>'porte_cod' = %s OR metadata->>'porte' = %s)")
            params.extend([porte, porte])
        if source:
            filters.append("metadata->>'source' = %s")
            params.append(source)

        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        params.extend([to_vector_literal(query_embedding), int(top_k)])

        query = f"""
            SELECT id,
                   cnpj,
                   text_chunk,
                   metadata,
                   created_at,
                   1 - (embedding <=> %s::vector) AS score
            FROM {TABLE_NAME}
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        with self.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            score = float(row["score"])
            if score < min_score:
                continue
            metadata = row["metadata"] or {}
            results.append(
                {
                    "id": int(row["id"]),
                    "cnpj": row["cnpj"],
                    "text_chunk": row["text_chunk"],
                    "score": round(score, 4),
                    "metadata": metadata,
                    "created_at": row["created_at"].isoformat()
                    if row.get("created_at")
                    else None,
                }
            )
        return results

    def list_companies(
        self,
        uf: str | None = None,
        porte: str | None = None,
        source: str | None = None,
        search: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Listagem filtrada (sem busca vetorial), uma linha por CNPJ."""
        filters: list[str] = []
        params: list[Any] = []

        if uf:
            filters.append("metadata->>'uf' = %s")
            params.append(uf.upper())
        if porte:
            filters.append("(metadata->>'porte_cod' = %s OR metadata->>'porte' = %s)")
            params.extend([porte, porte])
        if source:
            filters.append("metadata->>'source' = %s")
            params.append(source)
        if search:
            filters.append("metadata->>'razao_social' ILIKE %s")
            params.append(f"%{search}%")

        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        params.extend([int(limit), int(offset)])

        query = f"""
            SELECT DISTINCT ON (cnpj)
                   cnpj,
                   text_chunk,
                   metadata,
                   created_at
            FROM {TABLE_NAME}
            {where}
            ORDER BY cnpj, created_at DESC
            LIMIT %s OFFSET %s
        """
        with self.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        return [
            {
                "cnpj": row["cnpj"],
                "text_chunk": row["text_chunk"],
                "metadata": row["metadata"] or {},
                "created_at": row["created_at"].isoformat()
                if row.get("created_at")
                else None,
            }
            for row in rows
        ]

    def get_by_cnpj(self, cnpj: str) -> list[dict[str, Any]]:
        digits = "".join(ch for ch in str(cnpj) if ch.isdigit())[:14]
        with self.cursor() as cur:
            cur.execute(
                f"SELECT cnpj, text_chunk, metadata FROM {TABLE_NAME} WHERE cnpj = %s",
                (digits,),
            )
            return [dict(row) for row in cur.fetchall()]

    def count(self) -> int:
        """Total de embeddings armazenados."""
        try:
            with self.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) AS n FROM {TABLE_NAME}")
                return int(cur.fetchone()["n"])
        except psycopg2.Error as exc:
            logger.error("count falhou: %s", exc)
            return 0

    def count_companies(self) -> int:
        """Total de CNPJs distintos."""
        with self.cursor() as cur:
            cur.execute(f"SELECT COUNT(DISTINCT cnpj) AS n FROM {TABLE_NAME}")
            return int(cur.fetchone()["n"])

    def sources(self) -> list[str]:
        """Fontes distintas presentes na base."""
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT metadata->>'source' AS source
                FROM {TABLE_NAME}
                WHERE metadata->>'source' IS NOT NULL
                ORDER BY 1
                """
            )
            return [row["source"] for row in cur.fetchall() if row["source"]]

    def stats(self) -> dict[str, Any]:
        """Agregados usados pelo endpoint /stats."""
        with self.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*)                     AS total_embeddings,
                       COUNT(DISTINCT cnpj)         AS total_companies,
                       MAX(created_at)              AS last_ingestion
                FROM {TABLE_NAME}
                """
            )
            base = cur.fetchone() or {}

            cur.execute(
                f"""
                SELECT COALESCE(metadata->>'source', 'desconhecida') AS source,
                       COUNT(*) AS total
                FROM {TABLE_NAME}
                GROUP BY 1
                ORDER BY 2 DESC
                """
            )
            by_source = {row["source"]: int(row["total"]) for row in cur.fetchall()}

            cur.execute(
                f"""
                SELECT metadata->>'uf' AS uf, COUNT(*) AS total
                FROM {TABLE_NAME}
                WHERE metadata->>'uf' IS NOT NULL
                GROUP BY 1
                ORDER BY 2 DESC
                LIMIT 10
                """
            )
            by_uf = {row["uf"]: int(row["total"]) for row in cur.fetchall()}

        last = base.get("last_ingestion")
        return {
            "total_companies": int(base.get("total_companies") or 0),
            "total_embeddings": int(base.get("total_embeddings") or 0),
            "sources": sorted(by_source),
            "by_source": by_source,
            "top_uf": by_uf,
            "last_ingestion": last.isoformat() if last else None,
            "embedding_dim": self.dimension,
        }


_default_store: VectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store() -> VectorStore:
    """Singleton usado pela API e pelos scripts."""
    global _default_store
    if _default_store is None:
        with _store_lock:
            if _default_store is None:
                _default_store = VectorStore()
    return _default_store


if __name__ == "__main__":  # pragma: no cover
    store = VectorStore()
    store.ensure_schema()
    logger.info("health: %s", store.health())
    logger.info("embeddings=%d empresas=%d", store.count(), store.count_companies())
    store.close()
