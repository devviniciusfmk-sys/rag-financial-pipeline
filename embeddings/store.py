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
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    "CREATE INDEX IF NOT EXISTS {table}_razao_trgm "
    "ON {table} USING gin ((metadata->>'razao_social') gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS {table}_vec_idx "
    "ON {table} USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
)


def to_vector_literal(vector: Sequence[float] | np.ndarray) -> str:
    """Converte um vetor em literal aceito por pgvector: '[0.1,0.2,...]'."""
    array = np.asarray(vector, dtype=np.float32).ravel()
    return "[" + ",".join(f"{float(x):.7g}" for x in array) + "]"


def build_search_query(
    table: str,
    vector_literal: str,
    query_text: str | None,
    filters: dict[str, Any],
    top_k: int,
    name_weight: float = 0.4,
    text_weight: float = 0.6,
    prefix_weight: float = 0.4,
) -> tuple[str, list[Any]]:
    """Monta o SQL da busca hibrida e a lista de parametros na ordem correta.

    Funcao pura, separada de `VectorStore.search` para poder ser testada sem
    banco: sao 12 placeholders em 4 sinais, e trocar a ordem de dois deles
    corrompe o ranking silenciosamente.
    """
    where_parts: list[str] = []
    filter_params: list[Any] = []
    if filters.get("uf"):
        where_parts.append("metadata->>'uf' = %s")
        filter_params.append(str(filters["uf"]).upper())
    if filters.get("porte"):
        where_parts.append("(metadata->>'porte_cod' = %s OR metadata->>'porte' = %s)")
        filter_params.extend([filters["porte"], filters["porte"]])
    if filters.get("source"):
        where_parts.append("metadata->>'source' = %s")
        filter_params.append(filters["source"])
    where = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

    term = query_text.strip() if query_text and query_text.strip() else None
    w_name = float(name_weight) if term else 0.0
    w_text = float(text_weight) if term else 0.0
    w_prefix = float(prefix_weight) if term else 0.0
    prefix_pattern = f"{term}%" if term else "~~sem-prefixo~~"

    query = f"""
        SELECT id,
               cnpj,
               text_chunk,
               metadata,
               created_at,
               1 - (embedding <=> %s::vector) AS vector_score,
               word_similarity(%s, COALESCE(metadata->>'razao_social', '')) AS name_score,
               word_similarity(%s, text_chunk) AS text_score,
               (COALESCE(metadata->>'razao_social', '') ILIKE %s)::int AS prefix_hit
        FROM {table}
        {where}
        ORDER BY (1 - (embedding <=> %s::vector))
                 + %s * word_similarity(%s, COALESCE(metadata->>'razao_social', ''))
                 + %s * word_similarity(%s, text_chunk)
                 + %s * (COALESCE(metadata->>'razao_social', '') ILIKE %s)::int
                 DESC
        LIMIT %s
    """
    params = (
        [vector_literal, term or "", term or "", prefix_pattern]
        + filter_params
        + [
            vector_literal,
            w_name, term or "",
            w_text, term or "",
            w_prefix, prefix_pattern,
            int(top_k),
        ]
    )
    return query, params


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

    def rebuild_vector_index(self) -> None:
        """Recria o indice ivfflat com os dados ja carregados.

        O ivfflat calcula os centroides no momento da criacao: construido sobre
        tabela vazia, ele nasce sem particionamento util. Recriar depois da
        carga e o que a documentacao do pgvector recomenda.
        """
        with self.cursor(dict_rows=False) as cur:
            cur.execute(f"DROP INDEX IF EXISTS {TABLE_NAME}_vec_idx")
            cur.execute(
                f"CREATE INDEX {TABLE_NAME}_vec_idx ON {TABLE_NAME} "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            )
            cur.execute(f"ANALYZE {TABLE_NAME}")
        logger.info("indice vetorial reconstruido sobre os dados carregados")

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
        min_score: float = -1.0,
        query_text: str | None = None,
        name_weight: float = 0.4,
        text_weight: float = 0.6,
        prefix_weight: float = 0.4,
    ) -> list[dict[str, Any]]:
        """Busca hibrida: cosseno no embedding + trigrama na razao social.

        O vetor sozinho erra nomes proprios ("Itau Unibanco" nao casava com
        nada); o trigrama sozinho nao entende semantica. O score final e
        `vetor + name_weight * similaridade_do_nome`, e ambos os componentes
        voltam no resultado para inspecao.

        Usa `word_similarity` e nao `similarity`: a segunda compara as strings
        inteiras, entao "Nu Pagamentos" contra "Nu Pagamentos S.A. - Instituicao
        De Pagamento" dava 0.412 -- empatado com "Pagamentos Limitados Ltda"
        (0.407). `word_similarity` procura a melhor subsequencia dentro do alvo
        e da 1.000 para o primeiro, 0.786 para o segundo.

        Quatro sinais somados:
          * cosseno do embedding (semantica)
          * trigrama na razao social (nome proprio)
          * trigrama no texto do chunk (termo literal como "Open Finance",
            que aparece no texto mas nao no nome da instituicao)
          * bonus de prefixo exato na razao social

        `min_score` filtra pelo score do vetor; o default -1.0 nao descarta
        nada, para nunca devolver lista vazia quando ha candidatos ruins.
        """
        term = query_text.strip() if query_text and query_text.strip() else None
        weight = float(name_weight) if term else 0.0
        text_weight = float(text_weight) if term else 0.0
        prefix_weight = float(prefix_weight) if term else 0.0

        query, params = build_search_query(
            table=TABLE_NAME,
            vector_literal=to_vector_literal(query_embedding),
            query_text=query_text,
            filters={"uf": uf, "porte": porte, "source": source},
            top_k=top_k,
            name_weight=name_weight,
            text_weight=text_weight,
            prefix_weight=prefix_weight,
        )
        with self.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            vector_score = float(row["vector_score"])
            name_score = float(row["name_score"] or 0.0)
            text_score = float(row["text_score"] or 0.0)
            prefix_hit = int(row["prefix_hit"] or 0)
            if vector_score < min_score:
                continue
            metadata = row["metadata"] or {}
            results.append(
                {
                    "id": int(row["id"]),
                    "cnpj": row["cnpj"],
                    "text_chunk": row["text_chunk"],
                    "score": round(
                        vector_score
                        + weight * name_score
                        + text_weight * text_score
                        + prefix_weight * prefix_hit,
                        4,
                    ),
                    "vector_score": round(vector_score, 4),
                    "name_score": round(name_score, 4),
                    "text_score": round(text_score, 4),
                    "prefix_hit": prefix_hit,
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
