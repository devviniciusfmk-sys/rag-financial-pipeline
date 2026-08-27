-- Executado uma unica vez, na inicializacao do cluster.

-- Banco de aplicacao do Metabase (metadados, dashboards, usuarios).
-- Fica separado do banco `postgres`, onde vivem os embeddings.
CREATE DATABASE metabase;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;

-- A criacao da tabela tambem e feita em runtime por VectorStore.ensure_schema(),
-- aqui garantimos o schema mesmo antes da primeira execucao da aplicacao.
CREATE TABLE IF NOT EXISTS company_embeddings (
    id          BIGSERIAL PRIMARY KEY,
    cnpj        VARCHAR(14) NOT NULL,
    text_chunk  TEXT        NOT NULL,
    embedding   vector(384) NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS company_embeddings_cnpj_chunk_uidx
    ON company_embeddings (cnpj, md5(text_chunk));
CREATE INDEX IF NOT EXISTS company_embeddings_cnpj_idx
    ON company_embeddings (cnpj);
CREATE INDEX IF NOT EXISTS company_embeddings_metadata_gin
    ON company_embeddings USING gin (metadata jsonb_path_ops);
CREATE INDEX IF NOT EXISTS company_embeddings_vec_idx
    ON company_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
