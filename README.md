# RAG Financial Pipeline (Brasil)

Pipeline RAG sobre dados públicos financeiros brasileiros: **Receita Federal**
(cadastro de empresas), **CVM** (companhias abertas) e **Open Finance Brasil**
(diretório de participantes).

Stack: Python · FastAPI · pgvector · sentence-transformers · OpenRouter · tenacity · rich

## Arquitetura

```
ingestion/downloader.py  -> baixa os datasets (tqdm + retry + skip)
ingestion/cleaner.py     -> limpa, normaliza CNPJ e monta os chunks de texto
embeddings/generator.py  -> all-MiniLM-L6-v2 local, batches de 32 (384 dims)
embeddings/store.py      -> PostgreSQL + pgvector (company_embeddings)
retrieval/search.py      -> query -> embedding -> busca por cosseno
pipeline/chain.py        -> contexto + prompt + OpenRouter (com fallback)
api/main.py + routes.py  -> FastAPI: /health /ask /search /companies /stats
scripts/build_index.py   -> orquestra tudo de ponta a ponta
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                # preencha OPENROUTER_API_KEY

docker compose up -d                                # Postgres 16 + pgvector na porta 54322
docker compose ps                                   # aguarde o healthcheck ficar "healthy"
```

## Ingestão

```bash
python -m ingestion.downloader --extract      # só baixa (data/raw)
python -m ingestion.cleaner --limit 5000      # só limpa (data/processed/chunks.jsonl)
python -m scripts.build_index --limit 5000    # download + limpeza + embeddings + pgvector
python -m scripts.build_index --skip-download --reset
```

> `Empresas0.zip` da Receita tem centenas de MB. O modo demo lê apenas
> `DEMO_ROW_LIMIT` linhas (padrão 20.000); ajuste com `--limit`.
> Se você também baixar `Estabelecimentos0.zip` para `data/raw/`, o cleaner
> enriquece automaticamente cada empresa com UF, CNAE e situação cadastral.

## API

```bash
uvicorn api.main:app --reload --port 8000     # docs em http://localhost:8000/docs
```

| Método | Rota         | Exemplo                                                  |
| ------ | ------------ | -------------------------------------------------------- |
| GET    | `/health`    | status do banco, do modelo e do LLM                       |
| POST   | `/ask`       | `{"question": "Quais bancos participam do Open Finance?"}`|
| GET    | `/search`    | `/search?q=nubank&limit=5`                                |
| GET    | `/companies` | `/companies?uf=RS&porte=01&limit=20`                      |
| GET    | `/stats`     | totais por fonte, UF e última ingestão                    |

```bash
curl -s localhost:8000/stats
curl -s "localhost:8000/search?q=nubank&limit=5"
curl -s localhost:8000/ask -H "content-type: application/json" \
     -d '{"question":"Quais bancos participam do Open Finance?"}'
```

## Testes

```bash
pip install -r requirements-dev.txt
pytest                      # 70 testes
pytest tests/test_cleaner.py -v
```

| Arquivo | Cobre |
| --- | --- |
| `tests/test_cleaner.py` | normalização de CNPJ (com DV), capital, porte/situação, leitura das 3 fontes a partir de fixtures sintéticas, chunking e round-trip do JSONL |
| `tests/test_embeddings.py` | modelo real: shape `(n, 384)`, `float32`, vetores normalizados, batches; DDL e literal do pgvector |
| `tests/test_api.py` | `/health`, `/search`, `/companies`, `/stats`, `/ask`, CORS, validações 422 e erros 500/502/503 |
| `tests/test_chain.py` | fallback de modelo, sem contexto (não chama o LLM), chave inválida e falha de todos os modelos |

Nenhum teste precisa de banco, rede ou chave de API: `tests/conftest.py` injeta
fakes de `VectorStore`/`RAGChain` e um stub de `psycopg2` quando o driver não
está instalado. O teste do modelo real é pulado se o all-MiniLM não estiver em
cache nem puder ser baixado.

## Modelos

`LLM_MODEL` é tentado primeiro; qualquer falha (rate limit, timeout, modelo
indisponível) cai automaticamente para `FALLBACK_MODEL`. A resposta de `/ask`
traz o campo `model` com o modelo que efetivamente respondeu e `fallback_used`.

## Banco

Tabela `company_embeddings`:

| coluna       | tipo          |
| ------------ | ------------- |
| `id`         | BIGSERIAL PK  |
| `cnpj`       | VARCHAR(14)   |
| `text_chunk` | TEXT          |
| `embedding`  | vector(384)   |
| `metadata`   | JSONB         |
| `created_at` | TIMESTAMPTZ   |

Índices: `ivfflat (embedding vector_cosine_ops)`, GIN em `metadata`, e único em
`(cnpj, md5(text_chunk))` — que torna a ingestão idempotente (upsert).
