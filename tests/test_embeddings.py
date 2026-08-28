"""Testes de `embeddings.generator` e do SQL de `embeddings.store`.

O teste com o modelo real e pulado automaticamente quando o all-MiniLM-L6-v2
nao esta em cache local nem pode ser baixado (ex.: CI sem rede).
"""

from __future__ import annotations

import numpy as np
import pytest

from embeddings.generator import EmbeddingGenerator
from embeddings.store import DDL_STATEMENTS, TABLE_NAME, to_vector_literal

EXPECTED_DIM = 384


@pytest.fixture(scope="module")
def generator() -> EmbeddingGenerator:
    """Carrega o modelo real uma unica vez para o modulo inteiro."""
    gen = EmbeddingGenerator()
    try:
        gen.warm_up()
    except Exception as exc:  # noqa: BLE001 - ambiente sem modelo/rede
        pytest.skip(f"modelo de embeddings indisponivel: {exc}")
    return gen


# --------------------------------------------------------------------------- #
# Geracao com o modelo real
# --------------------------------------------------------------------------- #
def test_generate_retorna_shape_384_float32(generator: EmbeddingGenerator) -> None:
    textos = [
        "Empresa: Banco do Brasil S.A. CNPJ: 00000000000191. UF: DF",
        "Empresa: Nu Pagamentos S.A. CNPJ: 18236120000158. UF: SP",
    ]
    vetores = generator.generate(textos, show_progress=False)

    assert isinstance(vetores, np.ndarray)
    assert vetores.shape == (2, EXPECTED_DIM)
    assert vetores.dtype == np.float32
    assert generator.dimension == EXPECTED_DIM


def test_vetores_sao_normalizados(generator: EmbeddingGenerator) -> None:
    vetores = generator.generate(["banco digital brasileiro"], show_progress=False)
    assert np.isclose(np.linalg.norm(vetores[0]), 1.0, atol=1e-5)


def test_generate_one_retorna_vetor_1d(generator: EmbeddingGenerator) -> None:
    vetor = generator.generate_one("nubank")
    assert vetor.shape == (EXPECTED_DIM,)
    assert vetor.dtype == np.float32


def test_generate_list_e_serializavel(generator: EmbeddingGenerator) -> None:
    lista = generator.generate_list(["nubank"], show_progress=False)
    assert isinstance(lista, list) and len(lista) == 1
    assert len(lista[0]) == EXPECTED_DIM
    assert all(isinstance(valor, float) for valor in lista[0][:10])


def test_similaridade_maior_entre_textos_relacionados(
    generator: EmbeddingGenerator,
) -> None:
    vetores = generator.generate(
        [
            "Empresa: Banco Itau. CNAE: bancos e servicos financeiros",
            "Empresa: Banco Bradesco. CNAE: bancos e servicos financeiros",
            "Empresa: Padaria do Ze. CNAE: fabricacao de paes",
        ],
        show_progress=False,
    )
    similar = float(vetores[0] @ vetores[1])
    distinto = float(vetores[0] @ vetores[2])
    assert similar > distinto


def test_processa_em_batches(generator: EmbeddingGenerator) -> None:
    # Mais itens que o batch_size exercita o loop de lotes.
    textos = [f"Empresa numero {i}" for i in range(70)]
    vetores = generator.generate(textos, show_progress=False, batch_size=32)
    assert vetores.shape == (70, EXPECTED_DIM)


# --------------------------------------------------------------------------- #
# Casos que nao exigem o modelo carregado
# --------------------------------------------------------------------------- #
def test_lista_vazia_nao_carrega_modelo() -> None:
    gen = EmbeddingGenerator()
    gen._dimension = EXPECTED_DIM  # evita o download so para saber a dimensao
    vetores = gen.generate([], show_progress=False)
    assert vetores.shape == (0, EXPECTED_DIM)
    assert gen._model is None


def test_sanitize_trata_nulos_e_vazios() -> None:
    assert EmbeddingGenerator._sanitize(None) == " "
    assert EmbeddingGenerator._sanitize("   ") == " "
    assert EmbeddingGenerator._sanitize("texto\x00sujo") == "texto sujo"


# --------------------------------------------------------------------------- #
# SQL do VectorStore (sem conexao)
# --------------------------------------------------------------------------- #
def test_to_vector_literal_formata_para_pgvector() -> None:
    literal = to_vector_literal(np.array([0.1, -0.25, 3.0], dtype=np.float32))
    assert literal == "[0.1,-0.25,3]"


def test_ddl_declara_dimensao_e_indices() -> None:
    ddl = "\n".join(s.format(table=TABLE_NAME, dim=EXPECTED_DIM) for s in DDL_STATEMENTS)

    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl
    assert f"embedding   vector({EXPECTED_DIM}) NOT NULL" in ddl
    assert "metadata    JSONB" in ddl
    assert "created_at  TIMESTAMPTZ" in ddl
    assert "ivfflat (embedding vector_cosine_ops)" in ddl
    assert f"{TABLE_NAME}_cnpj_chunk_uidx" in ddl  # upsert idempotente


# --------------------------------------------------------------------------- #
# Busca hibrida (SQL montado sem conexao)
# --------------------------------------------------------------------------- #
def test_ddl_cria_indice_trigrama_para_nomes() -> None:
    ddl = "\n".join(s.format(table=TABLE_NAME, dim=EXPECTED_DIM) for s in DDL_STATEMENTS)
    assert "CREATE EXTENSION IF NOT EXISTS pg_trgm" in ddl
    assert "gin ((metadata->>'razao_social') gin_trgm_ops)" in ddl


def test_build_search_query_ordem_dos_placeholders() -> None:
    """13 placeholders em 4 sinais: trocar dois corrompe o ranking em silencio."""
    from embeddings.store import build_search_query

    sql, params = build_search_query(
        table="company_embeddings",
        vector_literal="[0.1,0.2]",
        query_text="Nu Pagamentos",
        filters={"uf": "sp", "porte": None, "source": None},
        top_k=5,
    )

    assert sql.count("%s") == len(params) == 13
    # SELECT: vetor, nome, texto, prefixo
    assert params[:4] == ["[0.1,0.2]", "Nu Pagamentos", "Nu Pagamentos", "Nu Pagamentos%"]
    # WHERE: uf normalizada
    assert params[4] == "SP"
    # ORDER BY: vetor, (peso,nome), (peso,texto), (peso,prefixo), limite
    assert params[5:] == [
        "[0.1,0.2]",
        0.4, "Nu Pagamentos",
        0.6, "Nu Pagamentos",
        0.4, "Nu Pagamentos%",
        5,
    ]


def test_build_search_query_sem_texto_zera_os_pesos_lexicais() -> None:
    from embeddings.store import build_search_query

    sql, params = build_search_query(
        table="company_embeddings",
        vector_literal="[0.1]",
        query_text=None,
        filters={},
        top_k=3,
    )
    assert sql.count("%s") == len(params)
    assert 0.4 not in params and 0.6 not in params  # so o cosseno pontua
    assert "~~sem-prefixo~~" in params  # padrao que nao casa com nada


def test_build_search_query_usa_os_quatro_sinais() -> None:
    from embeddings.store import build_search_query

    sql, _ = build_search_query(
        table="company_embeddings",
        vector_literal="[0.1]",
        query_text="itau",
        filters={},
        top_k=3,
    )
    assert "embedding <=>" in sql
    assert "word_similarity(%s, COALESCE(metadata->>'razao_social', ''))" in sql
    assert "word_similarity(%s, text_chunk)" in sql
    assert "ILIKE %s)::int" in sql
    # similarity() inteira empatava nome longo com lixo; tem que ser word_similarity
    assert "similarity(COALESCE" not in sql


def test_build_search_query_aplica_filtros_estruturados() -> None:
    from embeddings.store import build_search_query

    sql, params = build_search_query(
        table="company_embeddings",
        vector_literal="[0.1]",
        query_text=None,
        filters={"uf": "RS", "porte": "01", "source": "bacen"},
        top_k=1,
    )
    assert "metadata->>'uf' = %s" in sql
    assert "metadata->>'source' = %s" in sql
    assert params[4:8] == ["RS", "01", "01", "bacen"]
