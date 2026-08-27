"""Testes de `ingestion.cleaner` com fixtures sinteticas das tres fontes."""

from __future__ import annotations

import json

import pytest

from ingestion.cleaner import (
    NOT_INFORMED,
    build_chunk_text,
    clean_text,
    dedupe_chunks,
    format_capital,
    format_cnpj,
    load_chunks,
    map_porte,
    map_situacao,
    normalize_cnpj,
    parse_capital,
    save_chunks,
)


# --------------------------------------------------------------------------- #
# Normalizacao de CNPJ (inclui calculo de digito verificador)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("00000000", "00000000000191"),  # CNPJ basico do Banco do Brasil + DV
        ("11111111", "11111111000191"),
        ("18.236.120/0001-58", "18236120000158"),
        ("18236120000158", "18236120000158"),
        ("60.701.190/0001-04", "60701190000104"),
    ],
)
def test_normalize_cnpj_calcula_digito_verificador(entrada: str, esperado: str) -> None:
    assert normalize_cnpj(entrada) == esperado


def test_normalize_cnpj_valores_vazios() -> None:
    assert normalize_cnpj(None) == ""
    assert normalize_cnpj("") == ""
    assert normalize_cnpj("sem digitos") == ""


def test_normalize_cnpj_sem_padding_para_matriz() -> None:
    # Fontes que ja trazem o CNPJ completo nao devem ganhar sufixo 0001.
    assert normalize_cnpj("123", pad_to_matriz=False) == "00000000000123"


def test_format_cnpj_aplica_mascara() -> None:
    assert format_cnpj("00000000000191") == "00.000.000/0001-91"


# --------------------------------------------------------------------------- #
# Campos numericos e categoricos
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("90000000000,00", 90000000000.0),
        ("1.000.000,00", 1000000.0),
        ("10000", 10000.0),
        (1234.5, 1234.5),
        ("", 0.0),
        (None, 0.0),
        ("texto invalido", 0.0),
    ],
)
def test_parse_capital(entrada, esperado: float) -> None:
    assert parse_capital(entrada) == esperado


def test_format_capital_usa_padrao_brasileiro() -> None:
    assert format_capital(1234567.5) == "1.234.567,50"
    assert format_capital(0.0) == "0,00"


def test_map_porte_e_situacao() -> None:
    assert map_porte("01").startswith("Micro")
    assert map_porte("03").startswith("Empresa de Pequeno Porte")
    assert map_porte("05").startswith("Demais")
    assert map_situacao("02") == "Ativa"
    assert map_situacao("08") == "Baixada"
    assert map_situacao("ATIVO") == "Ativa"


# --------------------------------------------------------------------------- #
# Limpeza de texto
# --------------------------------------------------------------------------- #
def test_clean_text_remove_controle_e_espacos() -> None:
    assert clean_text("  BANCO   DO\x07 BRASIL  ", title_case=True) == "Banco Do Brasil"


def test_clean_text_trata_nulos() -> None:
    assert clean_text(None) == NOT_INFORMED
    assert clean_text("") == NOT_INFORMED
    assert clean_text("nan") == NOT_INFORMED
    assert clean_text("-") == NOT_INFORMED
    assert clean_text("", default="") == ""


def test_clean_text_preserva_acentos() -> None:
    assert clean_text("ASSOCIAÇÃO SÃO JOÃO") == "ASSOCIAÇÃO SÃO JOÃO"


def test_build_chunk_text_contem_todos_os_campos() -> None:
    texto = build_chunk_text(
        razao_social="Nu Pagamentos S.A.",
        cnpj="18236120000158",
        situacao="Ativa",
        cnae="Servicos Financeiros",
        porte="Demais",
        uf="SP",
        capital=1000.0,
    )
    for token in ("Empresa:", "CNPJ:", "Situacao:", "CNAE:", "Porte:", "UF:", "Capital: R$"):
        assert token in texto
    assert "1.000,00" in texto


# --------------------------------------------------------------------------- #
# Leitura das fontes
# --------------------------------------------------------------------------- #
def test_load_receita_federal_le_zip_latin1(cleaner_with_raw) -> None:
    df = cleaner_with_raw.load_receita_federal(limit=100)
    # 3 linhas no CSV, mas a de razao social vazia e descartada
    assert len(df) == 2
    assert df.iloc[0]["cnpj"] == "00000000000191"
    assert df.iloc[0]["razao_social"] == "Banco Do Brasil Sa"
    assert df.iloc[0]["capital_social_num"] == 90000000000.0
    assert df.iloc[1]["porte_cod"] == "01"


def test_load_cvm_le_csv_com_cabecalho(cleaner_with_raw) -> None:
    df = cleaner_with_raw.load_cvm()
    assert len(df) == 2
    assert set(df["cnpj"]) == {"00000000000191", "18236120000158"}
    assert set(df["uf"]) == {"DF", "SP"}
    assert df.iloc[0]["situacao_desc"] == "Ativa"


def test_load_bacen_descarta_registro_sem_cnpj(cleaner_with_raw) -> None:
    df = cleaner_with_raw.load_bacen()
    assert len(df) == 2  # o terceiro item do JSON nao tem CNPJ
    assert "18236120000158" in set(df["cnpj"])
    assert df.iloc[0]["source"] == "bacen"


def test_load_fonte_ausente_retorna_vazio(tmp_path, monkeypatch) -> None:
    from ingestion import cleaner

    monkeypatch.setattr(cleaner, "RAW_DIR", tmp_path)
    assert cleaner.load_receita_federal(limit=10).empty
    assert cleaner.load_cvm().empty
    assert cleaner.load_bacen().empty


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_build_all_chunks_agrega_as_tres_fontes(cleaner_with_raw) -> None:
    chunks = cleaner_with_raw.build_all_chunks(limit=100)

    contagem: dict[str, int] = {}
    for chunk in chunks:
        fonte = chunk.metadata["source"]
        contagem[fonte] = contagem.get(fonte, 0) + 1

    assert contagem == {"receita_federal": 2, "cvm": 2, "bacen": 2}
    assert len(chunks) == 6


def test_chunk_tem_formato_e_metadata_esperados(cleaner_with_raw) -> None:
    chunks = cleaner_with_raw.build_all_chunks(limit=100)
    chunk = chunks[0]

    assert len(chunk.cnpj) == 14 and chunk.cnpj.isdigit()
    for token in ("Empresa:", "CNPJ:", "Situacao:", "CNAE:", "Porte:", "UF:", "Capital: R$"):
        assert token in chunk.text
    for chave in ("razao_social", "cnpj_formatado", "situacao", "cnae", "porte", "uf",
                  "capital_social", "source"):
        assert chave in chunk.metadata


def test_dedupe_chunks_preserva_ordem(cleaner_with_raw) -> None:
    chunks = cleaner_with_raw.build_all_chunks(limit=100)
    assert dedupe_chunks(chunks + chunks) == chunks


def test_save_e_load_chunks_roundtrip(cleaner_with_raw, tmp_path) -> None:
    chunks = cleaner_with_raw.build_all_chunks(limit=100)
    destino = tmp_path / "chunks.jsonl"

    save_chunks(chunks, destino)
    assert destino.exists()

    linhas = destino.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == len(chunks)
    assert set(json.loads(linhas[0])) == {"cnpj", "text", "metadata"}

    recarregados = list(load_chunks(destino))
    assert recarregados == chunks


def test_load_chunks_arquivo_inexistente(tmp_path) -> None:
    assert list(load_chunks(tmp_path / "nao-existe.jsonl")) == []
