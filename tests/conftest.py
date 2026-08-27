"""Fixtures compartilhadas e stubs para rodar a suite sem banco de dados.

Nenhum teste abre conexao real com o Postgres: `VectorStore` e substituido por
fakes. Quando o `psycopg2` nao esta instalado (ex.: ambiente so de lint/CI leve),
injetamos um stub minimo para que os imports de `embeddings.store` funcionem.
"""

from __future__ import annotations

import json
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest


def _install_psycopg2_stub() -> None:
    """Registra um psycopg2 falso em sys.modules (apenas se o real faltar)."""
    try:  # pragma: no cover - caminho normal, com o driver instalado
        import psycopg2  # noqa: F401

        return
    except ImportError:  # pragma: no cover - ambiente sem driver
        pass

    psycopg2 = types.ModuleType("psycopg2")

    class Error(Exception):
        """Equivalente a psycopg2.Error."""

    class _Extensions:
        class connection:  # noqa: N801 - espelha o nome real
            ...

        class cursor:  # noqa: N801 - espelha o nome real
            ...

    def connect(*args: Any, **kwargs: Any):
        raise Error("psycopg2 stub: sem banco disponivel nos testes")

    psycopg2.Error = Error
    psycopg2.extensions = _Extensions
    psycopg2.connect = connect

    sql = types.ModuleType("psycopg2.sql")

    class SQL:
        def __init__(self, statement: str) -> None:
            self.statement = statement

        def format(self, *args: Any, **kwargs: Any) -> str:
            return self.statement

    class Identifier(str):
        ...

    sql.SQL = SQL
    sql.Identifier = Identifier

    extras = types.ModuleType("psycopg2.extras")

    class Json:
        def __init__(self, obj: Any) -> None:
            self.obj = obj

    class RealDictCursor:
        ...

    def execute_values(cur, statement, rows, template=None, page_size=100):
        return None

    extras.Json = Json
    extras.RealDictCursor = RealDictCursor
    extras.execute_values = execute_values

    pool = types.ModuleType("psycopg2.pool")

    class ThreadedConnectionPool:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise Error("psycopg2 stub: sem banco disponivel nos testes")

    pool.ThreadedConnectionPool = ThreadedConnectionPool

    psycopg2.sql = sql
    psycopg2.extras = extras
    psycopg2.pool = pool

    sys.modules.setdefault("psycopg2", psycopg2)
    sys.modules.setdefault("psycopg2.sql", sql)
    sys.modules.setdefault("psycopg2.extras", extras)
    sys.modules.setdefault("psycopg2.pool", pool)


_install_psycopg2_stub()


# --------------------------------------------------------------------------- #
# Amostras sinteticas das tres fontes publicas
# --------------------------------------------------------------------------- #
RFB_CSV = (
    '"00000000";"BANCO DO BRASIL SA";"2038";"16";"90000000000,00";"05";""\n'
    '"11111111";"PADARIA DO ZE LTDA";"2062";"49";"10000,00";"01";""\n'
    '"22222222";"";"2062";"49";"0,00";"00";""\n'
)

CVM_CSV = (
    "CNPJ_CIA;DENOM_SOCIAL;DENOM_COMERC;SIT;CD_CVM;SETOR_ATIV;TP_MERC;CATEG_REG;UF;MUN\n"
    "00.000.000/0001-91;BANCO DO BRASIL S.A.;BB;ATIVO;1023;Bancos;Bolsa;"
    "Categoria A;DF;BRASILIA\n"
    "18.236.120/0001-58;NU PAGAMENTOS S.A.;NUBANK;ATIVO;2437;Servicos Financeiros;"
    "Bolsa;Categoria A;SP;SAO PAULO\n"
)

BACEN_JSON: list[dict[str, Any]] = [
    {
        "OrganisationId": "abc",
        "OrganisationName": "Nu Pagamentos S.A.",
        "RegistrationNumber": "18236120000158",
        "Status": "Active",
        "City": "Sao Paulo",
        "Size": "Large",
        "AuthorisationServers": [{"a": 1}, {"b": 2}],
    },
    {
        "OrganisationId": "def",
        "OrganisationName": "Itau Unibanco S.A.",
        "RegistrationNumber": "60701190000104",
        "Status": "Active",
        "City": "Sao Paulo",
        "AuthorisationServers": [],
    },
    {"OrganisationName": "registro sem cnpj"},
]


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Diretorio data/raw sintetico com as tres fontes ja "baixadas"."""
    target = tmp_path / "raw"
    target.mkdir()

    with zipfile.ZipFile(target / "Empresas0.zip", "w") as zf:
        zf.writestr("K3241.K03200Y0.D40406.EMPRECSV", RFB_CSV.encode("latin-1"))

    (target / "cad_cia_aberta.csv").write_bytes(CVM_CSV.encode("latin-1"))
    (target / "open_finance_participants.json").write_text(
        json.dumps(BACEN_JSON), encoding="utf-8"
    )
    return target


@pytest.fixture
def cleaner_with_raw(raw_dir: Path, monkeypatch: pytest.MonkeyPatch):
    """Modulo `ingestion.cleaner` apontando para o raw_dir sintetico."""
    from ingestion import cleaner

    monkeypatch.setattr(cleaner, "RAW_DIR", raw_dir)
    return cleaner


# --------------------------------------------------------------------------- #
# Fakes de infraestrutura
# --------------------------------------------------------------------------- #
SAMPLE_ROW: dict[str, Any] = {
    "id": 1,
    "cnpj": "18236120000158",
    "text_chunk": (
        "Empresa: Nu Pagamentos S.A. CNPJ: 18236120000158. Situacao: Ativa. "
        "CNAE: Servicos Financeiros. Porte: Companhia aberta. UF: SP. "
        "Capital: R$ 0,00"
    ),
    "score": 0.9231,
    "created_at": None,
    "metadata": {
        "razao_social": "Nu Pagamentos S.A.",
        "cnpj_formatado": "18.236.120/0001-58",
        "uf": "SP",
        "porte": "Companhia aberta",
        "porte_cod": "05",
        "situacao": "Ativa",
        "cnae": "Servicos Financeiros",
        "capital_social": 0.0,
        "source": "bacen",
    },
}


class FakeStore:
    """Substituto de `VectorStore` que nao toca em banco algum."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows if rows is not None else [SAMPLE_ROW]
        self.closed = False
        self.schema_calls = 0

    def ensure_schema(self) -> None:
        self.schema_calls += 1

    def close(self) -> None:
        self.closed = True

    def count(self) -> int:
        return len(self.rows)

    def health(self) -> dict[str, Any]:
        return {
            "connected": True,
            "table_exists": True,
            "pgvector": "0.7.4",
            "version": "PostgreSQL 16.3",
            "rows": len(self.rows),
        }

    def search(self, query_embedding, top_k: int = 5, **kwargs: Any):
        return self.rows[:top_k]

    def get_by_cnpj(self, cnpj: str):
        return [row for row in self.rows if row["cnpj"] == cnpj]

    def list_companies(self, **kwargs: Any):
        limit = int(kwargs.get("limit", 20))
        return self.rows[:limit]

    def stats(self) -> dict[str, Any]:
        return {
            "total_companies": 1234,
            "total_embeddings": 1234,
            "sources": ["bacen", "cvm", "receita_federal"],
            "by_source": {"receita_federal": 1000, "cvm": 200, "bacen": 34},
            "top_uf": {"SP": 800},
            "last_ingestion": None,
            "embedding_dim": 384,
        }


class FakeGenerator:
    """Substituto de `EmbeddingGenerator` (vetores zerados, sem carregar torch)."""

    dimension = 384

    def __init__(self) -> None:
        self.warm_ups = 0

    def warm_up(self) -> None:
        self.warm_ups += 1

    def generate_one(self, text: str):
        import numpy as np

        return np.zeros(self.dimension, dtype="float32")

    def generate(self, texts, **kwargs: Any):
        import numpy as np

        return np.zeros((len(list(texts)), self.dimension), dtype="float32")


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def fake_generator() -> FakeGenerator:
    return FakeGenerator()
