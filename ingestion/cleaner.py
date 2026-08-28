"""Leitura, limpeza e chunking dos datasets baixados por `ingestion.downloader`.

Saida canonica: uma lista de `Chunk` (cnpj, text, metadata), pronta para virar
embedding. O texto segue sempre o mesmo template:

    "Empresa: {razao_social}. CNPJ: {cnpj}. Situacao: {situacao}.
     CNAE: {cnae}. Porte: {porte}. UF: {uf}. Capital: R$ {capital}"
"""

from __future__ import annotations

import json
import re
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from config import PROCESSED_DIR, RAW_DIR, settings, setup_logging

logger = setup_logging()

ENCODING = "latin-1"
SEPARATOR = ";"
NOT_INFORMED = "NAO INFORMADO"

NULL_TOKENS = {
    "",
    "-",
    "--",
    "n/a",
    "na",
    "nan",
    "none",
    "null",
    "nao informado",
    "não informado",
    "nao informada",
    "não informada",
    "0",
    "00000000",
    "*",
}

# Layout oficial do arquivo EMPRESAS da Receita Federal (sem cabecalho)
RFB_EMPRESAS_COLUMNS = [
    "cnpj_basico",
    "razao_social",
    "natureza_juridica",
    "qualificacao_responsavel",
    "capital_social",
    "porte",
    "ente_federativo_responsavel",
]

# Layout do arquivo ESTABELECIMENTOS (usado apenas se estiver presente em data/raw)
RFB_ESTAB_COLUMNS = [
    "cnpj_basico",
    "cnpj_ordem",
    "cnpj_dv",
    "identificador_matriz_filial",
    "nome_fantasia",
    "situacao_cadastral",
    "data_situacao_cadastral",
    "motivo_situacao_cadastral",
    "nome_cidade_exterior",
    "pais",
    "data_inicio_atividade",
    "cnae_fiscal_principal",
    "cnae_fiscal_secundaria",
    "tipo_logradouro",
    "logradouro",
    "numero",
    "complemento",
    "bairro",
    "cep",
    "uf",
    "municipio",
    "ddd_1",
    "telefone_1",
    "ddd_2",
    "telefone_2",
    "ddd_fax",
    "fax",
    "correio_eletronico",
    "situacao_especial",
    "data_situacao_especial",
]

PORTE_MAP = {
    "00": "Nao informado",
    "01": "Microempresa (ME)",
    "03": "Empresa de Pequeno Porte (EPP)",
    "05": "Demais (medio/grande porte)",
}

SITUACAO_MAP = {
    "01": "Nula",
    "02": "Ativa",
    "03": "Suspensa",
    "04": "Inapta",
    "08": "Baixada",
}

CVM_SITUACAO_MAP = {
    "ATIVO": "Ativa",
    "CANCELADA": "Cancelada",
    "EM ANALISE": "Em analise",
}


@dataclass
class Chunk:
    """Unidade de texto do indice.

    `text` e o que se mostra ao usuario (template completo, legivel).
    `embed_text` e o que vira vetor: so o conteudo discriminante, sem o
    boilerplate que se repete em todos os registros e dilui o embedding.
    """

    cnpj: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embed_text: str = ""

    def __post_init__(self) -> None:
        if not self.embed_text:
            self.embed_text = self.text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Normalizacao de campos
# --------------------------------------------------------------------------- #
def clean_text(value: Any, default: str = NOT_INFORMED, title_case: bool = False) -> str:
    """Remove nulos, caracteres de controle e espacos redundantes."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    text = str(value)
    if text.lower().strip() in {"nan", "nat", "none"}:
        return default

    # Normaliza compatibilidade unicode e descarta categorias de controle
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # Caracteres tipograficos que a latin-1 costuma trazer sujos
    text = text.replace("�", " ").replace("\x00", " ")
    text = re.sub(r"[\"'`^~]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" .,;-")

    if text.lower() in NULL_TOKENS:
        return default
    if title_case and text.isupper():
        text = text.title()
    return text or default


def normalize_cnpj(value: Any, pad_to_matriz: bool = True) -> str:
    """Retorna somente digitos. CNPJ basico (8) vira matriz 0001 + DV calculado."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    digits = re.sub(r"\D", "", str(value))
    if not digits:
        return ""
    if len(digits) >= 14:
        return digits[:14]
    if len(digits) == 12:
        return digits + _cnpj_check_digits(digits)
    if len(digits) <= 8 and pad_to_matriz:
        base = digits.zfill(8)
        partial = base + "0001"
        return partial + _cnpj_check_digits(partial)
    return digits.zfill(14)


def _cnpj_check_digits(first_twelve: str) -> str:
    """Calcula os dois digitos verificadores de um CNPJ (modulo 11)."""
    def dv(numbers: str, weights: list[int]) -> str:
        total = sum(int(n) * w for n, w in zip(numbers, weights))
        rest = total % 11
        return "0" if rest < 2 else str(11 - rest)

    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    w2 = [6] + w1
    d1 = dv(first_twelve, w1)
    d2 = dv(first_twelve + d1, w2)
    return d1 + d2


def format_cnpj(cnpj: str) -> str:
    """00000000000191 -> 00.000.000/0001-91"""
    d = re.sub(r"\D", "", cnpj or "").zfill(14)[:14]
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"


def parse_capital(value: Any) -> float:
    """Converte '1000000,00' / '1.000.000,00' / 1000000.0 em float."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return 0.0
    raw = re.sub(r"[^\d,.\-]", "", raw)
    if "," in raw:  # formato brasileiro
        raw = raw.replace(".", "").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return 0.0


def format_capital(value: float) -> str:
    """1000000.0 -> '1.000.000,00'"""
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def normalize_municipio(value: Any) -> str:
    """"Sao Paulo, Sp" e "Sao Paulo" viram a mesma cidade.

    O diretorio do Open Finance as vezes anexa a UF ao nome da cidade, o que
    duplicava a mesma cidade em categorias diferentes nos agrupamentos.
    """
    text = clean_text(value, default="")
    if not text:
        return ""
    return text.split(",")[0].strip().title()[:60]


def map_porte(code: Any) -> str:
    raw = clean_text(code, default="")
    key = re.sub(r"\D", "", raw).zfill(2)[:2]
    return PORTE_MAP.get(key, raw or NOT_INFORMED)


def map_situacao(code: Any) -> str:
    raw = clean_text(code, default="")
    key = re.sub(r"\D", "", raw).zfill(2)[:2]
    if key in SITUACAO_MAP:
        return SITUACAO_MAP[key]
    upper = raw.upper()
    return CVM_SITUACAO_MAP.get(upper, raw or NOT_INFORMED)


def build_chunk_text(
    razao_social: str,
    cnpj: str,
    situacao: str,
    cnae: str,
    porte: str,
    uf: str,
    capital: float,
) -> str:
    """Template unico de texto para embedding."""
    return (
        f"Empresa: {razao_social}. "
        f"CNPJ: {cnpj}. "
        f"Situacao: {situacao}. "
        f"CNAE: {cnae}. "
        f"Porte: {porte}. "
        f"UF: {uf}. "
        f"Capital: R$ {format_capital(capital)}"
    )


def build_embed_text(
    razao_social: str,
    cnae: str,
    porte: str,
    uf: str,
    municipio: str = "",
    segmento: str = "",
    situacao: str = "",
) -> str:
    """Texto enxuto que vai para o vetor.

    Sem rotulos fixos e sem campos vazios: o template completo era identico em
    todos os 22 mil registros, dominava o embedding e afundava a similaridade
    de consultas curtas como "Nu Pagamentos".

    Ex.: "Nu Pagamentos S.A. - instituicao de pagamento, Open Finance,
          segmento S1/S2, Sao Paulo"
    """
    partes = [razao_social.strip()]
    extras = [
        cnae,
        segmento and f"segmento {segmento}",
        porte,
        municipio,
        uf if uf and uf != NOT_INFORMED else "",
        situacao if situacao and situacao not in {NOT_INFORMED, "Ativa"} else "",
    ]
    limpos = [
        str(e).strip()
        for e in extras
        if e and str(e).strip() and str(e).strip() != NOT_INFORMED
    ]
    vistos: set[str] = set()
    unicos = [e for e in limpos if not (e.lower() in vistos or vistos.add(e.lower()))]
    return f"{partes[0]} - {', '.join(unicos)}" if unicos else partes[0]


# --------------------------------------------------------------------------- #
# Leitura dos arquivos brutos
# --------------------------------------------------------------------------- #
def _series(df: pd.DataFrame, column: str, default: Any = "") -> pd.Series:
    """Devolve a coluna pedida ou uma Series constante do mesmo tamanho."""
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index, dtype=object)


def _resolve_limit(limit: int | None) -> int | None:
    """Normaliza o parametro `limit` das funcoes de leitura.

    `None` ou qualquer valor <= 0 significam "sem teto" (le o arquivo inteiro);
    valores positivos limitam o numero de linhas lidas por fonte.
    """
    if limit is None or limit <= 0:
        return None
    return limit


def _read_csv(
    handle: Any,
    columns: list[str] | None = None,
    limit: int | None = None,
    header: int | None = None,
) -> pd.DataFrame:
    """read_csv com os defaults dos dados abertos brasileiros."""
    df = pd.read_csv(
        handle,
        sep=SEPARATOR,
        encoding=ENCODING,
        header=header,
        names=columns,
        dtype=str,
        nrows=limit,
        quotechar='"',
        keep_default_na=False,
        na_values=["", "NA", "NULL"],
        on_bad_lines="skip",
    )
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _read_csv_chunks(handle: Any, columns: list[str], chunksize: int = 200_000):
    """Mesmo dialeto de `_read_csv`, porem em blocos (arquivos de varios GB)."""
    return pd.read_csv(
        handle,
        sep=SEPARATOR,
        encoding=ENCODING,
        header=None,
        names=columns,
        dtype=str,
        quotechar='"',
        keep_default_na=False,
        na_values=["", "NA", "NULL"],
        on_bad_lines="skip",
        chunksize=chunksize,
    )


def _open_rfb_member(zip_path: Path, marker: str) -> tuple[str, Any] | None:
    """Localiza dentro do zip o membro cujo nome contem `marker` (ex.: EMPRE)."""
    zf = zipfile.ZipFile(zip_path)
    for name in zf.namelist():
        if marker.upper() in name.upper():
            return name, zf.open(name)
    zf.close()
    return None


def load_receita_federal(
    path: Path | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Le Empresas0.zip (ou o CSV ja extraido) e retorna o DataFrame limpo.

    `limit=None` ou `limit<=0` lem o arquivo inteiro.
    """
    zip_path = path or (RAW_DIR / "Empresas0.zip")
    limit = _resolve_limit(limit)

    if not zip_path.exists():
        logger.warning("receita federal: arquivo ausente em %s", zip_path)
        return pd.DataFrame(columns=RFB_EMPRESAS_COLUMNS)

    # Fase 1: os estabelecimentos definem quais CNPJs entram no lote. Sem isso,
    # as primeiras N linhas de cada arquivo nao se correspondem e o join resulta
    # em quase nada -- que era a causa do "UF: NAO INFORMADO" em massa.
    estab = _load_rfb_estabelecimentos(limit=limit)
    keys: set[str] | None = None
    if estab is not None and not estab.empty:
        keys = set(estab["cnpj_basico"])
        logger.info("receita federal: %d matrizes com UF/CNAE disponiveis", len(keys))

    df = _read_rfb_empresas(zip_path, limit=limit, keys=keys)
    if df.empty:
        return pd.DataFrame(columns=RFB_EMPRESAS_COLUMNS)

    df = df.dropna(subset=["razao_social"])
    df["cnpj"] = df["cnpj_basico"].map(normalize_cnpj)
    df["razao_social"] = df["razao_social"].map(lambda v: clean_text(v, title_case=True))
    df["capital_social_num"] = df["capital_social"].map(parse_capital)
    df["porte_desc"] = df["porte"].map(map_porte)
    df["porte_cod"] = df["porte"].map(
        lambda v: re.sub(r"\D", "", clean_text(v, default="")).zfill(2)[:2]
    )
    df = df[df["cnpj"] != ""]

    if estab is not None and not estab.empty:
        df = df.merge(estab, on="cnpj_basico", how="left")
        enriquecidos = int(df["uf"].notna().sum()) if "uf" in df.columns else 0
        logger.info(
            "receita federal: %d/%d registros enriquecidos com UF/CNAE",
            enriquecidos,
            len(df),
        )

    for col, default in (
        ("uf", NOT_INFORMED),
        ("cnae_fiscal_principal", NOT_INFORMED),
        ("situacao_cadastral", NOT_INFORMED),
    ):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)

    df["situacao_desc"] = df["situacao_cadastral"].map(map_situacao)
    df["source"] = "receita_federal"
    logger.info("receita federal: %d registros limpos", len(df))
    return df


def _read_rfb_empresas(
    zip_path: Path, limit: int | None, keys: set[str] | None
) -> pd.DataFrame:
    """Le o arquivo EMPRESAS, opcionalmente so os cnpj_basico em `keys`.

    Sem `keys`, faz uma leitura direta com `nrows=limit`. Com `keys`, varre em
    blocos e para assim que juntar `limit` empresas do conjunto pedido.
    """
    is_zip = zip_path.suffix.lower() == ".zip"

    def _open():
        if not is_zip:
            return zip_path, None
        member = _open_rfb_member(zip_path, "EMPRE")
        if member is None:
            return None, None
        return member[1], member[0]

    handle, member_name = _open()
    if handle is None:
        logger.warning("receita federal: nenhum membro EMPRECSV em %s", zip_path)
        return pd.DataFrame(columns=RFB_EMPRESAS_COLUMNS)

    logger.info(
        "lendo %s%s (limite=%s%s)",
        zip_path.name,
        f"::{member_name}" if member_name else "",
        limit or "sem teto",
        ", filtrado por estabelecimentos" if keys else "",
    )

    if keys is None:
        try:
            return _read_csv(handle, columns=RFB_EMPRESAS_COLUMNS, limit=limit)
        finally:
            if member_name:
                handle.close()

    blocos: list[pd.DataFrame] = []
    total = 0
    try:
        for bloco in _read_csv_chunks(handle, RFB_EMPRESAS_COLUMNS):
            bloco.columns = [str(c).strip().lower() for c in bloco.columns]
            filtrado = bloco[bloco["cnpj_basico"].isin(keys)]
            if not filtrado.empty:
                blocos.append(filtrado)
                total += len(filtrado)
            if limit and total >= limit:
                break
    finally:
        if member_name:
            handle.close()

    if not blocos:
        return pd.DataFrame(columns=RFB_EMPRESAS_COLUMNS)
    df = pd.concat(blocos, ignore_index=True)
    return df.head(limit) if limit else df


def _load_rfb_estabelecimentos(limit: int | None = None) -> pd.DataFrame | None:
    """Le Estabelecimentos*.zip, se presente, e devolve so as matrizes.

    O arquivo tem varios GB e mistura matrizes e filiais, entao a leitura e
    feita em blocos ate juntar `limit` matrizes (None = arquivo inteiro).
    Recebe o limite ja resolvido por `_resolve_limit`.
    """
    candidates = list(RAW_DIR.glob("Estabelecimentos*.zip")) + list(
        RAW_DIR.glob("*ESTABELE*")
    )
    if not candidates:
        logger.info(
            "estabelecimentos ausente: UF/CNAE ficarao como '%s' "
            "(baixe Estabelecimentos0.zip para enriquecer)",
            NOT_INFORMED,
        )
        return None

    target = candidates[0]
    keep = ["cnpj_basico", "uf", "cnae_fiscal_principal", "situacao_cadastral"]
    handle: Any = target
    member_name: str | None = None

    try:
        if target.suffix.lower() == ".zip":
            member = _open_rfb_member(target, "ESTABELE")
            if member is None:
                return None
            member_name, handle = member[0], member[1]

        matrizes: list[pd.DataFrame] = []
        total = 0
        for bloco in _read_csv_chunks(handle, RFB_ESTAB_COLUMNS):
            bloco.columns = [str(c).strip().lower() for c in bloco.columns]
            somente_matriz = bloco[
                bloco["identificador_matriz_filial"].astype(str).str.strip() == "1"
            ]
            if not somente_matriz.empty:
                matrizes.append(somente_matriz[keep])
                total += len(somente_matriz)
            if limit and total >= limit:
                break
    except (zipfile.BadZipFile, OSError, ValueError, KeyError) as exc:
        logger.warning("estabelecimentos ignorado (%s): %s", target.name, exc)
        return None
    finally:
        if member_name and hasattr(handle, "close"):
            handle.close()

    if not matrizes:
        return None

    df = pd.concat(matrizes, ignore_index=True)
    if limit:
        df = df.head(limit)
    df = df.drop_duplicates(subset=["cnpj_basico"])
    df["uf"] = df["uf"].map(lambda v: clean_text(v, default="").upper())
    df["cnae_fiscal_principal"] = df["cnae_fiscal_principal"].map(
        lambda v: clean_text(v, default="")
    )
    logger.info("estabelecimentos: %d matrizes lidas de %s", len(df), target.name)
    return df


def load_cvm(path: Path | None = None, limit: int | None = None) -> pd.DataFrame:
    """Le cad_cia_aberta.csv (companhias abertas registradas na CVM)."""
    csv_path = path or (RAW_DIR / "cad_cia_aberta.csv")
    if not csv_path.exists():
        logger.warning("cvm: arquivo ausente em %s", csv_path)
        return pd.DataFrame()

    df = _read_csv(csv_path, limit=_resolve_limit(limit), header=0)
    rename = {
        "cnpj_cia": "cnpj_raw",
        "denom_social": "razao_social",
        "denom_comerc": "nome_fantasia",
        "sit": "situacao_cadastral",
        "setor_ativ": "setor_atividade",
        "uf": "uf",
        "cd_cvm": "codigo_cvm",
        "categ_reg": "categoria_registro",
        "tp_merc": "tipo_mercado",
        "mun": "municipio",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    if "razao_social" not in df.columns:
        logger.warning("cvm: layout inesperado, colunas=%s", list(df.columns)[:10])
        return pd.DataFrame()

    df["cnpj"] = _series(df, "cnpj_raw").map(lambda v: normalize_cnpj(v, pad_to_matriz=False))
    df["razao_social"] = df["razao_social"].map(lambda v: clean_text(v, title_case=True))
    df["uf"] = _series(df, "uf", NOT_INFORMED).map(lambda v: clean_text(v).upper())
    df["situacao_desc"] = _series(df, "situacao_cadastral").map(map_situacao)
    df["setor_atividade"] = _series(df, "setor_atividade", NOT_INFORMED).map(clean_text)
    df["capital_social_num"] = 0.0
    df["porte_desc"] = "Companhia aberta"
    df["porte_cod"] = "05"
    df["source"] = "cvm"
    df = df[(df["cnpj"] != "") & (df["razao_social"] != NOT_INFORMED)]
    df = df.drop_duplicates(subset=["cnpj"])
    logger.info("cvm: %d registros limpos", len(df))
    return df


def _bacen_setor(size: Any) -> str:
    """Descricao usada no campo CNAE do chunk, preservando o segmento."""
    segmento = clean_text(size, default="")
    base = "Instituicao participante do Open Finance Brasil"
    return f"{base} - segmento {segmento}" if segmento else base


def load_bacen(path: Path | None = None) -> pd.DataFrame:
    """Le o diretorio de participantes do Open Finance (JSON)."""
    json_path = path or (RAW_DIR / "open_finance_participants.json")
    if not json_path.exists():
        logger.warning("bacen: arquivo ausente em %s", json_path)
        return pd.DataFrame()

    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        payload = payload.get("data") or payload.get("participants") or []
    if not isinstance(payload, list):
        logger.warning("bacen: payload inesperado (%s)", type(payload).__name__)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        cnpj = normalize_cnpj(item.get("RegistrationNumber"), pad_to_matriz=False)
        name = clean_text(
            item.get("OrganisationName")
            or item.get("LegalEntityName")
            or item.get("RegisteredName"),
            title_case=True,
        )
        if not cnpj or name == NOT_INFORMED:
            continue
        servers = item.get("AuthorisationServers") or []
        rows.append(
            {
                "cnpj": cnpj,
                "razao_social": name,
                "situacao_desc": clean_text(item.get("Status"), default="Ativa"),
                # O diretorio do Open Finance publica cidade, nao UF. Misturar as
                # duas contaminava o agrupamento por estado com nomes de cidade.
                "uf": NOT_INFORMED,
                "municipio": normalize_municipio(item.get("City")),
                # `Size` no diretorio do Open Finance e o segmento prudencial do
                # Bacen (S1/S2, SCD/SEP, IP...), nao o porte da empresa. Vai para
                # a chave propria `segmento` e continua visivel no texto do chunk.
                "segmento": clean_text(item.get("Size"), default=""),
                "setor_atividade": _bacen_setor(item.get("Size")),
                "capital_social_num": 0.0,
                "porte_desc": "Instituicao financeira",
                # Sem classificacao de porte da Receita: nao inventamos '05'.
                "porte_cod": "",
                "organisation_id": clean_text(item.get("OrganisationId"), default=""),
                "authorisation_servers": len(servers) if isinstance(servers, list) else 0,
                "source": "bacen",
            }
        )

    df = pd.DataFrame(rows).drop_duplicates(subset=["cnpj"])
    logger.info("bacen: %d participantes limpos", len(df))
    return df


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def _row_to_chunk(row: dict[str, Any]) -> Chunk | None:
    cnpj = str(row.get("cnpj") or "")
    razao_social = clean_text(row.get("razao_social"))
    if not cnpj or razao_social == NOT_INFORMED:
        return None

    situacao = clean_text(row.get("situacao_desc"))
    cnae = clean_text(
        row.get("cnae_fiscal_principal") or row.get("setor_atividade")
    )
    porte = clean_text(row.get("porte_desc"))
    uf = clean_text(row.get("uf")).upper()
    capital = float(row.get("capital_social_num") or 0.0)
    source = str(row.get("source") or "desconhecida")

    text = build_chunk_text(razao_social, cnpj, situacao, cnae, porte, uf, capital)
    embed_text = build_embed_text(
        razao_social=razao_social,
        cnae=cnae,
        porte=porte,
        uf=uf,
        municipio=str(row.get("municipio") or ""),
        segmento=str(row.get("segmento") or ""),
        situacao=situacao,
    )
    metadata = {
        "razao_social": razao_social,
        "cnpj_formatado": format_cnpj(cnpj),
        "situacao": situacao,
        "cnae": cnae,
        "porte": porte,
        "porte_cod": str(row.get("porte_cod") or ""),
        "uf": uf,
        "capital_social": capital,
        "source": source,
    }
    for optional in (
        "nome_fantasia",
        "municipio",
        "codigo_cvm",
        "organisation_id",
        "segmento",
    ):
        value = row.get(optional)
        if value not in (None, "", NOT_INFORMED) and not pd.isna(value):
            metadata[optional] = clean_text(value)
    return Chunk(cnpj=cnpj, text=text, metadata=metadata, embed_text=embed_text)


def dataframe_to_chunks(df: pd.DataFrame) -> list[Chunk]:
    """Converte um DataFrame ja normalizado em chunks de texto."""
    if df is None or df.empty:
        return []
    chunks: list[Chunk] = []
    for row in df.to_dict(orient="records"):
        chunk = _row_to_chunk(row)
        if chunk is not None:
            chunks.append(chunk)
    return chunks


def dedupe_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    """Remove chunks com mesmo (cnpj, texto), preservando a ordem."""
    seen: set[tuple[str, str]] = set()
    unique: list[Chunk] = []
    for chunk in chunks:
        key = (chunk.cnpj, chunk.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def build_all_chunks(
    limit: int | None = None, sources: list[str] | None = None
) -> list[Chunk]:
    """Le as fontes disponiveis em data/raw e devolve os chunks.

    `limit=None` ou `limit<=0` processam todos os registros de cada fonte.
    `sources` restringe quais fontes processar (default: todas).
    """
    chunks: list[Chunk] = []
    loaders = (
        ("receita_federal", lambda: load_receita_federal(limit=limit)),
        ("cvm", lambda: load_cvm(limit=limit)),
        ("bacen", load_bacen),
    )
    if sources:
        loaders = tuple(item for item in loaders if item[0] in set(sources))
    for name, loader in loaders:
        try:
            df = loader()
        except (OSError, ValueError, KeyError) as exc:
            logger.error("falha ao processar %s: %s", name, exc)
            continue
        produced = dataframe_to_chunks(df)
        logger.info("%s: %d chunks", name, len(produced))
        chunks.extend(produced)

    unique = dedupe_chunks(chunks)
    logger.info("total: %d chunks (%d duplicados removidos)", len(unique), len(chunks) - len(unique))
    return unique


def save_chunks(chunks: list[Chunk], path: Path | None = None) -> Path:
    """Persiste os chunks em JSONL dentro de data/processed."""
    out = path or (PROCESSED_DIR / "chunks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    logger.info("chunks salvos em %s (%d linhas)", out, len(chunks))
    return out


def load_chunks(path: Path | None = None) -> Iterator[Chunk]:
    """Le de volta o JSONL gerado por `save_chunks`."""
    src = path or (PROCESSED_DIR / "chunks.jsonl")
    if not src.exists():
        return
    with src.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield Chunk(
                cnpj=data["cnpj"],
                text=data["text"],
                metadata=data.get("metadata", {}),
                embed_text=data.get("embed_text", ""),
            )


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Limpa os dados e gera os chunks")
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.demo_row_limit,
        help="max de linhas por fonte (0 = sem teto)",
    )
    args = parser.parse_args()

    produced = build_all_chunks(limit=args.limit)
    save_chunks(produced)
    for sample in produced[:3]:
        logger.info("exemplo: %s", sample.text)
