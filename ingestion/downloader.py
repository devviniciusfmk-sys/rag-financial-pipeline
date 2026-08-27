"""Download dos datasets publicos brasileiros usados pelo pipeline.

Fontes:
  * Receita Federal - Empresas0.zip (modo demo: apenas 1 arquivo do lote)
  * CVM            - cadastro de companhias abertas (CSV)
  * Bacen / OPF    - diretorio de participantes do Open Finance (JSON)

Caracteristicas: barra de progresso (tqdm), retry exponencial (tenacity),
download atomico (.part -> arquivo final) e skip quando o arquivo ja existe.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import requests
from requests.exceptions import ChunkedEncodingError
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import HTTPError, Timeout
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

from config import RAW_DIR, setup_logging

logger = setup_logging()

USER_AGENT = "Mozilla/5.0 (compatible; rag-financial-pipeline/1.0)"
CHUNK_SIZE = 1024 * 256  # 256 KiB
CONNECT_TIMEOUT = 15
READ_TIMEOUT = 120

RETRYABLE = (ReqConnectionError, Timeout, ChunkedEncodingError, HTTPError, OSError)

# A Receita nao serve mais os zips por caminho HTTP direto: o dado aberto do
# CNPJ vive num share publico Nextcloud (SERPRO), acessivel por WebDAV usando
# o token do share como usuario do basic auth e senha vazia.
RFB_WEBDAV = "https://arquivos.receitafederal.gov.br/public.php/webdav"
RFB_SHARE_TOKEN = "YggdBLfdninEJX9"
RFB_FALLBACK_FOLDER = "2026-08"  # usado se a listagem do share falhar
_FOLDER_RE = re.compile(r"(\d{4}-\d{2})")
_rfb_folder_cache: str | None = None


@dataclass(frozen=True)
class Source:
    """Descreve um dataset remoto."""

    key: str
    url: str
    filename: str
    kind: str  # "zip" | "csv" | "json"
    description: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    auth: tuple[str, str] | None = None
    # Quando True, a URL e montada com a pasta mensal mais recente do share.
    rfb_share: bool = False

    @property
    def path(self) -> Path:
        return RAW_DIR / self.filename


SOURCES: dict[str, Source] = {
    "receita_federal": Source(
        key="receita_federal",
        url="",  # resolvida em runtime: <share>/<AAAA-MM>/Empresas0.zip
        filename="Empresas0.zip",
        kind="zip",
        description="Receita Federal - cadastro de empresas (fatia 0, modo demo)",
        auth=(RFB_SHARE_TOKEN, ""),
        rfb_share=True,
    ),
    "receita_federal_estabelecimentos": Source(
        key="receita_federal_estabelecimentos",
        url="",  # resolvida em runtime: <share>/<AAAA-MM>/Estabelecimentos0.zip
        filename="Estabelecimentos0.zip",
        kind="zip",
        auth=(RFB_SHARE_TOKEN, ""),
        rfb_share=True,
        description=(
            "Receita Federal - estabelecimentos (fatia 0): traz UF, CNAE e "
            "situacao cadastral, ausentes no arquivo de empresas"
        ),
    ),
    "cvm": Source(
        key="cvm",
        url="https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv",
        filename="cad_cia_aberta.csv",
        kind="csv",
        description="CVM - cadastro de companhias abertas",
    ),
    "bacen": Source(
        key="bacen",
        url="https://data.directory.openbankingbrasil.org.br/participants",
        filename="open_finance_participants.json",
        kind="json",
        description="Open Finance Brasil - diretorio de participantes",
        headers={"Accept": "application/json"},
    ),
}


class DownloadError(RuntimeError):
    """Falha definitiva no download de uma fonte."""


def _session(
    extra_headers: dict[str, str] | None = None,
    auth: tuple[str, str] | None = None,
) -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    if extra_headers:
        sess.headers.update(extra_headers)
    if auth:
        sess.auth = auth
    return sess


def parse_rfb_folders(webdav_xml: str) -> list[str]:
    """Extrai as pastas AAAA-MM de uma resposta PROPFIND do share da RFB."""
    return sorted({m.group(1) for m in _FOLDER_RE.finditer(webdav_xml)})


def latest_rfb_folder() -> str:
    """Descobre a competencia mais recente publicada no share da Receita.

    Evita que a URL apodreca a cada mes. Se a listagem falhar, cai no valor
    fixo de `RFB_FALLBACK_FOLDER`.
    """
    global _rfb_folder_cache
    if _rfb_folder_cache:
        return _rfb_folder_cache

    try:
        with _session(auth=(RFB_SHARE_TOKEN, "")) as sess:
            resp = sess.request(
                "PROPFIND",
                f"{RFB_WEBDAV}/",
                headers={"Depth": "1"},
                timeout=(CONNECT_TIMEOUT, 60),
            )
            resp.raise_for_status()
        folders = parse_rfb_folders(resp.text)
        if folders:
            _rfb_folder_cache = folders[-1]
            logger.info("receita federal: competencia mais recente = %s", _rfb_folder_cache)
            return _rfb_folder_cache
    except (requests.RequestException, ValueError) as exc:
        logger.warning("nao foi possivel listar o share da RFB (%s)", exc)

    _rfb_folder_cache = RFB_FALLBACK_FOLDER
    logger.warning("usando competencia fixa %s", _rfb_folder_cache)
    return _rfb_folder_cache


def resolve_url(source: Source) -> str:
    """URL final da fonte (as da RFB dependem da competencia publicada)."""
    if source.rfb_share:
        return f"{RFB_WEBDAV}/{latest_rfb_folder()}/{source.filename}"
    return source.url


def _is_valid(source: Source, path: Path) -> bool:
    """Valida superficialmente um arquivo ja presente em disco."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        if source.kind == "zip":
            with zipfile.ZipFile(path) as zf:
                return bool(zf.namelist())
        if source.kind == "json":
            with path.open("r", encoding="utf-8") as fh:
                json.load(fh)
            return True
    except (zipfile.BadZipFile, json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    return True


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    retry=retry_if_exception_type(RETRYABLE),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _download_stream(source: Source, destination: Path) -> Path:
    """Baixa a fonte para `destination` com barra de progresso."""
    tmp = destination.with_suffix(destination.suffix + ".part")
    tmp.unlink(missing_ok=True)

    url = resolve_url(source)
    written = 0
    total = 0
    with _session(source.headers, auth=source.auth) as sess:
        with sess.get(
            url,
            stream=True,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            allow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0)
            bar = tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"{source.key:<16}",
                leave=True,
            )
            try:
                with tmp.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        bar.update(len(chunk))
            finally:
                bar.close()

    if written == 0:
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"{source.key}: resposta vazia de {url}")

    if total and written != total:
        tmp.unlink(missing_ok=True)
        raise ChunkedEncodingError(
            f"{source.key}: download incompleto ({written}/{total} bytes)"
        )

    tmp.replace(destination)
    return destination


def download_source(source: Source, force: bool = False) -> Path:
    """Baixa uma fonte, pulando quando ja existe um arquivo valido em disco."""
    dest = source.path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if not force and _is_valid(source, dest):
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("skip %s: ja existe (%.1f MB) em %s", source.key, size_mb, dest)
        return dest

    logger.info("baixando %s -> %s", resolve_url(source), dest)
    try:
        path = _download_stream(source, dest)
    except Exception as exc:  # noqa: BLE001 - convertido em erro de dominio
        raise DownloadError(
            f"falha ao baixar {source.key} ({resolve_url(source)}): {exc}"
        ) from exc

    if not _is_valid(source, path):
        path.unlink(missing_ok=True)
        raise DownloadError(f"{source.key}: arquivo baixado invalido ({source.kind})")

    logger.info("ok %s (%.1f MB)", source.key, path.stat().st_size / (1024 * 1024))
    return path


def download_receita_federal(force: bool = False) -> Path:
    """Receita Federal - apenas Empresas0.zip (modo demo)."""
    return download_source(SOURCES["receita_federal"], force=force)


def download_receita_federal_estabelecimentos(force: bool = False) -> Path:
    """Receita Federal - Estabelecimentos0.zip (UF, CNAE e situacao cadastral)."""
    return download_source(SOURCES["receita_federal_estabelecimentos"], force=force)


def download_cvm(force: bool = False) -> Path:
    """CVM - cadastro de companhias abertas."""
    return download_source(SOURCES["cvm"], force=force)


def download_bacen_participants(force: bool = False) -> Path:
    """Open Finance Brasil - diretorio de participantes (JSON)."""
    return download_source(SOURCES["bacen"], force=force)


def extract_zip(zip_path: Path, target_dir: Path | None = None) -> list[Path]:
    """Extrai um zip para `target_dir` (default: <raw>/<nome-do-zip>/)."""
    target = target_dir or (zip_path.parent / zip_path.stem)
    target.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        for member in tqdm(members, desc=f"unzip {zip_path.name}", unit="file"):
            safe_name = Path(member.filename).name  # protecao contra zip-slip
            out_path = target / safe_name
            if out_path.exists() and out_path.stat().st_size == member.file_size:
                extracted.append(out_path)
                continue
            with zf.open(member) as src, out_path.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=CHUNK_SIZE)
            extracted.append(out_path)
    return extracted


def download_all(force: bool = False, keys: list[str] | None = None) -> dict[str, Path]:
    """Baixa todas as fontes. Falha em uma fonte nao aborta as demais."""
    selected = keys or list(SOURCES)
    results: dict[str, Path] = {}
    failures: dict[str, str] = {}

    for key in selected:
        source = SOURCES.get(key)
        if source is None:
            logger.warning("fonte desconhecida ignorada: %s", key)
            continue
        try:
            results[key] = download_source(source, force=force)
        except DownloadError as exc:
            failures[key] = str(exc)
            logger.error("%s", exc)

    if failures:
        logger.warning(
            "concluido com %d falha(s): %s", len(failures), ", ".join(failures)
        )
    logger.info("downloads disponiveis: %s", ", ".join(results) or "nenhum")
    return results


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(
        description="Baixa os datasets publicos do pipeline"
    )
    parser.add_argument(
        "--only", nargs="*", choices=sorted(SOURCES), help="baixa apenas estas fontes"
    )
    parser.add_argument("--force", action="store_true", help="rebaixa mesmo se existir")
    parser.add_argument("--extract", action="store_true", help="extrai os zips baixados")
    args = parser.parse_args()

    paths = download_all(force=args.force, keys=args.only)
    if args.extract:
        for name, downloaded in paths.items():
            if downloaded.suffix.lower() == ".zip":
                files = extract_zip(downloaded)
                logger.info("%s: %d arquivo(s) extraido(s)", name, len(files))
