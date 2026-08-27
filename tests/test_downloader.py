"""Testes de `ingestion.downloader` — nenhum acesso a rede."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from ingestion import downloader as D

PROPFIND_XML = """<?xml version="1.0"?>
<d:multistatus xmlns:d="DAV:">
  <d:response><d:href>/public.php/webdav/</d:href></d:response>
  <d:response><d:href>/public.php/webdav/2023-05/</d:href></d:response>
  <d:response><d:href>/public.php/webdav/2026-07/</d:href></d:response>
  <d:response><d:href>/public.php/webdav/2026-08/</d:href></d:response>
  <d:response><d:href>/public.php/webdav/2024-11/</d:href></d:response>
</d:multistatus>"""


# --------------------------------------------------------------------------- #
# Descoberta da competencia publicada pela Receita
# --------------------------------------------------------------------------- #
def test_parse_rfb_folders_ordena_e_deduplica() -> None:
    folders = D.parse_rfb_folders(PROPFIND_XML)
    assert folders == ["2023-05", "2024-11", "2026-07", "2026-08"]
    assert folders[-1] == "2026-08"


def test_parse_rfb_folders_sem_pastas() -> None:
    assert D.parse_rfb_folders("<d:multistatus></d:multistatus>") == []


def test_latest_rfb_folder_usa_fallback_quando_listagem_falha(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    monkeypatch.setattr(D, "_rfb_folder_cache", None)

    def explode(*args, **kwargs):
        raise requests.ConnectionError("share fora do ar")

    monkeypatch.setattr(D.requests.Session, "request", explode)
    assert D.latest_rfb_folder() == D.RFB_FALLBACK_FOLDER


# --------------------------------------------------------------------------- #
# Montagem das URLs
# --------------------------------------------------------------------------- #
def test_resolve_url_da_receita_usa_o_share_webdav(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(D, "_rfb_folder_cache", "2026-08")

    url = D.resolve_url(D.SOURCES["receita_federal"])
    assert url == f"{D.RFB_WEBDAV}/2026-08/Empresas0.zip"

    url_estab = D.resolve_url(D.SOURCES["receita_federal_estabelecimentos"])
    assert url_estab == f"{D.RFB_WEBDAV}/2026-08/Estabelecimentos0.zip"


def test_resolve_url_das_demais_fontes_e_estatica() -> None:
    assert D.resolve_url(D.SOURCES["cvm"]).startswith("https://dados.cvm.gov.br/")
    assert D.resolve_url(D.SOURCES["bacen"]).endswith("/participants")


def test_fontes_da_receita_levam_o_token_do_share() -> None:
    for key in ("receita_federal", "receita_federal_estabelecimentos"):
        source = D.SOURCES[key]
        assert source.rfb_share is True
        assert source.auth == (D.RFB_SHARE_TOKEN, "")


# --------------------------------------------------------------------------- #
# Validacao de arquivos ja baixados (logica do "skip se ja existe")
# --------------------------------------------------------------------------- #
def test_is_valid_rejeita_zip_corrompido(tmp_path: Path) -> None:
    corrompido = tmp_path / "Empresas0.zip"
    corrompido.write_bytes(b"isto nao e um zip")
    assert D._is_valid(D.SOURCES["receita_federal"], corrompido) is False


def test_is_valid_aceita_zip_integro(tmp_path: Path) -> None:
    valido = tmp_path / "Empresas0.zip"
    with zipfile.ZipFile(valido, "w") as zf:
        zf.writestr("EMPRECSV", b"conteudo")
    assert D._is_valid(D.SOURCES["receita_federal"], valido) is True


def test_is_valid_rejeita_json_invalido(tmp_path: Path) -> None:
    quebrado = tmp_path / "participants.json"
    quebrado.write_text("{nao e json", encoding="utf-8")
    assert D._is_valid(D.SOURCES["bacen"], quebrado) is False


def test_is_valid_aceita_json_valido(tmp_path: Path) -> None:
    ok = tmp_path / "participants.json"
    ok.write_text(json.dumps([{"OrganisationName": "x"}]), encoding="utf-8")
    assert D._is_valid(D.SOURCES["bacen"], ok) is True


def test_is_valid_rejeita_arquivo_vazio(tmp_path: Path) -> None:
    vazio = tmp_path / "cad_cia_aberta.csv"
    vazio.touch()
    assert D._is_valid(D.SOURCES["cvm"], vazio) is False


# --------------------------------------------------------------------------- #
# Extracao
# --------------------------------------------------------------------------- #
def test_extract_zip_protege_contra_zip_slip(tmp_path: Path) -> None:
    malicioso = tmp_path / "payload.zip"
    with zipfile.ZipFile(malicioso, "w") as zf:
        zf.writestr("../../fora.txt", b"nao deveria escapar")

    destino = tmp_path / "saida"
    extraidos = D.extract_zip(malicioso, destino)

    assert extraidos == [destino / "fora.txt"]
    assert not (tmp_path.parent / "fora.txt").exists()
