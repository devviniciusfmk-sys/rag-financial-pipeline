"""Pipeline de ponta a ponta: download -> limpeza -> embeddings -> pgvector.

Uso:
    python -m scripts.build_index                 # tudo, com limite demo
    python -m scripts.build_index --limit 5000    # menos linhas da Receita
    python -m scripts.build_index --skip-download # reaproveita data/raw
    python -m scripts.build_index --reset         # limpa a tabela antes
    python -m scripts.build_index --full          # indexa tudo, sem teto
    python -m scripts.build_index --only bacen --skip-download   # so uma fonte
"""

from __future__ import annotations

import argparse
import time

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from config import settings, setup_logging
from embeddings.generator import EmbeddingGenerator
from embeddings.store import VectorStore
from ingestion.cleaner import Chunk, build_all_chunks, save_chunks
from ingestion.downloader import download_all

logger = setup_logging()
console = Console()

DB_BATCH = 500


def index_chunks(
    chunks: list[Chunk], store: VectorStore, generator: EmbeddingGenerator
) -> int:
    """Vetoriza e grava os chunks em lotes."""
    if not chunks:
        logger.warning("nenhum chunk para indexar")
        return 0

    total = 0
    batches = range(0, len(chunks), DB_BATCH)
    for start in tqdm(list(batches), desc="indexando", unit="lote"):
        batch = chunks[start : start + DB_BATCH]
        # embeda o texto enxuto; o template completo continua sendo o exibido
        vectors = generator.generate(
            [c.embed_text for c in batch], show_progress=False
        )
        records = [
            (chunk.cnpj, chunk.text, vectors[i], chunk.metadata)
            for i, chunk in enumerate(batch)
        ]
        total += store.save_many(records)
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description="Constroi o indice vetorial")
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.demo_row_limit,
        help="max de linhas por fonte (0 = sem teto)",
    )
    parser.add_argument(
        "--full", action="store_true", help="Index all records without limit"
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--reset", action="store_true", help="trunca a tabela antes")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["receita_federal", "cvm", "bacen"],
        help="reprocessa apenas estas fontes, substituindo-as no banco",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    limit = None if args.full else args.limit
    logger.info(
        "modo: %s", "completo (sem teto)" if not limit else f"demo ({limit} linhas/fonte)"
    )

    if not args.skip_download:
        console.rule("[bold cyan]1/4 download")
        download_all(force=args.force_download)
    else:
        logger.info("download pulado (--skip-download)")

    console.rule("[bold cyan]2/4 limpeza e chunking")
    chunks = build_all_chunks(limit=limit, sources=args.only)
    if not chunks:
        logger.error("nenhum chunk gerado - verifique os arquivos em data/raw")
        return 1
    save_chunks(chunks)

    console.rule("[bold cyan]3/4 embeddings + pgvector")
    generator = EmbeddingGenerator()
    store = VectorStore(dimension=generator.dimension)
    store.ensure_schema()
    if args.reset:
        store.truncate()
    elif args.only:
        # Substituicao cirurgica: remove so as fontes pedidas, preserva o resto.
        for source in args.only:
            removed = store.delete_by_source(source)
            logger.info("removidos %d registros antigos de %s", removed, source)
    indexed = index_chunks(chunks, store, generator)

    store.rebuild_vector_index()

    console.rule("[bold cyan]4/4 resumo")
    stats = store.stats()
    table = Table(show_header=False, box=None)
    table.add_row("chunks processados", str(indexed))
    table.add_row("embeddings na base", str(stats["total_embeddings"]))
    table.add_row("empresas distintas", str(stats["total_companies"]))
    table.add_row("fontes", ", ".join(stats["sources"]) or "-")
    table.add_row("tempo total", f"{time.perf_counter() - started:.1f}s")
    console.print(table)

    store.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
