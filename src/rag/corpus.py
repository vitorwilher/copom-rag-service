"""Carregamento do corpus — `data/corpus.jsonl` → `RetrievedChunk`s.

Fonte única da verdade em runtime: o JSONL versionado gerado por
`scripts/ingest.py`. O pipeline e a API carregam o corpus daqui uma vez e
indexam no `HybridRetriever`.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.retriever import RetrievedChunk

DEFAULT_CORPUS = Path(__file__).resolve().parent.parent.parent / "data" / "corpus.jsonl"


def load_corpus(path: Path | None = None) -> list[RetrievedChunk]:
    """Lê o corpus JSONL e o converte em chunks.

    Args:
        path: caminho do JSONL (default: `data/corpus.jsonl`).

    Returns:
        Lista de `RetrievedChunk` (sem score; será atribuído na recuperação).

    Raises:
        FileNotFoundError: se o corpus não existir (rode `scripts/ingest.py`).
    """
    path = path or DEFAULT_CORPUS
    if not path.exists():
        raise FileNotFoundError(
            f"corpus não encontrado em {path} — rode `python scripts/ingest.py`"
        )
    chunks: list[RetrievedChunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            chunks.append(
                RetrievedChunk(
                    text=row["text"],
                    source=row["source"],
                    metadata=row.get("metadata", {}),
                )
            )
    return chunks
