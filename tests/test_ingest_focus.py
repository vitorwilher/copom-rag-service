"""Testes da ingestão do Focus — cache de expectativas → chunks descritivos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# scripts/ não é um pacote instalável; adiciona ao path para importar ingest.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ingest import build_focus_corpus  # noqa: E402


def test_build_focus_corpus_gera_chunk_por_reuniao(tmp_path):
    cache = {
        "279": {
            "data": "2026-06-17",
            "expectativas": {
                "Selic": {"2026": 14.0, "2027": 12.0},
                "IPCA": {"2026": 5.3},
            },
        },
        "278": {
            "data": "2026-04-29",
            "expectativas": {"Selic": {"2026": 13.0}},
        },
    }
    src = tmp_path / "focus_cache.json"
    src.write_text(json.dumps(cache), encoding="utf-8")

    rows = build_focus_corpus(src)

    assert len(rows) == 2
    # ordenado por número de reunião (int), então 278 antes de 279
    r278, r279 = rows
    assert r278["source"] == "focus_2026-04-29"
    assert r279["source"] == "focus_2026-06-17"
    # metadata de proveniência + marcador de tipo
    assert r279["metadata"]["meeting"] == "279"
    assert r279["metadata"]["kind"] == "focus"
    # texto descritivo em linguagem natural, com os números e a âncora da reunião
    txt = r279["text"]
    assert "279" in txt and "2026-06-17" in txt
    assert "14.0 ao fim de 2026" in txt
    assert "Focus" in txt


def test_build_focus_corpus_ignora_reuniao_sem_expectativas(tmp_path):
    cache = {"279": {"data": "2026-06-17", "expectativas": {}}}
    src = tmp_path / "focus_cache.json"
    src.write_text(json.dumps(cache), encoding="utf-8")
    assert build_focus_corpus(src) == []
