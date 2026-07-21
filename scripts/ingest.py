"""Ingestão do corpus — atas do Copom → `data/corpus.jsonl`.

Lê um cache de atas `{nro_reuniao: {data, texto}}`, chunka cada ata em pedaços
com sobreposição e grava um `corpus.jsonl` versionável e auto-contido neste
projeto. Cada linha do JSONL é um chunk com proveniência (`ata_NNN`, data da
reunião, índice do chunk).

Fonte default: `data/atas_full_cache.json` — atas **completas** (seções A, B e
C) baixadas por `scripts/download_atas.py`. Se ausente, cai para o cache
truncado do projeto irmão Sentimento_COPOM. O runtime (retriever/pipeline) lê
apenas o `corpus.jsonl` — não depende de nenhum dos caches em produção.

Uso (idempotente)::

    python scripts/download_atas.py   # gera data/atas_full_cache.json
    python scripts/ingest.py          # → data/corpus.jsonl
    python scripts/ingest.py --source /outro/cache.json --out data/corpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "data"

# Fonte default: atas completas baixadas do BCB (download_atas.py).
DEFAULT_SOURCE = _DATA / "atas_full_cache.json"
# Cache do Focus ancorado às reuniões (download_focus.py). Opcional: se existir,
# é ingerido junto das atas como chunks `focus_AAAA-MM-DD`.
FOCUS_SOURCE = _DATA / "focus_cache.json"
# Fallback: cache truncado do projeto irmão Sentimento_COPOM.
FALLBACK_SOURCE = (
    Path(__file__).resolve().parent.parent.parent
    / "Sentimento_COPOM"
    / "atas_cache.json"
)
DEFAULT_OUT = _DATA / "corpus.jsonl"

# Parâmetros de chunking (em caracteres). Atas do Copom são densas; ~1200 chars
# (~250 palavras) mantém um argumento inteiro por chunk, com 200 de overlap para
# não cortar números/frases no limite.
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fatia um texto em janelas com sobreposição, quebrando em espaços.

    Args:
        text: texto integral da ata.
        size: tamanho-alvo de cada chunk (caracteres).
        overlap: sobreposição entre chunks consecutivos (caracteres).

    Returns:
        Lista de chunks não vazios.
    """
    text = " ".join(text.split())  # normaliza espaços/quebras
    if len(text) <= size:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # recua até o último espaço para não cortar palavra/número
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_corpus(source: Path) -> list[dict]:
    """Lê o cache de atas e produz a lista de chunks com metadados.

    Args:
        source: caminho para `atas_cache.json` (dict reunião → {data, texto}).

    Returns:
        Lista de dicts `{text, source, metadata}` prontos para o JSONL.
    """
    raw = json.loads(source.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for meeting in sorted(raw, key=int):
        ata = raw[meeting]
        data = ata.get("data", "")
        texto = ata.get("texto", "")
        for i, chunk in enumerate(chunk_text(texto)):
            rows.append(
                {
                    "text": chunk,
                    "source": f"ata_{meeting}",
                    "metadata": {"meeting": meeting, "data": data, "chunk": i},
                }
            )
    return rows


_INDICADOR_LABEL = {
    "Selic": "a taxa Selic",
    "IPCA": "o IPCA (inflação)",
    "PIB Total": "o crescimento do PIB",
    "Câmbio": "a taxa de câmbio (R$/US$)",
}


def build_focus_corpus(source: Path) -> list[dict]:
    """Lê o cache do Focus e produz chunks descritivos por reunião.

    Cada reunião vira um chunk `focus_AAAA-MM-DD` em linguagem natural — "na
    véspera da reunião X, o mercado (Focus) esperava Selic de Y% ao fim de
    AAAA..." — para que o retrieval case perguntas sobre expectativas de mercado
    ancoradas à data/reunião. O snapshot é o vigente quando o Copom decidiu.

    Args:
        source: caminho para `focus_cache.json` (dict reunião → {data, expectativas}).

    Returns:
        Lista de dicts `{text, source, metadata}` prontos para o JSONL.
    """
    raw = json.loads(source.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for meeting in sorted(raw, key=int):
        rec = raw[meeting]
        data = rec.get("data", "")
        exp = rec.get("expectativas", {})
        if not exp:
            continue
        partes: list[str] = []
        for ind, por_ano in exp.items():
            label = _INDICADOR_LABEL.get(ind, ind)
            anos = "; ".join(f"{v} ao fim de {ano}" for ano, v in por_ano.items())
            partes.append(f"{label}: {anos}")
        texto = (
            f"Expectativas do mercado (boletim Focus do Banco Central) vigentes na "
            f"véspera da reunião {meeting} do Copom, em {data}. Medianas projetadas — "
            + ". ".join(partes) + "."
        )
        rows.append(
            {
                "text": texto,
                "source": f"focus_{data}",
                "metadata": {"meeting": meeting, "data": data, "chunk": 0, "kind": "focus"},
            }
        )
    return rows


def main() -> None:
    """Ponto de entrada CLI: lê o cache, chunka e grava o corpus JSONL."""
    parser = argparse.ArgumentParser(description="Ingestão de atas → corpus.jsonl")
    parser.add_argument("--source", type=Path, default=None,
                        help="atas_cache.json de origem "
                             "(default: data/atas_full_cache.json, senão o cache do Sentimento_COPOM).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"corpus JSONL de saída (default: {DEFAULT_OUT}).")
    args = parser.parse_args()

    source = args.source
    if source is None:
        if DEFAULT_SOURCE.exists():
            source = DEFAULT_SOURCE
        elif FALLBACK_SOURCE.exists():
            source = FALLBACK_SOURCE
            print(f"[aviso] usando cache truncado do Sentimento_COPOM — "
                  f"rode scripts/download_atas.py para as atas completas.")
        else:
            raise SystemExit(
                "nenhuma fonte encontrada — rode `python scripts/download_atas.py`"
            )
    elif not source.exists():
        raise SystemExit(f"fonte não encontrada: {source}")

    rows = build_corpus(source)
    n_atas_chunks = len(rows)

    # Focus (opcional): ingere junto se o cache existir (download_focus.py).
    n_focus = 0
    if FOCUS_SOURCE.exists():
        focus_rows = build_focus_corpus(FOCUS_SOURCE)
        rows.extend(focus_rows)
        n_focus = len(focus_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    atas = len({r["source"] for r in rows if not r["source"].startswith("focus_")})
    print(f"corpus gerado: {len(rows)} chunks "
          f"({n_atas_chunks} de {atas} atas + {n_focus} do Focus) → {args.out}")


if __name__ == "__main__":
    main()
