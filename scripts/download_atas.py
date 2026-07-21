"""Download das atas do Copom — API do BCB → `data/atas_full_cache.json`.

Baixa o **texto completo** de cada ata publicada pelo Banco Central (seções A,
B e C — conjuntura, análise e a decisão/votos, onde ficam os números de juros),
diferente do cache do projeto irmão Sentimento_COPOM, que trunca em ~4500 chars
só a seção A. É esse texto completo que alimenta o RAG.

Endpoints públicos do BCB (mesma fonte usada pelo Sentimento_COPOM)::

    lista   → https://www.bcb.gov.br/api/servico/sitebcb/copom/atas
    detalhe → https://www.bcb.gov.br/api/servico/sitebcb/copom/atas_detalhes

Uso::

    python scripts/download_atas.py               # todas as atas publicadas
    python scripts/download_atas.py --desde 232   # só reuniões >= 232
    python scripts/download_atas.py --limite 5    # só as 5 mais recentes (teste)

Grava um dict `{nro_reuniao: {data, texto}}` — mesmo formato do cache do
Sentimento_COPOM, para que `scripts/ingest.py` consuma qualquer um dos dois.
Idempotente: relê o cache existente e só baixa o que falta (a menos de --forcar).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BCB_LISTA = "https://www.bcb.gov.br/api/servico/sitebcb/copom/atas"
BCB_DETALHE = "https://www.bcb.gov.br/api/servico/sitebcb/copom/atas_detalhes"
HEADERS = {"Accept": "application/json", "User-Agent": "CopomRAGService/1.0"}

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "atas_full_cache.json"


def html_para_texto(html: str) -> str:
    """Converte o HTML da ata em texto limpo (sem notas, scripts, estilos).

    Args:
        html: conteúdo `textoAta` da resposta do BCB.

    Returns:
        Texto normalizado (espaços colapsados), sem truncar.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["sup", "script", "style"]):
        tag.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True))


def listar_reunioes() -> list[int]:
    """Lista os números de todas as reuniões com ata publicada.

    Returns:
        Lista ordenada de `nroReuniao`.
    """
    resp = requests.get(
        BCB_LISTA, params={"quantidade": 1000}, headers=HEADERS, timeout=30
    )
    resp.raise_for_status()
    return sorted(item["nroReuniao"] for item in resp.json()["conteudo"])


def baixar_ata(nro: int, tentativas: int = 3) -> dict | None:
    """Baixa e limpa o texto completo de uma ata, com retry exponencial.

    Args:
        nro: número da reunião.
        tentativas: número de tentativas antes de desistir.

    Returns:
        `{data, texto}` ou `None` se falhar após as tentativas.
    """
    for i in range(tentativas):
        try:
            resp = requests.get(
                BCB_DETALHE, params={"nro_reuniao": nro}, headers=HEADERS, timeout=30
            )
            resp.raise_for_status()
            item = resp.json()["conteudo"][0]
            texto = html_para_texto(item.get("textoAta", ""))
            if not texto:
                return None
            return {"data": item["dataReferencia"][:10], "texto": texto}
        except Exception as exc:  # noqa: BLE001 — rede: tenta de novo ou desiste
            if i < tentativas - 1:
                time.sleep(2**i)
            else:
                print(f"  [ERRO] ata {nro}: {exc}")
                return None


def main() -> None:
    """Ponto de entrada CLI: baixa as atas completas para o cache JSON."""
    parser = argparse.ArgumentParser(description="Download das atas completas do Copom.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"cache de saída (default: {DEFAULT_OUT}).")
    parser.add_argument("--desde", type=int, default=None,
                        help="baixar só reuniões com número >= este.")
    parser.add_argument("--limite", type=int, default=None,
                        help="baixar só as N reuniões mais recentes (teste).")
    parser.add_argument("--forcar", action="store_true",
                        help="rebaixar mesmo as atas já presentes no cache.")
    args = parser.parse_args()

    cache: dict = {}
    if args.out.exists() and not args.forcar:
        cache = json.loads(args.out.read_text(encoding="utf-8"))

    reunioes = listar_reunioes()
    if args.desde is not None:
        reunioes = [n for n in reunioes if n >= args.desde]
    if args.limite is not None:
        reunioes = reunioes[-args.limite:]

    pendentes = [n for n in reunioes if str(n) not in cache]
    print(f"{len(reunioes)} reuniões alvo; {len(pendentes)} a baixar "
          f"(cache já tem {len(cache)}).")

    baixadas = 0
    for nro in pendentes:
        doc = baixar_ata(nro)
        if doc:
            cache[str(nro)] = doc
            baixadas += 1
            print(f"  + ata {nro} ({doc['data']}) — {len(doc['texto'])} chars")
        time.sleep(0.3)  # cortesia com a API do BCB

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    print(f"cache gravado: {len(cache)} atas → {args.out} ({baixadas} novas).")


if __name__ == "__main__":
    main()
