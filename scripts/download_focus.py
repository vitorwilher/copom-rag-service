"""Download do boletim Focus — API de Expectativas do BCB → cache JSON.

Para cada reunião do Copom (datas lidas do cache de atas), busca as expectativas
de mercado do boletim **Focus** vigentes **na véspera** da reunião — ou seja, o
que o mercado projetava *quando o Copom decidiu*. Isso ancora o Focus às mesmas
datas das atas, no espírito de "fidelidade": a pergunta "o que o mercado esperava
na reunião 279?" passa a ter resposta no corpus.

Fonte (API Olinda do BCB, Expectativas de Mercado)::

    https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoAnuais

Captura, por reunião, a **mediana** das expectativas (baseCalculo=0, últimos 30
dias) para um conjunto de indicadores-chave, nos anos de referência do ano da
reunião e do seguinte.

IMPORTANTE — encoding: o OData do BCB rejeita as aspas simples percent-encoded
que o `requests` gera por padrão (retorna 400 "Boolean/String"). Montamos a query
manualmente com `urllib.parse.quote(filtro, safe="'")` para preservar as aspas.

Uso::

    python scripts/download_focus.py               # todas as reuniões do cache
    python scripts/download_focus.py --limite 5    # só as 5 reuniões mais recentes
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
from datetime import date, timedelta
from pathlib import Path

import requests

_DATA = Path(__file__).resolve().parent.parent / "data"
ATAS_CACHE = _DATA / "atas_full_cache.json"
DEFAULT_OUT = _DATA / "focus_cache.json"

FOCUS_URL = (
    "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/"
    "ExpectativasMercadoAnuais"
)
HEADERS = {"Accept": "application/json", "User-Agent": "CopomRAGService/1.0"}

# Indicadores-chave do Focus (nomes exatos na base Anuais do BCB —
# atenção: câmbio é "Câmbio", não "Taxa de câmbio", nesta base).
INDICADORES = ["Selic", "IPCA", "PIB Total", "Câmbio"]
# Janela (dias) antes da reunião em que procuramos o Focus mais recente.
JANELA_DIAS = 12


def _get(filtro: str, top: int = 60) -> list[dict]:
    """Consulta a API do Focus com um filtro OData, contornando o encoding.

    Args:
        filtro: expressão `$filter` OData (com aspas simples literais).
        top: máximo de linhas a retornar.

    Returns:
        Lista de registros (`value`), ou lista vazia em falha.
    """
    query = (
        f"$top={top}&$format=json"
        f"&$orderby={urllib.parse.quote('Data desc', safe='')}"
        f"&$filter={urllib.parse.quote(filtro, safe=chr(39))}"
    )
    try:
        resp = requests.get(f"{FOCUS_URL}?{query}", headers=HEADERS, timeout=30)
        resp.raise_for_status()
        return resp.json().get("value", [])
    except Exception as exc:  # noqa: BLE001 — rede/API: registra e segue
        print(f"  [ERRO] Focus: {exc}")
        return []


def expectativas_para(data_reuniao: str) -> dict:
    """Coleta a mediana das expectativas Focus vigentes na véspera da reunião.

    Para cada indicador, pega o registro mais recente (Focus é diário) dentro da
    janela `[data_reuniao - JANELA_DIAS, data_reuniao]`, para os anos de
    referência do ano da reunião e do seguinte.

    Args:
        data_reuniao: data da reunião em ISO ('AAAA-MM-DD').

    Returns:
        Dict `{indicador: {ano_ref: mediana}}` com o que havia disponível.
    """
    fim = date.fromisoformat(data_reuniao)
    inicio = fim - timedelta(days=JANELA_DIAS)
    ano = fim.year
    anos_ref = {str(ano), str(ano + 1)}

    out: dict[str, dict] = {}
    for ind in INDICADORES:
        filtro = (
            f"Indicador eq '{ind}' and baseCalculo eq 0 "
            f"and Data ge '{inicio.isoformat()}' and Data le '{fim.isoformat()}'"
        )
        linhas = _get(filtro)
        # Para cada ano de referência, fica com a Data mais recente (linhas já
        # vêm ordenadas por Data desc — a primeira ocorrência é a mais nova).
        por_ano: dict[str, float] = {}
        for row in linhas:
            ref = str(row.get("DataReferencia", ""))
            if ref in anos_ref and ref not in por_ano and row.get("Mediana") is not None:
                por_ano[ref] = float(row["Mediana"])
        if por_ano:
            out[ind] = dict(sorted(por_ano.items()))
        time.sleep(0.2)  # cortesia com a API
    return out


def main() -> None:
    """Ponto de entrada CLI: baixa o Focus ancorado às datas das reuniões."""
    parser = argparse.ArgumentParser(description="Download do Focus ancorado às reuniões do Copom.")
    parser.add_argument("--atas", type=Path, default=ATAS_CACHE,
                        help=f"cache de atas para as datas (default: {ATAS_CACHE}).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"cache de saída do Focus (default: {DEFAULT_OUT}).")
    parser.add_argument("--limite", type=int, default=None,
                        help="baixar só as N reuniões mais recentes (teste).")
    parser.add_argument("--forcar", action="store_true",
                        help="rebaixar mesmo reuniões já no cache.")
    args = parser.parse_args()

    if not args.atas.exists():
        raise SystemExit(f"cache de atas não encontrado: {args.atas} — rode download_atas.py")
    atas = json.loads(args.atas.read_text(encoding="utf-8"))

    cache: dict = {}
    if args.out.exists() and not args.forcar:
        cache = json.loads(args.out.read_text(encoding="utf-8"))

    # reuniões (nro, data) ordenadas por data
    reunioes = sorted(((nro, a["data"]) for nro, a in atas.items()), key=lambda x: x[1])
    if args.limite is not None:
        reunioes = reunioes[-args.limite:]

    pendentes = [(n, d) for n, d in reunioes if n not in cache]
    print(f"{len(reunioes)} reuniões alvo; {len(pendentes)} a baixar "
          f"(cache já tem {len(cache)}).")

    baixadas = 0
    for nro, data in pendentes:
        exp = expectativas_para(data)
        if exp:
            cache[nro] = {"data": data, "expectativas": exp}
            baixadas += 1
            resumo = ", ".join(f"{k}={v}" for k, v in exp.items())
            print(f"  + focus reunião {nro} ({data}) — {resumo}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"cache Focus gravado: {len(cache)} reuniões → {args.out} ({baixadas} novas).")


if __name__ == "__main__":
    main()
