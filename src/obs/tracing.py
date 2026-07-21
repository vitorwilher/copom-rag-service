"""Observability — tracing de custo, latência e tokens por request.

Este módulo oferece um *context manager* leve (`trace`) que instrumenta cada
chamada ao pipeline RAG registrando três dimensões operacionais críticas para
um serviço de LLM em produção:

- **latência** (tempo de parede em segundos);
- **tokens** (entrada + saída, quando o provedor os reporta);
- **custo** (estimado a partir do preço por 1M de tokens do modelo).

O objetivo é permitir, no futuro, exportar essas métricas para um backend de
observabilidade (OpenTelemetry, Prometheus, Langfuse, etc.). Por ora o stub
apenas agrega em memória e imprime um resumo estruturado.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


# Preços de referência (US$ por 1M de tokens). Ajustar conforme o modelo servido.
# TODO: mover para configuração / carregar do provedor via `claude-api` reference.
PRICE_TABLE: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 3.00, "output": 15.00},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
}


@dataclass
class TraceRecord:
    """Registro de uma única passagem instrumentada pelo pipeline.

    Attributes:
        name: rótulo do span (ex.: "ask", "retrieve", "generate").
        model: identificador do modelo LLM usado no span.
        latency_s: latência de parede em segundos.
        input_tokens: tokens de entrada consumidos.
        output_tokens: tokens de saída gerados.
        cost_usd: custo estimado em dólares.
        metadata: campos livres (pergunta, nº de documentos, etc.).
    """

    name: str
    model: str = "unknown"
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict = field(default_factory=dict)

    def compute_cost(self) -> float:
        """Estima o custo em US$ a partir da tabela de preços do modelo.

        Returns:
            Custo estimado em dólares (0.0 se o modelo não estiver tabelado).
        """
        prices = PRICE_TABLE.get(self.model)
        if prices is None:
            return 0.0
        self.cost_usd = (
            self.input_tokens / 1_000_000 * prices["input"]
            + self.output_tokens / 1_000_000 * prices["output"]
        )
        return self.cost_usd

    def to_dict(self) -> dict:
        """Serializa o registro num dict plano e estável (uma linha do JSONL).

        Formato pensado para ser consumido depois por um exportador OTel/
        Prometheus ou simplesmente agregado por `summarize_traces`.

        Returns:
            Dict com as dimensões operacionais do span + metadata livre.
        """
        return {
            "name": self.name,
            "model": self.model,
            "latency_s": round(self.latency_s, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "metadata": self.metadata,
        }


@contextmanager
def trace(name: str, model: str = "unknown", **metadata) -> Iterator[TraceRecord]:
    """Context manager que mede latência e agrega tokens/custo de um span.

    Uso::

        with trace("ask", model="claude-haiku-4-5", pergunta=q) as rec:
            resposta = pipeline.run(q)
            rec.input_tokens = resposta.usage.input_tokens
            rec.output_tokens = resposta.usage.output_tokens

    O chamador é responsável por preencher `input_tokens`/`output_tokens` a
    partir do objeto de *usage* retornado pelo SDK do provedor. Ao sair do
    bloco, a latência é fechada e o custo é calculado automaticamente.

    Args:
        name: rótulo do span.
        model: identificador do modelo (usado para precificar).
        **metadata: campos livres anexados ao registro.

    Yields:
        TraceRecord mutável a ser preenchido dentro do bloco `with`.
    """
    record = TraceRecord(name=name, model=model, metadata=dict(metadata))
    start = time.perf_counter()
    try:
        yield record
    finally:
        record.latency_s = time.perf_counter() - start
        record.compute_cost()
        # TODO: emitir para backend de observabilidade (OTel/Langfuse/Prometheus)
        #       em vez de apenas imprimir.
        _emit(record)


# Variável de ambiente que ativa o sink JSONL persistente. Vazia (default) →
# só imprime, sem tocar disco: mantém os testes e o eval offline sem efeitos
# colaterais em arquivos. Setada → cada span também é anexado ao arquivo.
TRACE_FILE_ENV = "COPOM_TRACE_FILE"


def _emit(record: TraceRecord) -> None:
    """Sink de métricas: imprime o span e, se configurado, persiste em JSONL.

    O print legível permanece (útil no run interativo do eval). Além dele,
    quando `COPOM_TRACE_FILE` aponta um caminho, o span é anexado como uma
    linha JSON — o formato que um exportador OTel/Prometheus consumiria e que
    `summarize_traces` agrega em custo/latência por sessão. Falha de escrita não
    derruba o request (observabilidade nunca deve quebrar o caminho principal).
    """
    print(
        f"[trace] {record.name:<10} model={record.model} "
        f"lat={record.latency_s:.3f}s "
        f"tok_in={record.input_tokens} tok_out={record.output_tokens} "
        f"cost=US${record.cost_usd:.6f}"
    )
    path = os.getenv(TRACE_FILE_ENV)
    if not path:
        return
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    except OSError:
        pass  # sink de observabilidade nunca deve derrubar o caminho principal


def summarize_traces(path: str | Path) -> dict:
    """Agrega um arquivo de traces JSONL em métricas de sessão.

    Lê o JSONL escrito por `_emit` e soma custo, latência e tokens no total e
    por modelo/span — a base de um painel de observabilidade (quanto custou a
    sessão, onde foi o tempo, quantas degradações ocorreram).

    Args:
        path: caminho do arquivo de traces JSONL.

    Returns:
        Dict com `spans`, totais (`cost_usd`, `latency_s`, tokens),
        `degraded` (nº de spans que caíram para fallback) e quebras
        `by_model` / `by_name`.
    """
    path = Path(path)
    total = {
        "spans": 0, "cost_usd": 0.0, "latency_s": 0.0,
        "input_tokens": 0, "output_tokens": 0, "degraded": 0,
    }
    by_model: dict[str, dict] = {}
    by_name: dict[str, dict] = {}
    if not path.exists():
        return {**total, "by_model": by_model, "by_name": by_name}

    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total["spans"] += 1
            total["cost_usd"] += rec.get("cost_usd", 0.0)
            total["latency_s"] += rec.get("latency_s", 0.0)
            total["input_tokens"] += rec.get("input_tokens", 0)
            total["output_tokens"] += rec.get("output_tokens", 0)
            if (rec.get("metadata") or {}).get("degraded"):
                total["degraded"] += 1
            for key, bucket in ((rec.get("model", "unknown"), by_model),
                                (rec.get("name", "unknown"), by_name)):
                b = bucket.setdefault(key, {"spans": 0, "cost_usd": 0.0, "latency_s": 0.0})
                b["spans"] += 1
                b["cost_usd"] = round(b["cost_usd"] + rec.get("cost_usd", 0.0), 6)
                b["latency_s"] = round(b["latency_s"] + rec.get("latency_s", 0.0), 6)

    total["cost_usd"] = round(total["cost_usd"], 6)
    total["latency_s"] = round(total["latency_s"], 6)
    return {**total, "by_model": by_model, "by_name": by_name}


def _print_summary(summary: dict) -> None:
    """Imprime o resumo de traces em formato legível (CLI)."""
    print("\n=== COPOM RAG Service — Resumo de Traces ===\n")
    print(f"spans={summary['spans']}  "
          f"custo=US${summary['cost_usd']:.6f}  "
          f"latência={summary['latency_s']:.3f}s  "
          f"tok_in={summary['input_tokens']}  tok_out={summary['output_tokens']}  "
          f"degradações={summary['degraded']}")
    if summary["by_model"]:
        print("\npor modelo:")
        for model, b in summary["by_model"].items():
            print(f"  {model:<22} spans={b['spans']:<3} "
                  f"custo=US${b['cost_usd']:.6f}  lat={b['latency_s']:.3f}s")
    if summary["by_name"]:
        print("\npor span:")
        for name, b in summary["by_name"].items():
            print(f"  {name:<22} spans={b['spans']:<3} "
                  f"custo=US${b['cost_usd']:.6f}  lat={b['latency_s']:.3f}s")
    print()


def main() -> None:
    """CLI: agrega um arquivo de traces JSONL e imprime o resumo.

    Uso::

        COPOM_TRACE_FILE=data/traces.jsonl python -m eval.run_eval   # gera
        python -m obs.tracing data/traces.jsonl                      # resume
    """
    import argparse

    parser = argparse.ArgumentParser(description="Resumo de traces do COPOM RAG Service.")
    parser.add_argument("path", type=Path, nargs="?",
                        default=Path(os.getenv(TRACE_FILE_ENV, "data/traces.jsonl")),
                        help="arquivo de traces JSONL (default: $COPOM_TRACE_FILE ou data/traces.jsonl).")
    args = parser.parse_args()
    _print_summary(summarize_traces(args.path))


if __name__ == "__main__":
    main()
