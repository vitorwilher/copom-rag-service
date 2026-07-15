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

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


# Preços de referência (US$ por 1M de tokens). Ajustar conforme o modelo servido.
# TODO: mover para configuração / carregar do provedor via `claude-api` reference.
PRICE_TABLE: dict[str, dict[str, float]] = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
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


def _emit(record: TraceRecord) -> None:
    """Sink temporário de métricas — imprime o registro em formato legível.

    TODO: substituir por exportador real (span OTel, contador Prometheus).
    """
    print(
        f"[trace] {record.name:<10} model={record.model} "
        f"lat={record.latency_s:.3f}s "
        f"tok_in={record.input_tokens} tok_out={record.output_tokens} "
        f"cost=US${record.cost_usd:.6f}"
    )
