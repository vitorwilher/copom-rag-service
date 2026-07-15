"""Runner do eval harness — o CORAÇÃO do serviço.

Roda o golden set inteiro pelo pipeline RAG, avalia cada resposta com o
LLM-as-judge, agrega os scores e imprime um relatório. **Falha com exit code 1
se a média ficar abaixo do threshold** — é este gate que, plugado no CI, impede
que um deploy regrida a qualidade do serviço.

Uso::

    python -m eval.run_eval                       # threshold padrão
    python -m eval.run_eval --threshold 0.75      # gate customizado
    python -m eval.run_eval --golden custom.jsonl # outro golden set

O fluxo está completamente esqueletado: leitura do golden set, laço de execução
pergunta a pergunta (pipeline → judge), agregação, relatório e o gate de saída.
Os únicos pontos ainda inertes são `RAGPipeline.run` e `Judge.score`, que são
TODOs marcados nos respectivos módulos.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from eval.judge import Judge, JudgeVerdict
from rag.pipeline import RAGAnswer, RAGPipeline

# Threshold padrão do gate de qualidade (média mínima dos scores do juiz).
DEFAULT_THRESHOLD = 0.70
DEFAULT_GOLDEN = Path(__file__).parent / "golden_set.jsonl"


@dataclass
class EvalCase:
    """Um caso do golden set.

    Attributes:
        id: identificador do caso (ex.: "q01").
        question: pergunta a ser feita ao pipeline.
        expected: resposta esperada (gabarito) para o juiz.
        tags: rótulos livres para segmentar o relatório.
    """

    id: str
    question: str
    expected: str
    tags: list[str]


@dataclass
class EvalResult:
    """Resultado da avaliação de um caso.

    Attributes:
        case: o caso avaliado.
        answer: resposta gerada pelo pipeline.
        verdict: veredito do LLM-as-judge.
    """

    case: EvalCase
    answer: RAGAnswer
    verdict: JudgeVerdict


def load_golden_set(path: Path) -> list[EvalCase]:
    """Lê o golden set em JSONL e o converte em `EvalCase`s.

    Args:
        path: caminho para o arquivo `.jsonl`.

    Returns:
        Lista de casos de avaliação.
    """
    cases: list[EvalCase] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cases.append(
                EvalCase(
                    id=row["id"],
                    question=row["question"],
                    expected=row["expected"],
                    tags=row.get("tags", []),
                )
            )
    return cases


def evaluate_case(case: EvalCase, pipeline: RAGPipeline, judge: Judge) -> EvalResult:
    """Executa um caso: pipeline gera → juiz avalia.

    Args:
        case: caso do golden set.
        pipeline: pipeline RAG sob avaliação.
        judge: LLM-as-judge.

    Returns:
        `EvalResult` com resposta e veredito.

    Nota:
        Captura `NotImplementedError` do pipeline (enquanto retrieval + LLM são
        stubs) e o degrada para um `RAGAnswer` vazio, de modo que o runner rode
        de ponta a ponta e o gate reprove — em vez de estourar. Ao conectar o
        pipeline real, respostas de verdade fluem sem alteração neste código.
    """
    try:
        answer = pipeline.run(case.question)
    except NotImplementedError as exc:
        answer = RAGAnswer(answer=f"[não implementado: {exc}]", sources=[])
    verdict = judge.score(case.question, answer.answer, case.expected)
    return EvalResult(case=case, answer=answer, verdict=verdict)


def aggregate(results: list[EvalResult]) -> dict:
    """Agrega os scores dos vereditos.

    Args:
        results: resultados por caso.

    Returns:
        Dicionário com média, mínimo, nº de casos e nº de alucinações.
    """
    scores = [r.verdict.score for r in results]
    n = len(scores)
    return {
        "n": n,
        "mean": sum(scores) / n if n else 0.0,
        "min": min(scores) if scores else 0.0,
        "hallucinations": sum(1 for r in results if r.verdict.hallucination),
    }


def print_report(results: list[EvalResult], summary: dict, threshold: float) -> None:
    """Imprime o relatório por caso e o resumo agregado.

    Args:
        results: resultados por caso.
        summary: agregados de `aggregate`.
        threshold: gate de qualidade aplicado.
    """
    print("\n=== COPOM RAG Service — Eval Harness ===\n")
    print(f"{'ID':<6}{'SCORE':>7}  {'HALU':>5}  JUSTIFICATIVA")
    print("-" * 72)
    for r in results:
        halu = "sim" if r.verdict.hallucination else "-"
        rationale = r.verdict.rationale[:44]
        print(f"{r.case.id:<6}{r.verdict.score:>7.2f}  {halu:>5}  {rationale}")
    print("-" * 72)
    print(
        f"casos={summary['n']}  média={summary['mean']:.3f}  "
        f"mín={summary['min']:.3f}  alucinações={summary['hallucinations']}  "
        f"threshold={threshold:.2f}"
    )


def run(golden_path: Path, threshold: float) -> int:
    """Roda o eval completo e retorna o exit code.

    Args:
        golden_path: caminho do golden set.
        threshold: média mínima para aprovar.

    Returns:
        0 se a média >= threshold, 1 caso contrário (gate do CI).
    """
    cases = load_golden_set(golden_path)
    pipeline = RAGPipeline()
    judge = Judge()

    results = [evaluate_case(c, pipeline, judge) for c in cases]
    summary = aggregate(results)
    print_report(results, summary, threshold)

    passed = summary["mean"] >= threshold
    print(f"\nRESULTADO: {'PASSOU' if passed else 'FALHOU'} "
          f"(média {summary['mean']:.3f} {'>=' if passed else '<'} {threshold:.2f})\n")
    return 0 if passed else 1


def main() -> None:
    """Ponto de entrada CLI: parseia argumentos e sai com o código do gate."""
    parser = argparse.ArgumentParser(description="Eval harness do COPOM RAG Service.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Média mínima para aprovar (default {DEFAULT_THRESHOLD}).")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN,
                        help="Caminho do golden set JSONL.")
    args = parser.parse_args()
    sys.exit(run(args.golden, args.threshold))


if __name__ == "__main__":
    main()
