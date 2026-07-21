"""Testes do gate do eval harness — a lógica que faz o CI reprovar."""

from __future__ import annotations

from eval.judge import JudgeVerdict
from eval.run_eval import EvalCase, EvalResult, aggregate
from rag.pipeline import RAGAnswer


def _result(score: float, halu: bool = False) -> EvalResult:
    case = EvalCase(id="q", question="?", expected="x", tags=[])
    answer = RAGAnswer(answer="a", sources=[])
    verdict = JudgeVerdict(score=score, rationale="r", hallucination=halu)
    return EvalResult(case=case, answer=answer, verdict=verdict)


def test_aggregate_calcula_media_e_minimo():
    summary = aggregate([_result(0.8), _result(0.6), _result(1.0)])
    assert summary["n"] == 3
    assert abs(summary["mean"] - 0.8) < 1e-9
    assert summary["min"] == 0.6


def test_aggregate_conta_alucinacoes():
    summary = aggregate([_result(0.9, halu=True), _result(0.9, halu=False)])
    assert summary["hallucinations"] == 1


def test_gate_reprova_abaixo_do_threshold():
    summary = aggregate([_result(0.2), _result(0.3)])
    assert summary["mean"] < 0.70  # reprova o gate padrão


def test_gate_aprova_acima_do_threshold():
    summary = aggregate([_result(0.9), _result(0.8)])
    assert summary["mean"] >= 0.70  # aprova o gate padrão
