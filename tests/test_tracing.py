"""Testes do exportador de traces — sink JSONL + agregador de sessão."""

from __future__ import annotations

import json

from obs.tracing import TraceRecord, summarize_traces, trace


def test_trace_record_to_dict_serializavel():
    rec = TraceRecord(name="generate", model="claude-haiku-4-5")
    rec.input_tokens, rec.output_tokens = 2000, 150
    rec.compute_cost()
    d = rec.to_dict()
    # dict plano, JSON-serializável, com as dimensões operacionais
    json.dumps(d)  # não deve levantar
    assert d["name"] == "generate"
    assert d["input_tokens"] == 2000
    assert d["cost_usd"] > 0


def test_sink_jsonl_grava_span_quando_env_setada(tmp_path, monkeypatch):
    path = tmp_path / "traces.jsonl"
    monkeypatch.setenv("COPOM_TRACE_FILE", str(path))
    with trace("judge", model="claude-sonnet-5") as rec:
        rec.input_tokens, rec.output_tokens = 1000, 80
    assert path.exists()
    linhas = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(linhas) == 1
    rec_dict = json.loads(linhas[0])
    assert rec_dict["name"] == "judge"
    assert rec_dict["model"] == "claude-sonnet-5"


def test_sink_nao_grava_sem_env(tmp_path, monkeypatch):
    # sem COPOM_TRACE_FILE, nada é escrito (não suja disco nos testes/eval)
    monkeypatch.delenv("COPOM_TRACE_FILE", raising=False)
    path = tmp_path / "naodeveexistir.jsonl"
    with trace("generate", model="claude-haiku-4-5"):
        pass
    assert not path.exists()


def test_summarize_agrega_custo_latencia_e_degradacoes(tmp_path):
    path = tmp_path / "traces.jsonl"
    registros = [
        {"name": "generate", "model": "claude-haiku-4-5", "latency_s": 2.0,
         "input_tokens": 2000, "output_tokens": 100, "cost_usd": 0.0025, "metadata": {}},
        {"name": "judge", "model": "claude-sonnet-5", "latency_s": 3.0,
         "input_tokens": 1000, "output_tokens": 90, "cost_usd": 0.0044, "metadata": {}},
        {"name": "generate", "model": "claude-haiku-4-5", "latency_s": 1.0,
         "input_tokens": 2200, "output_tokens": 80, "cost_usd": 0.0026,
         "metadata": {"degraded": True, "degrade_reason": "RateLimitError"}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in registros), encoding="utf-8")

    s = summarize_traces(path)
    assert s["spans"] == 3
    assert s["degraded"] == 1  # captura a degradação sinalizada
    assert round(s["cost_usd"], 4) == round(0.0025 + 0.0044 + 0.0026, 4)
    assert s["by_model"]["claude-haiku-4-5"]["spans"] == 2
    assert s["by_name"]["judge"]["spans"] == 1


def test_summarize_arquivo_inexistente_retorna_vazio(tmp_path):
    s = summarize_traces(tmp_path / "nao_existe.jsonl")
    assert s["spans"] == 0
    assert s["by_model"] == {}
