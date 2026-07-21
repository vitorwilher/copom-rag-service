"""Testes do pipeline (modo fallback) e do juiz (modo heurístico).

Os testes forçam o modo offline (`_llm = None`) para exercitar a degradação
determinística sem depender de chave de API nem de rede.
"""

from __future__ import annotations

from eval.judge import (
    Judge,
    JudgeVerdict,
    _aggregate_verdicts,
    _extract_field,
    _heuristic_verdict,
)
from rag.pipeline import RAGPipeline
from rag.retriever import HybridRetriever, Reranker, RetrievedChunk

CHUNKS = [
    RetrievedChunk(
        "O regime de metas para a inflação ancora as expectativas na meta do CMN.",
        "ata_001", metadata={"chunk": 0},
    ),
    RetrievedChunk(
        "O boletim Focus agrega as expectativas de mercado para IPCA e Selic.",
        "ata_002", metadata={"chunk": 0},
    ),
]


def _offline_pipeline() -> RAGPipeline:
    """Pipeline com corpus fixo e sem cliente LLM (modo extrativo)."""
    hybrid = HybridRetriever(reranker=Reranker(top_n=2))
    hybrid.index(CHUNKS)
    pipe = RAGPipeline(retriever=hybrid)
    pipe._llm = None  # força fallback extrativo
    return pipe


def test_pipeline_fallback_retorna_resposta_e_fontes():
    pipe = _offline_pipeline()
    ans = pipe.run("O que é o regime de metas para a inflação?")
    assert ans.sources  # cita fontes reais
    assert all(s.startswith("ata_") for s in ans.sources)
    assert "ata_001" in ans.answer  # marcador de fonte no texto extrativo


def test_build_prompt_inclui_marcadores_de_fonte():
    pipe = _offline_pipeline()
    prompt = pipe.build_prompt("pergunta?", CHUNKS)
    assert "[ata_001]" in prompt and "[ata_002]" in prompt
    assert "Pergunta: pergunta?" in prompt


def test_judge_heuristico_pontua_por_cobertura():
    v = _heuristic_verdict(
        answer="O regime de metas ancora expectativas na meta.",
        expected="O regime de metas ancora as expectativas de inflação na meta.",
    )
    assert isinstance(v, JudgeVerdict)
    assert 0.0 < v.score <= 1.0
    assert v.hallucination is False  # heurística nunca sinaliza alucinação


def test_judge_heuristico_resposta_irrelevante_pontua_baixo():
    v = _heuristic_verdict(
        answer="Xyz abc def totalmente sem relação.",
        expected="O hiato do produto mede a atividade em relação ao potencial.",
    )
    assert v.score < 0.2


def test_judge_offline_usa_heuristica():
    j = Judge()
    j._llm = None  # força modo heurístico
    v = j.score("q", "resposta qualquer sobre metas", "resposta esperada sobre metas")
    assert isinstance(v, JudgeVerdict)


def test_aggregate_usa_mediana_robusta_a_outlier():
    verdicts = [
        JudgeVerdict(score=0.8, rationale="a"),
        JudgeVerdict(score=0.9, rationale="b"),
        JudgeVerdict(score=0.1, rationale="c"),  # outlier
    ]
    agg = _aggregate_verdicts(verdicts)
    assert agg.score == 0.8  # mediana ignora o outlier (média seria 0.6)
    assert "3 amostras" in agg.rationale


def test_aggregate_alucinacao_por_maioria():
    sim = JudgeVerdict(score=0.5, rationale="r", hallucination=True)
    nao = JudgeVerdict(score=0.5, rationale="r", hallucination=False)
    assert _aggregate_verdicts([sim, sim, nao]).hallucination is True
    assert _aggregate_verdicts([sim, nao, nao]).hallucination is False


def test_aggregate_amostra_unica_e_passthrough():
    v = JudgeVerdict(score=0.7, rationale="única")
    assert _aggregate_verdicts([v]) is v  # sem overhead de agregação


def test_extract_field_recupera_secoes_do_prompt():
    j = Judge()
    j._llm = None
    prompt = j.build_prompt("Qual a meta?", "resposta gerada aqui", "gabarito aqui")
    assert _extract_field(prompt, "RESPOSTA GERADA") == "resposta gerada aqui"
    assert _extract_field(prompt, "RESPOSTA ESPERADA") == "gabarito aqui"
