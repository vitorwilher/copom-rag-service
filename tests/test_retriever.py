"""Testes do retrieval híbrido — BM25, denso, RRF e rerank."""

from __future__ import annotations

from rag.retriever import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Reranker,
    RetrievedChunk,
    _data_por_extenso,
    indexing_text,
    tokenize,
)

CHUNKS = [
    RetrievedChunk(
        "O regime de metas para a inflação ancora as expectativas na meta do CMN.",
        "ata_001", metadata={"chunk": 0},
    ),
    RetrievedChunk(
        "O hiato do produto mede a atividade em relação ao potencial da economia.",
        "ata_002", metadata={"chunk": 0},
    ),
    RetrievedChunk(
        "O boletim Focus agrega as expectativas de mercado para IPCA e Selic.",
        "ata_003", metadata={"chunk": 0},
    ),
]


def test_tokenize_remove_stopwords_e_normaliza():
    toks = tokenize("O hiato DO produto")
    assert "hiato" in toks and "produto" in toks
    assert "do" not in toks  # stopword removida


def test_bm25_recupera_por_termo_lexical():
    bm25 = BM25Retriever(k=3)
    bm25.index(CHUNKS)
    hits = bm25.search("hiato do produto")
    assert hits[0].source == "ata_002"  # casamento léxico forte


def test_dense_recupera_por_similaridade():
    dense = DenseRetriever(k=3)
    dense.index(CHUNKS)
    hits = dense.search("expectativas de inflação")
    assert len(hits) == 3
    assert hits[0].score >= hits[-1].score  # ordenado por similaridade


def test_reranker_corta_em_top_n():
    reranker = Reranker(top_n=2)
    out = reranker.rerank("Focus expectativas", CHUNKS)
    assert len(out) == 2
    assert out[0].source == "ata_003"  # chunk do Focus vem primeiro


def test_hybrid_funde_e_reranqueia():
    hybrid = HybridRetriever(reranker=Reranker(top_n=2))
    hybrid.index(CHUNKS)
    hits = hybrid.retrieve("metas para a inflação")
    assert 1 <= len(hits) <= 2
    assert hits[0].source == "ata_001"


def test_rrf_deduplica_por_fonte():
    hybrid = HybridRetriever()
    hybrid.index(CHUNKS)
    fused = hybrid._rrf(CHUNKS, CHUNKS)  # mesma lista duas vezes
    sources = [c.source for c in fused]
    assert len(sources) == len(set(sources))  # sem duplicatas


def test_data_por_extenso():
    assert _data_por_extenso("2026-06-17") == "junho de 2026"
    assert _data_por_extenso("2025-01-29") == "janeiro de 2025"
    assert _data_por_extenso("") == ""  # formato inválido


def test_indexing_text_injeta_proveniencia():
    c = RetrievedChunk("texto qualquer", "ata_279",
                       metadata={"meeting": "279", "data": "2026-06-17"})
    idx = indexing_text(c)
    assert "ata_279" in idx and "279" in idx
    assert "junho de 2026" in idx  # data por extenso p/ casar linguagem natural
    assert "texto qualquer" in idx  # texto original preservado


def test_retrieval_ancorado_por_ata_prioriza_a_ata_certa():
    # Duas atas com conteúdo parecido; a pergunta ancora numa reunião específica.
    chunks = [
        RetrievedChunk("O Comitê decidiu manter a taxa Selic inalterada.",
                       "ata_268", metadata={"meeting": "268", "data": "2025-01-29"}),
        RetrievedChunk("O Comitê decidiu manter a taxa Selic inalterada.",
                       "ata_279", metadata={"meeting": "279", "data": "2026-06-17"}),
    ]
    r = HybridRetriever(reranker=Reranker(top_n=2))
    r.index(chunks)
    hits = r.retrieve("Qual a decisão na reunião de janeiro de 2025 (ata 268)?")
    assert hits[0].source == "ata_268"  # âncora da ata/data desempata


# --- Boost de decisão (Reranker) e sua interação com o Focus ---

_ATA_DECISAO = RetrievedChunk(
    "D) Decisão de política monetária. O Copom decidiu reduzir a Selic para 14,25% a.a.",
    "ata_279", metadata={"meeting": "279", "data": "2026-06-17"})
_ATA_DISCUSSAO = RetrievedChunk(
    "O Comitê discutiu a condução da política monetária e o balanço de riscos.",
    "ata_279", metadata={"meeting": "279", "data": "2026-06-17", "chunk": 1})
_FOCUS = RetrievedChunk(
    "Expectativas do mercado (Focus) na véspera da reunião 279: a taxa Selic 14,0 ao fim de 2026.",
    "focus_2026-06-17", metadata={"meeting": "279", "data": "2026-06-17", "kind": "focus"})


def test_boost_decisao_prioriza_chunk_com_o_numero():
    # Pergunta sobre a DECISÃO: o chunk com o número deve vencer o de discussão.
    r = Reranker(top_n=2)
    out = r.rerank("Qual a decisão de juros da ata 279?", [_ATA_DISCUSSAO, _ATA_DECISAO])
    assert out[0].source == "ata_279"
    assert "14,25" in out[0].text  # o chunk da decisão, não o da discussão


def test_pergunta_de_mercado_nao_e_sequestrada_pelo_boost():
    # Pergunta sobre FOCUS/mercado: o boost de decisão fica desligado, então o
    # chunk focus_* compete em pé de igualdade e não perde para a ata de decisão.
    r = Reranker(top_n=1)
    out = r.rerank(
        "O que o mercado (Focus) esperava para a Selic na reunião 279?",
        [_ATA_DECISAO, _FOCUS],
    )
    assert out[0].source.startswith("focus_")
