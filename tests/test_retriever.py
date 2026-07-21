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
