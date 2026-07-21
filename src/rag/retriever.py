"""Retrieval híbrido (BM25 esparso + denso) com reranking.

A recuperação combina dois sinais complementares:

- **BM25** (esparso, léxico) — forte em correspondência exata de termos, útil
  para jargão de política monetária ("meta para a inflação", "hiato do
  produto", "carry", nomes de reuniões do Copom);
- **denso** (embeddings) — captura similaridade semântica, robusto a paráfrases
  e sinônimos.

Os candidatos das duas fontes são fundidos por **Reciprocal Rank Fusion (RRF)** e
um **reranker** reordena o conjunto pela relevância fina à pergunta antes de
montar o contexto do prompt.

Design de robustez: o retriever denso usa embeddings de um provedor (Google /
OpenAI, chave em `.env`) quando disponível e cai para um embedding local
determinístico (TF-IDF por hashing) caso contrário — assim o eval roda em CI
sem chave e sem baixar modelos pesados. BM25, RRF e rerank são sempre reais.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from rag.embeddings import embed_texts


@dataclass
class RetrievedChunk:
    """Trecho recuperado do índice, com metadados de proveniência.

    Attributes:
        text: conteúdo textual do chunk.
        source: identificador da fonte (ex.: "ata_265", "focus_2026-07-11").
        score: score de relevância atribuído pelo retriever/reranker.
        metadata: campos livres (data da reunião, seção, url, etc.).
    """

    text: str
    source: str
    score: float = 0.0
    metadata: dict | None = None


_TOKEN_RE = re.compile(r"[0-9a-zà-ÿ]+", re.IGNORECASE)


_MESES = {
    "01": "janeiro", "02": "fevereiro", "03": "março", "04": "abril",
    "05": "maio", "06": "junho", "07": "julho", "08": "agosto",
    "09": "setembro", "10": "outubro", "11": "novembro", "12": "dezembro",
}


def _data_por_extenso(data_iso: str) -> str:
    """Converte 'AAAA-MM-DD' em 'mês de AAAA' (ex.: 'janeiro de 2025').

    A data ISO não casa com a query em linguagem natural ("janeiro de 2025");
    a forma por extenso sim. Retorna string vazia se o formato não bater.
    """
    m = re.match(r"(\d{4})-(\d{2})-\d{2}", data_iso or "")
    if not m:
        return ""
    ano, mes = m.group(1), m.group(2)
    return f"{_MESES.get(mes, '')} de {ano}"


def indexing_text(chunk: RetrievedChunk) -> str:
    """Texto usado para **indexar** um chunk — conteúdo + proveniência.

    Prefixa o texto com o identificador da ata, o número da reunião e a data
    (ISO **e** por extenso em pt-BR). Isso faz perguntas ancoradas numa ata/data
    específica ("decisão da ata 279", "reunião de junho de 2026", "janeiro de
    2025") casarem com o chunk certo — sem a proveniência, o retriever só via o
    texto e ignorava esses âncoras fortes. O `.text` do chunk retornado
    permanece limpo (sem o prefixo), para exibição e citação.

    Args:
        chunk: chunk do corpus.

    Returns:
        Texto enriquecido para indexação (BM25 e denso).
    """
    meta = chunk.metadata or {}
    reuniao = meta.get("meeting", "")
    data = meta.get("data", "")
    extenso = _data_por_extenso(data)
    return f"{chunk.source} reunião {reuniao} {data} {extenso} {chunk.text}"

# Stopwords PT-BR mínimas — removem ruído de alta frequência sem exigir NLTK.
_STOPWORDS = {
    "a", "o", "e", "de", "da", "do", "das", "dos", "que", "em", "no", "na",
    "nos", "nas", "um", "uma", "para", "por", "com", "os", "as", "ao", "aos",
    "se", "ou", "the", "of", "is", "qual", "como", "sobre", "entre",
}


def tokenize(text: str) -> list[str]:
    """Tokeniza texto PT-BR: minúsculas, alfanumérico, sem stopwords curtas.

    Args:
        text: texto a tokenizar.

    Returns:
        Lista de tokens.
    """
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


class BM25Retriever:
    """Retriever esparso baseado em BM25 (`rank-bm25` com fallback interno).

    Indexa o corpus tokenizado e recupera os top-k documentos por score BM25.
    Se `rank-bm25` não estiver instalado, usa uma implementação BM25 local
    equivalente — o serviço não fica inerte por falta de dependência.
    """

    def __init__(self, k: int = 20) -> None:
        """Inicializa o retriever esparso.

        Args:
            k: número de candidatos a retornar por consulta.
        """
        self.k = k
        self._index = None
        self._chunks: list[RetrievedChunk] = []
        self._corpus_tokens: list[list[str]] = []

    def index(self, chunks: list[RetrievedChunk]) -> None:
        """Constrói o índice BM25 a partir do corpus.

        Args:
            chunks: chunks do corpus (atas + Focus) já com proveniência.
        """
        self._chunks = chunks
        # Indexa texto + proveniência; retorna o chunk com o texto original.
        self._corpus_tokens = [tokenize(indexing_text(c)) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi

            self._index = BM25Okapi(self._corpus_tokens)
        except ImportError:
            self._index = _LocalBM25(self._corpus_tokens)

    def search(self, query: str) -> list[RetrievedChunk]:
        """Recupera os top-k chunks por score BM25.

        Args:
            query: pergunta do usuário.

        Returns:
            Lista de `RetrievedChunk` ordenada por score decrescente.
        """
        if self._index is None:
            raise RuntimeError("BM25Retriever.index precisa ser chamado antes de search")
        scores = self._index.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[RetrievedChunk] = []
        for i in ranked[: self.k]:
            c = self._chunks[i]
            out.append(RetrievedChunk(c.text, c.source, float(scores[i]), c.metadata))
        return out


@dataclass
class _LocalBM25:
    """BM25 Okapi mínimo — fallback quando `rank-bm25` não está instalado."""

    corpus: list[list[str]]
    k1: float = 1.5
    b: float = 0.75
    _df: Counter = field(default_factory=Counter)
    _idf: dict = field(default_factory=dict)
    _len: list[int] = field(default_factory=list)
    _avglen: float = 0.0

    def __post_init__(self) -> None:
        self._len = [len(doc) for doc in self.corpus]
        n = len(self.corpus)
        self._avglen = sum(self._len) / n if n else 0.0
        for doc in self.corpus:
            for term in set(doc):
                self._df[term] += 1
        for term, df in self._df.items():
            self._idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * len(self.corpus)
        for i, doc in enumerate(self.corpus):
            freqs = Counter(doc)
            dl = self._len[i] or 1
            for term in query_tokens:
                if term not in freqs:
                    continue
                idf = self._idf.get(term, 0.0)
                tf = freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self._avglen or 1))
                scores[i] += idf * (tf * (self.k1 + 1)) / denom
        return scores


class DenseRetriever:
    """Retriever denso sobre embeddings (provider via `.env` ou fallback local).

    Codifica query e documentos em vetores e recupera por similaridade de
    cosseno. Os embeddings do corpus são pré-computados no `index`.
    """

    def __init__(self, collection: str = "copom", k: int = 20) -> None:
        """Inicializa o retriever denso.

        Args:
            collection: nome lógico da coleção (rótulo de proveniência).
            k: número de candidatos a retornar por consulta.
        """
        self.collection = collection
        self.k = k
        self._chunks: list[RetrievedChunk] = []
        self._matrix: list[list[float]] = []

    def index(self, chunks: list[RetrievedChunk]) -> None:
        """Pré-computa os embeddings normalizados do corpus.

        Args:
            chunks: chunks do corpus.
        """
        self._chunks = chunks
        # Embeddings sobre texto + proveniência (mesma lógica do BM25).
        self._matrix = embed_texts([indexing_text(c) for c in chunks])

    def search(self, query: str) -> list[RetrievedChunk]:
        """Recupera os top-k chunks por similaridade semântica.

        Args:
            query: pergunta do usuário.

        Returns:
            Lista de `RetrievedChunk` ordenada por similaridade decrescente.
        """
        if not self._matrix:
            raise RuntimeError("DenseRetriever.index precisa ser chamado antes de search")
        qvec = embed_texts([query])[0]
        sims = [_cosine(qvec, vec) for vec in self._matrix]
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        out: list[RetrievedChunk] = []
        for i in ranked[: self.k]:
            c = self._chunks[i]
            out.append(RetrievedChunk(c.text, c.source, float(sims[i]), c.metadata))
        return out


def _cosine(a: list[float], b: list[float]) -> float:
    """Similaridade de cosseno entre dois vetores (0.0 se algum for nulo)."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# Sinais para o boost de "chunk de decisão" (ver Reranker.rerank).
# Uma pergunta sobre a decisão de juros usa estes termos...
_DECISION_QUERY_TERMS = {
    "decisão", "decidiu", "elevou", "reduziu", "manteve", "taxa", "selic",
    "juros", "patamar", "ponto", "pontos", "elevar", "reduzir", "manter",
}
# ...e o chunk que a responde carrega o dado explícito da decisão: o patamar
# numérico da Selic, o cabeçalho oficial da seção "D) Decisão de política
# monetária", ou o verbo de calibração aplicado à taxa.
_DECISION_CHUNK_RE = re.compile(
    r"selic\D{0,15}\d{1,2},\d{2}"      # "Selic para 14,25"
    r"|\d{1,2},\d{2}\s*%?\s*a\.?\s*a"  # "14,25% a.a."
    r"|decisão de política monetária"  # cabeçalho oficial da seção D
    r"|(?:elev|reduz|man)\w+\s+a\s+taxa"  # "reduzir a taxa"
    r"|\d,\d{2}\s*ponto",              # "1,00 ponto percentual"
    re.IGNORECASE,
)


class Reranker:
    """Reordena candidatos por relevância fina à pergunta.

    Reranking léxico-semântico leve: combina sobreposição de tokens da query
    com o score de recuperação, favorecendo chunks que casam termos raros da
    pergunta. Não exige modelo externo — determinístico e barato.

    **Boost de decisão** (domínio Copom): quando a pergunta é sobre a decisão de
    juros, chunks que *contêm o dado da decisão* (patamar da Selic, cabeçalho da
    seção "D) Decisão de política monetária") ganham um bônus. Sem ele, o chunk
    genérico de "discussão da condução da política monetária" — denso nos termos
    da pergunta mas sem o número — vencia o parágrafo que traz a decisão de fato.
    """

    def __init__(self, top_n: int = 6, decision_boost: float = 0.5) -> None:
        """Inicializa o reranker.

        Args:
            top_n: número de chunks a manter após o reranking.
            decision_boost: bônus aditivo a chunks com o dado da decisão quando
                a pergunta é sobre decisão de juros (0.0 desativa).
        """
        self.top_n = top_n
        self.decision_boost = decision_boost

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Reordena e corta os candidatos pelos `top_n` mais relevantes.

        Args:
            query: pergunta do usuário.
            chunks: candidatos fundidos de BM25 + denso.

        Returns:
            Sublista reordenada com no máximo `top_n` elementos.
        """
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return chunks[: self.top_n]

        wants_decision = bool(q_tokens & _DECISION_QUERY_TERMS)
        rescored: list[RetrievedChunk] = []
        for c in chunks:
            # inclui a proveniência para a âncora da ata contar no overlap
            c_tokens = set(tokenize(indexing_text(c)))
            overlap = len(q_tokens & c_tokens) / len(q_tokens)
            # combina cobertura de termos da query (peso maior) com o score de fusão
            final = 0.7 * overlap + 0.3 * min(c.score, 1.0)
            # bônus ao chunk que carrega o dado da decisão pedida
            if wants_decision and _DECISION_CHUNK_RE.search(c.text):
                final += self.decision_boost
            rescored.append(RetrievedChunk(c.text, c.source, final, c.metadata))
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[: self.top_n]


class HybridRetriever:
    """Orquestra BM25 + denso + reranking em uma única interface `retrieve`.

    Funde os candidatos das duas fontes por Reciprocal Rank Fusion (RRF) e
    delega o corte final ao reranker. O corpus é carregado uma vez via `index`.
    """

    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
    ) -> None:
        """Compõe o retriever híbrido a partir dos três componentes.

        Args:
            bm25: retriever esparso (default: `BM25Retriever()`).
            dense: retriever denso (default: `DenseRetriever()`).
            reranker: reranker final (default: `Reranker()`).
            rrf_k: constante de amortecimento do RRF (padrão da literatura: 60).
        """
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.reranker = reranker or Reranker()
        self.rrf_k = rrf_k
        self._indexed = False

    def index(self, chunks: list[RetrievedChunk]) -> None:
        """Indexa o corpus nos dois retrievers.

        Args:
            chunks: chunks do corpus (atas + Focus).
        """
        self.bm25.index(chunks)
        self.dense.index(chunks)
        self._indexed = True

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Recupera contexto relevante para a pergunta.

        Fluxo: BM25.search + DenseRetriever.search → fusão (RRF) → Reranker.

        Args:
            query: pergunta do usuário.

        Returns:
            Chunks finais que compõem o contexto do prompt.
        """
        if not self._indexed:
            raise RuntimeError("HybridRetriever.index precisa ser chamado antes de retrieve")
        sparse = self.bm25.search(query)
        dense = self.dense.search(query)
        fused = self._rrf(sparse, dense)
        return self.reranker.rerank(query, fused)

    def _rrf(
        self, sparse: list[RetrievedChunk], dense: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Funde duas listas ranqueadas via Reciprocal Rank Fusion.

        RRF soma 1/(k + rank) de cada lista em que o documento aparece,
        combinando os dois sinais sem exigir calibração de escala entre eles.

        Args:
            sparse: candidatos do BM25 (ordenados).
            dense: candidatos do denso (ordenados).

        Returns:
            Chunks únicos ordenados por score RRF decrescente.
        """
        scores: dict[str, float] = {}
        chosen: dict[str, RetrievedChunk] = {}
        for ranked in (sparse, dense):
            for rank, c in enumerate(ranked):
                key = f"{c.source}#{(c.metadata or {}).get('chunk', 0)}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                chosen.setdefault(key, c)
        ordered = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [
            RetrievedChunk(chosen[k].text, chosen[k].source, scores[k], chosen[k].metadata)
            for k in ordered
        ]
