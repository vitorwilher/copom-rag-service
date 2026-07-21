# CLAUDE.md — COPOM RAG Service

Briefing curto para sessões futuras. Projeto **EM DESENVOLVIMENTO**.

## O que é

RAG (Retrieval-Augmented Generation) servido como **API** (FastAPI + Docker)
sobre as **atas do Copom** e o **boletim Focus** do Banco Central do Brasil.
Frente do portfólio: **AI/LLM Engineer**.

O diferencial do card não é o RAG em si — é a **engenharia ao redor**:

1. **Eval harness** (`src/eval/`) — golden set + LLM-as-judge + gate de CI que
   FALHA (exit 1) se a qualidade média regride abaixo do threshold. **Este é o
   código-âncora do projeto.**
2. **Observability** (`src/obs/tracing.py`) — custo, latência e tokens por request.
3. **API** (`src/app/main.py`) — endpoint `POST /ask` (pergunta → resposta + fontes).

## Layout

```
src/
├── app/main.py         # FastAPI, endpoint /ask (chama pipeline.run)
├── rag/retriever.py    # retrieval híbrido BM25 + denso + RRF + rerank (real)
├── rag/embeddings.py   # embeddings denso: provider (.env) ou fallback local
├── rag/corpus.py       # carrega data/corpus.jsonl → RetrievedChunk
├── rag/pipeline.py     # orquestra retrieve → prompt → gerar (Claude + fallback)
├── eval/golden_set.jsonl  # 8 perguntas reais sobre o Copom + gabarito
├── eval/judge.py       # LLM-as-judge (messages.parse) + fallback heurístico
├── eval/run_eval.py    # CORAÇÃO: roda golden set, agrega, gate exit 1
└── obs/tracing.py      # trace() + sink JSONL (COPOM_TRACE_FILE) + summarize_traces
scripts/download_atas.py # API do BCB → data/atas_full_cache.json (atas completas)
scripts/download_focus.py # API Expectativas do BCB → data/focus_cache.json (ancorado às reuniões)
scripts/ingest.py       # atas_full_cache.json (+ focus_cache.json) → data/corpus.jsonl
data/corpus.jsonl       # 862 chunks: 814 de 48 atas (232–279) + 48 do Focus, versionado
tests/                  # pytest: retriever, pipeline, judge, gate, tracing, focus (29 testes)
portfolio/
└── copom-rag-service.qmd   # ficha CRISP-DM (padrão-ouro do portfólio)
```

## Estado atual (2026-07-21)

> **TL;DR:** motor no melhor ponto. **Gate = média 1,000, 0 alucinações**
> (`PYTHONPATH=src python -m eval.run_eval --samples 3`). **31 testes passam.**
> Focus ingerido (corpus 862 chunks). Tracing com sink JSONL. Tudo commitado na
> branch `feat/motor-rag-eval` (`7c4977c`, `6a95b2f`); **sem push/PR** (não pedido).
> Próximo: PR p/ main, ou atas <232, ou exportador OTel/Prometheus, ou embeddings
> via provider.

- **Motor implementado e rodável de ponta a ponta.** Retrieval híbrido real
  (BM25 + denso + RRF + rerank) sobre 862 chunks (48 atas **completas** A+B+C +
  48 do Focus) baixados do BCB. Pipeline e juiz chamam Claude via SDK quando há
  `ANTHROPIC_API_KEY` (no `.env` local, git-ignorado); sem chave, degradam para
  modos determinísticos (extrativo / heurístico) — eval roda offline sem custo.
- Ingestão: `python scripts/download_atas.py` (API do BCB → atas completas) →
  opcional `python scripts/download_focus.py` (API Expectativas do BCB → Focus
  ancorado às datas das reuniões) → `python scripts/ingest.py` (→ corpus.jsonl,
  atas + Focus). Baixa desde a reunião 232.
- **Propósito do eval = fidelidade às atas** (não é quiz conceitual). O usuário
  já domina os conceitos; quer saber o que a ata *específica* disse. O golden set
  tem 5 fatos ancorados em atas nomeadas (decisões de juros, balanço de riscos,
  hiato) + 3 abstenções (pergunta sem resposta nas atas → recusar, não inventar).
- **Retrieval ancorado**: `retriever.indexing_text()` indexa cada chunk com sua
  proveniência (`ata_NNN` + reunião + data ISO **e** por extenso). Sem isso,
  perguntas do tipo "decisão da ata 279 / junho de 2026" traziam o chunk errado.
- **Juiz estabilizado por self-consistency**: `Judge(samples=N)` amostra o juiz
  N vezes por caso (default 3) e agrega por **mediana** da nota + voto de maioria
  na alucinação — reduz a variância que fazia o gate oscilar. Flag `--samples`.
- **Focus ingerido** (48 reuniões): `download_focus.py` puxa da API Expectativas
  do BCB a mediana de Selic/IPCA/PIB/Câmbio vigente na véspera de cada reunião;
  `ingest.py` vira 1 chunk `focus_AAAA-MM-DD` por reunião. Corpus = **862 chunks**
  (814 atas + 48 Focus). O reranker desliga o boost de decisão em perguntas de
  mercado (`_MARKET_QUERY_TERMS`) para o chunk Focus não perder para as atas.
- **Tracing com sink real**: `COPOM_TRACE_FILE=path python -m eval.run_eval` grava
  cada span em JSONL (inclui `degraded`); `python -m obs.tracing <path>` agrega
  custo/latência/tokens por modelo e por span (`summarize_traces`).
- `python -m eval.run_eval` roda o gate; `pytest` roda **29 testes** (todos passam).
- **Ainda TODO**: atas < reunião 232 (API lista desde a 21); exportador OTel/
  Prometheus de fato (hoje JSONL, o formato que eles consumiriam); embeddings via
  provider (hoje fallback local — descomente openai/google em requirements).

## Convenções (herdadas do portfólio do Vitor)

- Conteúdo em **pt-BR**, terminologia econômica precisa (autor é economista).
- CRISP-DM como espinha dorsal da ficha em `portfolio/`.
- Falta gerar a imagem de capa `copom-rag-service-capa.png`.
- Render do Quarto e git só com pedido explícito.

## Stack

FastAPI · uvicorn · pydantic · chromadb (ou pgvector) · rank-bm25 · anthropic · pytest.
