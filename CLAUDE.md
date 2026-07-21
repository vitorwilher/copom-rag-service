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
└── obs/tracing.py      # context manager de custo/latência/tokens
scripts/download_atas.py # API do BCB → data/atas_full_cache.json (atas completas)
scripts/ingest.py       # atas_full_cache.json → data/corpus.jsonl
data/corpus.jsonl       # 814 chunks de 48 atas completas (232–279), versionado
tests/                  # pytest: retriever, pipeline, judge, gate (15 testes)
portfolio/
└── copom-rag-service.qmd   # ficha CRISP-DM (padrão-ouro do portfólio)
```

## Estado atual (2026-07-21)

- **Motor implementado e rodável de ponta a ponta.** Retrieval híbrido real
  (BM25 + denso + RRF + rerank) sobre 814 chunks de 48 atas **completas** (A+B+C)
  baixadas do BCB. Pipeline e juiz chamam Claude via SDK quando há
  `ANTHROPIC_API_KEY` (no `.env` local, git-ignorado); sem chave, degradam para
  modos determinísticos (extrativo / heurístico) — eval roda offline sem custo.
- Ingestão: `python scripts/download_atas.py` (API do BCB → atas completas) →
  `python scripts/ingest.py` (→ corpus.jsonl). Baixa desde a reunião 232.
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
- `python -m eval.run_eval` roda o gate; `pytest` roda 22 testes (todos passam).
- **Ainda TODO**: Focus não ingerido; atas < reunião 232 (API lista desde a 21);
  exportador real de métricas do tracing (OTel/Prometheus); embeddings via
  provider (hoje fallback local — descomente openai/google em requirements).

## Convenções (herdadas do portfólio do Vitor)

- Conteúdo em **pt-BR**, terminologia econômica precisa (autor é economista).
- CRISP-DM como espinha dorsal da ficha em `portfolio/`.
- Falta gerar a imagem de capa `copom-rag-service-capa.png`.
- Render do Quarto e git só com pedido explícito.

## Stack

FastAPI · uvicorn · pydantic · chromadb (ou pgvector) · rank-bm25 · anthropic · pytest.
