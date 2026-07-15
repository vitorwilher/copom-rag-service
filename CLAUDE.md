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
├── app/main.py         # FastAPI, endpoint /ask (stub)
├── rag/retriever.py    # retrieval híbrido BM25 + denso + reranking (stubs)
├── rag/pipeline.py     # orquestra retrieve → prompt → gerar (stub)
├── eval/golden_set.jsonl  # 8 perguntas reais sobre o Copom + gabarito
├── eval/judge.py       # LLM-as-judge → score [0,1] + rationale (Pydantic)
├── eval/run_eval.py    # CORAÇÃO: roda golden set, agrega, gate exit 1
└── obs/tracing.py      # context manager de custo/latência/tokens
portfolio/
└── copom-rag-service.qmd   # ficha CRISP-DM (padrão-ouro do portfólio)
```

## Estado atual (2026-07-14)

- Esqueleto rodável: imports corretos, docstrings, corpos com TODO explícito.
- **Ainda TODO**: retrieval real (índice BM25 + vector store), chamada ao LLM
  no pipeline e no juiz, exportador de métricas do tracing.
- Por design, o eval **reprova** enquanto o pipeline for stub (juiz retorna 0.0)
  — o gate deve reprovar um sistema vazio. Isso é correto, não um bug.

## Convenções (herdadas do portfólio do Vitor)

- Conteúdo em **pt-BR**, terminologia econômica precisa (autor é economista).
- CRISP-DM como espinha dorsal da ficha em `portfolio/`.
- Falta gerar a imagem de capa `copom-rag-service-capa.png`.
- Render do Quarto e git só com pedido explícito.

## Stack

FastAPI · uvicorn · pydantic · chromadb (ou pgvector) · rank-bm25 · anthropic · pytest.
