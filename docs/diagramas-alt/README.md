# Diagramas alternativos (arquivados)

Versões de diagrama que **não** são o oficial da ficha. O diagrama oficial do card
é [`portfolio/copom-rag-arquitetura-card.excalidraw`](../../portfolio/copom-rag-arquitetura-card.excalidraw)
— a arquitetura técnica completa (4 faixas: ingestão → retrieval → geração → eval),
para o público AI/LLM Engineer.

| Arquivo | O que é | Por que arquivado |
|---|---|---|
| `copom-rag-service-arquitetura.excalidraw` | Diagrama técnico anterior, com o retrieval híbrido detalhado | Superado pelo card-full (mais completo) |
| `copom-sinal-triade.excalidraw` | Visão de **negócio**: direção (Sentimento) + magnitude (Taylor) + narrativa (RAG) | Útil para público executivo/econômico, não técnico — pode voltar como diagrama de contexto |

Preservados porque servem a públicos diferentes; recuperáveis se a ficha ganhar
uma seção de contexto de negócio.

## Atenção ao editar o card-full

Em **2026-07-25** o card-full foi corrigido para bater com o código implementado
— o desenho original descrevia o projeto *planejado*, não o construído:

| Dizia | É |
|---|---|
| `vector store · chromadb/pgvector` | índice denso **em memória** |
| `reranker · cross-encoder` | reranker **léxico-semântico**, sem modelo externo |
| `chunking 512 tok · overlap 64` | **1200 caracteres**, overlap 200 |
| saída `{answer, sources[], abstained}` | `{answer, sources[]}` — não há `abstained` |

A **capa da ficha do portfólio é gerada a partir deste arquivo**
(`portfolio/copom-rag-service-capa.png`), então qualquer divergência nova acaba
publicada. A versão simples anterior está preservada em
`portfolio/copom-rag-service-capa-simples.png`.
