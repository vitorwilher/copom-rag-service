"""API FastAPI do COPOM RAG Service.

Expõe o pipeline RAG como serviço HTTP. O endpoint principal é `POST /ask`:
recebe uma pergunta em linguagem natural sobre política monetária brasileira e
retorna uma resposta fundamentada nas atas do Copom e no boletim Focus, com as
fontes citadas.

Executar em desenvolvimento::

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv

    load_dotenv()  # carrega ANTHROPIC_API_KEY etc.; sem .env, pipeline usa fallback
except ImportError:
    pass

from obs.tracing import trace
from rag.pipeline import RAGPipeline

app = FastAPI(
    title="COPOM RAG Service",
    description="RAG sobre atas do Copom + Focus, servido como API.",
    version="0.1.0",
)

# Pipeline único reaproveitado entre requests (retriever/índice carregados uma vez).
# TODO: mover para dependência com ciclo de vida (lifespan) quando houver estado real.
pipeline = RAGPipeline()


class AskRequest(BaseModel):
    """Corpo da requisição de `/ask`.

    Attributes:
        question: pergunta em linguagem natural sobre política monetária.
    """

    question: str = Field(
        ...,
        min_length=3,
        description="Pergunta sobre as atas do Copom ou o Focus.",
        examples=["Qual foi a decisão de juros na última reunião do Copom?"],
    )


class AskResponse(BaseModel):
    """Resposta de `/ask`.

    Attributes:
        answer: resposta em linguagem natural.
        sources: fontes citadas (atas/Focus).
        model: modelo LLM usado na geração.
    """

    answer: str
    sources: list[str] = []
    model: str = "claude-haiku-4-5"


@app.get("/health")
def health() -> dict[str, str]:
    """Healthcheck simples para orquestradores (Docker/compose/k8s)."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Responde a uma pergunta via pipeline RAG.

    Envolve a chamada em um span de tracing para registrar custo, latência e
    tokens por request (observability).

    Args:
        req: corpo com a pergunta do usuário.

    Returns:
        `AskResponse` com resposta, fontes e modelo.
    """
    with trace("ask", model=pipeline.model, question=req.question):
        answer = pipeline.run(req.question)
        return AskResponse(**answer.model_dump())
