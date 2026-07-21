"""Pipeline RAG — orquestra retrieve → montar prompt → gerar resposta.

Este é o núcleo de geração servido pela API e exercitado pelo eval harness.
O contrato é estável: `RAGPipeline.run(pergunta)` retorna um `RAGAnswer` com a
resposta em linguagem natural e a lista de fontes citadas (para auditabilidade —
requisito inegociável quando o domínio é comunicação de banco central).

Design de robustez: a chamada ao LLM é real (Claude via SDK Anthropic) quando
`ANTHROPIC_API_KEY` está no ambiente; sem chave, o pipeline degrada para uma
resposta extrativa determinística (concatena os trechos mais relevantes) para
que o eval rode em CI/offline sem custo. Toda geração é instrumentada por
`obs.tracing.trace` (custo/latência/tokens).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from obs.tracing import trace
from rag.corpus import load_corpus
from rag.retriever import HybridRetriever, RetrievedChunk


class RAGAnswer(BaseModel):
    """Saída estruturada do pipeline RAG.

    Attributes:
        answer: resposta em linguagem natural à pergunta do usuário.
        sources: fontes (atas/Focus) usadas para compor a resposta.
        model: modelo LLM que gerou a resposta.
    """

    answer: str = Field(description="Resposta em linguagem natural.")
    sources: list[str] = Field(
        default_factory=list,
        description="Identificadores das fontes citadas (ex.: 'ata_265').",
    )
    model: str = Field(default="claude-haiku-4-5")


# TODO: mover o system prompt para arquivo versionado e testá-lo no eval harness.
SYSTEM_PROMPT = """Você é um assistente especializado em política monetária \
brasileira. Responda estritamente com base nos trechos fornecidos das atas do \
Copom e do boletim Focus. Cite as fontes. Se a resposta não estiver no \
contexto, diga que não há informação suficiente — nunca invente números."""


class RAGPipeline:
    """Orquestra a passagem completa recuperação → prompt → geração.

    Componível: recebe um `HybridRetriever` e um cliente LLM (injetados para
    facilitar teste e troca de provedor). O corpus é carregado e indexado uma
    vez na construção.
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 1024,
    ) -> None:
        """Inicializa o pipeline e indexa o corpus.

        Args:
            retriever: retriever híbrido (default: `HybridRetriever()`).
            model: identificador do modelo LLM de geração.
            max_tokens: teto de tokens de saída da geração.
        """
        self.retriever = retriever or HybridRetriever()
        self.model = model
        self.max_tokens = max_tokens
        if not getattr(self.retriever, "_indexed", False):
            self.retriever.index(load_corpus())
        self._llm = _make_client()

    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Monta o prompt de usuário concatenando contexto e pergunta.

        Cada chunk é prefixado com seu marcador de fonte `[ata_NNN]` para
        permitir citação rastreável na resposta.

        Args:
            question: pergunta do usuário.
            chunks: trechos recuperados que formam o contexto.

        Returns:
            String do prompt de usuário pronta para envio ao LLM.
        """
        contexto = "\n\n".join(f"[{c.source}] {c.text}" for c in chunks)
        return f"Contexto:\n{contexto}\n\nPergunta: {question}"

    def run(self, question: str) -> RAGAnswer:
        """Executa o fluxo RAG completo para uma pergunta.

        Fluxo:
            1. `retriever.retrieve(question)` → chunks;
            2. `build_prompt(question, chunks)` → prompt;
            3. chamada ao LLM com `SYSTEM_PROMPT` → texto + usage;
            4. empacota em `RAGAnswer` com as fontes dos chunks.

        Args:
            question: pergunta do usuário.

        Returns:
            `RAGAnswer` com resposta e fontes.
        """
        chunks = self.retriever.retrieve(question)
        prompt = self.build_prompt(question, chunks)
        sources = _dedupe([c.source for c in chunks])

        with trace("generate", model=self.model, question=question) as rec:
            if self._llm is None:
                answer = _extractive_answer(question, chunks)
            else:
                try:
                    resp = self._llm.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    answer = "".join(
                        b.text for b in resp.content if b.type == "text"
                    )
                    rec.input_tokens = resp.usage.input_tokens
                    rec.output_tokens = resp.usage.output_tokens
                except Exception:  # noqa: BLE001 — API indisponível → extrativo
                    # Degrada para resposta extrativa (rede, rate limit, saldo).
                    answer = _extractive_answer(question, chunks)

        return RAGAnswer(answer=answer, sources=sources, model=self.model)


def _make_client():
    """Instancia o cliente Anthropic se houver chave; senão retorna None.

    Returns:
        Cliente `anthropic.Anthropic` ou `None` (modo fallback offline).
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        return anthropic.Anthropic()
    except ImportError:
        return None


def _extractive_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Resposta extrativa determinística — fallback sem LLM.

    Não gera linguagem natural nova: devolve os trechos mais relevantes com suas
    fontes, o suficiente para o eval exercitar o retrieval de ponta a ponta sem
    depender de uma chave de API. Se nada for recuperado, abstém-se.

    Args:
        question: pergunta do usuário (não usada; assinatura simétrica ao LLM).
        chunks: trechos recuperados.

    Returns:
        Texto extrativo com marcadores de fonte, ou abstenção explícita.
    """
    if not chunks:
        return "Não há informação suficiente no contexto recuperado."
    trechos = [f"[{c.source}] {c.text}" for c in chunks[:3]]
    return (
        "Com base nos trechos recuperados das atas do Copom:\n\n"
        + "\n\n".join(trechos)
    )


def _dedupe(items: list[str]) -> list[str]:
    """Remove duplicatas preservando a ordem de primeira ocorrência."""
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out
