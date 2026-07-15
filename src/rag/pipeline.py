"""Pipeline RAG — orquestra retrieve → montar prompt → gerar resposta.

Este é o núcleo de geração servido pela API e exercitado pelo eval harness.
O contrato é estável: `RAGPipeline.run(pergunta)` retorna um `RAGAnswer` com a
resposta em linguagem natural e a lista de fontes citadas (para auditabilidade —
requisito inegociável quando o domínio é comunicação de banco central).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

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
    facilitar teste e troca de provedor).
    """

    def __init__(
        self,
        retriever: HybridRetriever | None = None,
        model: str = "claude-haiku-4-5",
    ) -> None:
        """Inicializa o pipeline.

        Args:
            retriever: retriever híbrido (default: `HybridRetriever()`).
            model: identificador do modelo LLM de geração.
        """
        self.retriever = retriever or HybridRetriever()
        self.model = model
        self._llm = None  # TODO: anthropic.Anthropic(...) client

    def build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Monta o prompt de usuário concatenando contexto e pergunta.

        Args:
            question: pergunta do usuário.
            chunks: trechos recuperados que formam o contexto.

        Returns:
            String do prompt de usuário pronta para envio ao LLM.

        TODO: formatar cada chunk com marcador de fonte para permitir citação
              rastreável na resposta.
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

        TODO: conectar retrieval real e chamada ao LLM. Envolver a chamada em
              `obs.tracing.trace(...)` para registrar custo/latência/tokens.
        """
        # chunks = self.retriever.retrieve(question)
        # prompt = self.build_prompt(question, chunks)
        # with trace("generate", model=self.model) as rec:
        #     resp = self._llm.messages.create(...)
        #     rec.input_tokens = resp.usage.input_tokens
        #     rec.output_tokens = resp.usage.output_tokens
        # return RAGAnswer(answer=resp..., sources=[c.source for c in chunks])
        raise NotImplementedError(
            "RAGPipeline.run ainda não implementado — retrieval + LLM são TODO"
        )
