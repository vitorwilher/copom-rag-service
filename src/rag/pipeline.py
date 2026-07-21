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
import time

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
Copom e do boletim Focus. Cite a fonte pelo identificador (ex.: ata_279). Se a \
resposta não estiver no contexto, diga que não há informação suficiente — nunca \
invente números.

Responda de forma direta e factual. NÃO reproduza citações textuais entre aspas \
a menos que o trecho apareça literalmente no contexto; se for citar, copie \
exatamente — nunca parafraseie dentro de aspas nem invente número de parágrafo. \
Na dúvida, afirme o fato com suas palavras e aponte a fonte, sem aspas."""


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
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ) -> None:
        """Inicializa o pipeline e indexa o corpus.

        Args:
            retriever: retriever híbrido (default: `HybridRetriever()`).
            model: identificador do modelo LLM de geração.
            max_tokens: teto de tokens de saída da geração.
            max_retries: tentativas extras em falha transitória da API antes de
                degradar para o modo extrativo (0 desativa o retry).
            backoff_base: base do backoff exponencial em segundos (espera de
                `backoff_base * 2**tentativa` entre as tentativas).
        """
        self.retriever = retriever or HybridRetriever()
        self.model = model
        self.max_tokens = max_tokens
        self._max_retries = max_retries
        self._backoff_base = backoff_base
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
                # Sem chave: modo offline determinístico (não é uma falha).
                rec.metadata["mode"] = "extractive_no_key"
                answer = _extractive_answer(question, chunks)
            else:
                answer = self._generate(prompt, chunks, rec)

        return RAGAnswer(answer=answer, sources=sources, model=self.model)

    def _generate(self, prompt: str, chunks: list[RetrievedChunk], rec) -> str:
        """Gera a resposta via LLM com retry em falhas transitórias.

        Erros transitórios (rate limit, indisponibilidade, timeout de rede) são
        reexecutados com backoff exponencial. Só depois de esgotar as tentativas
        — ou diante de um erro claramente não-transitório — o pipeline degrada
        para a resposta extrativa. **A degradação é registrada no trace**
        (`rec.metadata["degraded"]` + motivo): silenciar essa queda mascarava
        respostas de baixa qualidade como se fossem geração real, contaminando o
        eval de forma não-determinística.

        Args:
            prompt: prompt de usuário já montado.
            chunks: trechos recuperados (para o fallback extrativo).
            rec: `TraceRecord` do span de geração, anotado com o desfecho.

        Returns:
            Texto da resposta (do LLM, ou extrativo se tudo falhar).
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._llm.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )
                rec.input_tokens = resp.usage.input_tokens
                rec.output_tokens = resp.usage.output_tokens
                rec.metadata["mode"] = "llm"
                if attempt:
                    rec.metadata["retries"] = attempt
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._max_retries and _is_transient(exc):
                    time.sleep(self._backoff_base * (2**attempt))
                    continue
                break

        # Esgotou as tentativas (ou erro não-transitório): degrada e SINALIZA.
        rec.metadata["mode"] = "extractive_fallback"
        rec.metadata["degraded"] = True
        rec.metadata["degrade_reason"] = type(last_exc).__name__ if last_exc else "unknown"
        return _extractive_answer("", chunks)


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


def _is_transient(exc: Exception) -> bool:
    """Classifica se um erro do SDK justifica retry (transitório) ou não.

    Transitório: rate limit (429), indisponibilidade (5xx), timeout/erro de
    conexão — condições que tendem a passar numa nova tentativa. Persistente:
    erro de autenticação, requisição inválida, saldo esgotado — retry só gasta
    tempo. A checagem é por nome de classe/atributo para não acoplar a versões
    específicas do SDK anthropic.

    Args:
        exc: exceção capturada na chamada ao LLM.

    Returns:
        True se vale a pena reexecutar; False para degradar de imediato.
    """
    name = type(exc).__name__
    if name in {
        "RateLimitError", "APITimeoutError", "APIConnectionError",
        "InternalServerError", "APIStatusError", "OverloadedError",
    }:
        return True
    status = getattr(exc, "status_code", None)
    return status in {429, 500, 502, 503, 504, 529}


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
