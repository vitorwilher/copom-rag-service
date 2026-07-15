"""LLM-as-judge — avalia respostas do pipeline contra a resposta esperada.

Dado o trio `(pergunta, resposta_gerada, resposta_esperada)`, um LLM juiz emite
um **score contínuo em [0, 1]** e uma **justificativa** textual. O score mede
correção factual e cobertura em relação ao gabarito do golden set, com atenção
especial a números (juros, metas, projeções) — em domínio de política monetária,
um número errado é uma falha grave.

A saída do juiz é estruturada via Pydantic (`JudgeVerdict`), garantindo que o
runner sempre receba um score numérico parseável mesmo que o texto do juiz varie.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# TODO: mover para configuração compartilhada com o pipeline.
JUDGE_MODEL = "claude-sonnet-4-5"

JUDGE_SYSTEM_PROMPT = """Você é um avaliador rigoroso de respostas sobre \
política monetária brasileira. Compare a RESPOSTA GERADA com a RESPOSTA \
ESPERADA para a PERGUNTA. Atribua um score de 0.0 (totalmente incorreta ou \
alucinada) a 1.0 (correta, completa e sem invenções). Penalize fortemente \
números incorretos. Retorne o score e uma justificativa curta e objetiva."""


class JudgeVerdict(BaseModel):
    """Veredito estruturado do LLM-as-judge para uma única resposta.

    Attributes:
        score: nota contínua em [0, 1] (correção + cobertura).
        rationale: justificativa textual objetiva da nota.
        hallucination: True se o juiz identificou informação inventada.
    """

    score: float = Field(ge=0.0, le=1.0, description="Nota em [0, 1].")
    rationale: str = Field(description="Justificativa curta da nota.")
    hallucination: bool = Field(
        default=False, description="Houve informação inventada?"
    )


class Judge:
    """Encapsula o LLM juiz e o parsing da saída estruturada."""

    def __init__(self, model: str = JUDGE_MODEL) -> None:
        """Inicializa o juiz.

        Args:
            model: identificador do modelo LLM usado como juiz.
        """
        self.model = model
        self._llm = None  # TODO: anthropic.Anthropic(...) client

    def build_prompt(self, question: str, answer: str, expected: str) -> str:
        """Monta o prompt de avaliação para o juiz.

        Args:
            question: pergunta do golden set.
            answer: resposta gerada pelo pipeline.
            expected: resposta esperada (gabarito).

        Returns:
            Prompt de usuário para o LLM juiz.
        """
        return (
            f"PERGUNTA:\n{question}\n\n"
            f"RESPOSTA ESPERADA:\n{expected}\n\n"
            f"RESPOSTA GERADA:\n{answer}\n\n"
            "Avalie a RESPOSTA GERADA."
        )

    def score(self, question: str, answer: str, expected: str) -> JudgeVerdict:
        """Avalia uma resposta e retorna o veredito estruturado.

        Args:
            question: pergunta do golden set.
            answer: resposta gerada pelo pipeline.
            expected: resposta esperada (gabarito).

        Returns:
            `JudgeVerdict` com score, justificativa e flag de alucinação.

        Nota:
            Enquanto a chamada ao LLM não está conectada, retorna um veredito
            neutro (score 0.0) com justificativa-stub — o que faz o eval FALHAR
            por padrão, sinalizando corretamente que o pipeline ainda não gera
            respostas reais. Isso é intencional: o gate deve reprovar um sistema
            vazio.

        TODO: chamar `self._llm.messages.create(...)` com saída estruturada
              (tool use / JSON schema de `JudgeVerdict`) e envolver em
              `obs.tracing.trace("judge", model=self.model)`.
        """
        _ = self.build_prompt(question, answer, expected)  # prompt pronto p/ envio
        # with trace("judge", model=self.model) as rec:
        #     resp = self._llm.messages.create(..., tools=[JudgeVerdict schema])
        #     rec.input_tokens = resp.usage.input_tokens
        #     rec.output_tokens = resp.usage.output_tokens
        # return JudgeVerdict.model_validate(resp_tool_input)
        return JudgeVerdict(
            score=0.0,
            rationale="STUB: juiz LLM não conectado — ver Judge.score (TODO).",
            hallucination=False,
        )
