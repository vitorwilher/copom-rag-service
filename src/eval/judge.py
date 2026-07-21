"""LLM-as-judge — avalia respostas do pipeline contra a resposta esperada.

Dado o trio `(pergunta, resposta_gerada, resposta_esperada)`, um LLM juiz emite
um **score contínuo em [0, 1]** e uma **justificativa** textual. O score mede
correção factual e cobertura em relação ao gabarito do golden set, com atenção
especial a números (juros, metas, projeções) — em domínio de política monetária,
um número errado é uma falha grave.

A saída do juiz é estruturada via Pydantic (`JudgeVerdict`) usando os *structured
outputs* do SDK (`messages.parse`), garantindo que o runner sempre receba um
score numérico parseável mesmo que o texto do juiz varie.

Design de robustez: o juiz chama Claude de verdade quando `ANTHROPIC_API_KEY`
existe; sem chave, cai para um juiz heurístico determinístico (sobreposição
léxica entre resposta e gabarito) — o eval roda em CI/offline e ainda produz um
score defensável, em vez de reprovar tudo com 0.0 fixo.
"""

from __future__ import annotations

import os
import re
import statistics

from pydantic import BaseModel, Field

from obs.tracing import trace

# TODO: mover para configuração compartilhada com o pipeline.
JUDGE_MODEL = "claude-sonnet-5"

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

    def __init__(
        self,
        model: str = JUDGE_MODEL,
        max_tokens: int = 1024,
        samples: int = 3,
    ) -> None:
        """Inicializa o juiz.

        Args:
            model: identificador do modelo LLM usado como juiz.
            max_tokens: teto de tokens de saída do veredito.
            samples: nº de avaliações independentes por caso (self-consistency).
                Agregadas por mediana da nota + voto de maioria na alucinação, o
                que reduz a variância do juiz (~1/√n) e **estabiliza o gate**,
                que antes oscilava em cima do threshold. Ignorado no fallback
                heurístico (determinístico). `samples=1` = comportamento antigo.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.samples = max(1, samples)
        self._llm = _make_client()

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
        """Avalia uma resposta e retorna o veredito estruturado agregado.

        Amostra o juiz `self.samples` vezes e agrega (self-consistency):
        **mediana** das notas (robusta a outliers) e **voto de maioria** na flag
        de alucinação. É isso que estabiliza o gate perto do threshold.

        Args:
            question: pergunta do golden set.
            answer: resposta gerada pelo pipeline.
            expected: resposta esperada (gabarito).

        Returns:
            `JudgeVerdict` agregado (score, justificativa, flag de alucinação).
        """
        if self._llm is None:
            return _heuristic_verdict(answer, expected)

        prompt = self.build_prompt(question, answer, expected)
        verdicts = [self._score_once(prompt, question) for _ in range(self.samples)]
        return _aggregate_verdicts(verdicts)

    def _score_once(self, prompt: str, question: str) -> JudgeVerdict:
        """Uma única avaliação do juiz LLM (com fallback heurístico em falha).

        Args:
            prompt: prompt de avaliação já montado.
            question: pergunta (para o span de tracing).

        Returns:
            `JudgeVerdict` de uma amostra.
        """
        try:
            with trace("judge", model=self.model, question=question) as rec:
                resp = self._llm.messages.parse(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                    output_format=JudgeVerdict,
                )
                rec.input_tokens = resp.usage.input_tokens
                rec.output_tokens = resp.usage.output_tokens
            return resp.parsed_output
        except Exception:  # noqa: BLE001 — API indisponível → juiz heurístico
            # Degrada em vez de derrubar o eval (rede, rate limit, saldo).
            # Extrai a resposta do prompt para a heurística (formato conhecido).
            answer = _extract_field(prompt, "RESPOSTA GERADA")
            expected = _extract_field(prompt, "RESPOSTA ESPERADA")
            return _heuristic_verdict(answer, expected)


def _make_client():
    """Instancia o cliente Anthropic se houver chave; senão retorna None."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        return anthropic.Anthropic()
    except ImportError:
        return None


_TOKEN_RE = re.compile(r"[0-9a-zà-ÿ]+", re.IGNORECASE)


def _heuristic_verdict(answer: str, expected: str) -> JudgeVerdict:
    """Juiz heurístico determinístico — fallback sem LLM.

    Score = cobertura dos termos do gabarito pela resposta (fração dos tokens do
    esperado presentes na resposta). É um proxy grosseiro de correção/cobertura
    para o eval rodar offline; **não substitui o juiz LLM**.

    Não emite flag de alucinação: detectar informação inventada exige julgamento
    semântico que a heurística não tem. Sinalizar `hallucination` por presença de
    números daria falso positivo justamente no modo extrativo (que cita as atas
    verbatim — o oposto de alucinar). A flag fica reservada ao juiz LLM.

    Args:
        answer: resposta gerada.
        expected: resposta esperada (gabarito).

    Returns:
        `JudgeVerdict` heurístico (sempre `hallucination=False`).
    """
    exp_tokens = {t for t in _TOKEN_RE.findall(expected.lower()) if len(t) > 2}
    ans_tokens = {t for t in _TOKEN_RE.findall(answer.lower()) if len(t) > 2}
    if not exp_tokens:
        return JudgeVerdict(score=0.0, rationale="Gabarito vazio.", hallucination=False)

    coverage = len(exp_tokens & ans_tokens) / len(exp_tokens)
    return JudgeVerdict(
        score=round(coverage, 3),
        rationale=f"Heurístico (sem LLM): cobertura de termos do gabarito = {coverage:.2f}.",
        hallucination=False,
    )


def _aggregate_verdicts(verdicts: list[JudgeVerdict]) -> JudgeVerdict:
    """Agrega N amostras do juiz em um veredito estável (self-consistency).

    Usa **mediana** das notas (robusta a um outlier ocasional) e **voto de
    maioria** na flag de alucinação. A justificativa reporta a dispersão para
    tornar a estabilidade auditável.

    Args:
        verdicts: amostras independentes do juiz (>= 1).

    Returns:
        `JudgeVerdict` agregado.
    """
    scores = [v.score for v in verdicts]
    median = round(statistics.median(scores), 3)
    halluc = sum(v.hallucination for v in verdicts) > len(verdicts) / 2

    if len(verdicts) == 1:
        return verdicts[0]

    spread = max(scores) - min(scores)
    base = verdicts[len(scores) // 2].rationale  # justificativa de uma amostra
    rationale = (
        f"[mediana de {len(verdicts)} amostras; nota={median:.2f}, "
        f"dispersão={spread:.2f}] {base}"
    )
    return JudgeVerdict(score=median, rationale=rationale, hallucination=halluc)


def _extract_field(prompt: str, label: str) -> str:
    """Recupera um campo (`RESPOSTA GERADA`, `RESPOSTA ESPERADA`) do prompt.

    Usado só no caminho de fallback heurístico, quando a chamada ao juiz falha e
    precisamos dos textos originais (o prompt tem formato conhecido e fixo,
    definido em `Judge.build_prompt`).

    Args:
        prompt: prompt de avaliação montado por `build_prompt`.
        label: rótulo da seção a extrair.

    Returns:
        Conteúdo da seção, ou string vazia se não encontrado.
    """
    m = re.search(rf"{re.escape(label)}:\n(.*?)(?:\n\n[A-ZÇÃ]|\Z)", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""
