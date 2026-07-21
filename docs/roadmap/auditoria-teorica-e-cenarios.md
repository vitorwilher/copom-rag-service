# Auditoria Teórica do Motor + Design dos Cenários em Linguagem Natural

> **🅿️ ROADMAP FUTURO — escopo parqueado (2026-07-21).**
> Foi **esta auditoria** que motivou parquear o motor de probabilidade: os achados
> abaixo (sobreconfiança, poder out-of-sample modesto, interação não significativa)
> mostraram que o motor não está maduro para produto. O foco voltou à **Sentimento
> COPOM API** e ao **RAG básico**. Documento preservado como o registro do porquê —
> e do que seria preciso para retomar com rigor.
>
> **Status original:** auditoria empírica sobre dados reais (`base_taylor.csv`,
> 47 reuniões) + design dos cenários.
> **Data:** 2026-07-20.

---

## PARTE 1 — Auditoria teórica interna do ordered probit

Escopo desta fase (decisão do Vitor): **consistência interna**, não âncora de
mercado. A validação contra a curva de DI/Anbima fica para uma fase futura (exige
dados de mercado que o portfólio ainda não tem).

O modelo **foi de fato estimado** com os dados reais — não é hipótese. Resultados:

### ✅ O que passou (o modelo tem base sólida)

| Verificação | Resultado |
|---|---|
| **Sinais econômicos** | Todos corretos: tom↑→alta, desvio de inflação↑→alta, hiato↑→alta, Selic defasada↑→menos alta (reversão) |
| **Significância** | `desvio_inflacao` (p=0,001), `score_ensemble` / tom (p=0,004), `selic_lag` (p<0,001) — todos significantes a 1% |
| **Ajuste in-sample** | Pseudo-R² = 0,515 — bom para 3 estados, n=47 |
| **Tom da ata como feature** | O `score_ensemble` do Sentimento COPOM é estatisticamente relevante — **valida a integração dos projetos**: o tom carrega informação sobre a decisão futura |

### ⚠️ O que exige cautela (achados que mudam o design)

1. **`hiato` é insignificante (p=0,867).** O hiato do produto, teoricamente um
   driver, não ajuda **neste** conjunto. Provável causa: mal medido no Brasil e já
   parcialmente capturado por `desvio_inflacao`. **Decisão:** removê-lo da
   especificação principal (parcimônia), mantendo-o como candidato a revisitar com
   melhor medida de atividade.

2. **A interação `tom × desvio` NÃO é significativa (p=0,246).** Este é o achado
   mais importante — e humilde. O **cenário-guia** ("ata data-dependente faz o dado
   de inflação pesar mais") era a hipótese central, mas **os dados não a sustentam
   com n=47**. O coeficiente tem o sinal certo (+0,60) e melhora o pseudo-R²
   marginalmente, mas o AIC piora (59,0 vs. 58,4) — ou seja, não paga seu custo de
   parâmetro. **Decisão honesta:** manter a interação **fora** do modelo estatístico
   por ora; a intuição do "data-dependente pesa mais" é economicamente válida, mas
   não é estatisticamente distinguível com esta amostra. Registrar como hipótese a
   testar quando a amostra crescer.

3. **Poder out-of-sample é modesto.** No *walk-forward* (22 reuniões previstas fora
   da amostra): acurácia de **54,5%** em 3 classes (vs. 40,4% do chute majoritário —
   então **há sinal real**, mas moderado). Porém o **log-loss de 2,66 é RUIM** —
   pior que o chute uniforme (1,10). Tradução: o modelo **acerta a classe mais
   provável razoavelmente, mas suas probabilidades são mal-calibradas e
   sobreconfiantes** fora da amostra. Isto é decisivo para o gráfico (ver abaixo).

4. **🚩 Sinal de separação perfeita.** 4 de 47 previsões in-sample têm prob > 0,99.
   Com n pequeno e um probit, isso indica **separação quasi-perfeita** — o modelo
   fica sobreconfiante, e os erros-padrão dos parâmetros ficam pouco confiáveis
   (o Hessiano falhou em inverter em parte das janelas do walk-forward). É a causa
   raiz do log-loss ruim.

### O que a auditoria muda no GRÁFICO

O ponto 3+4 é a lição central e ela **valida uma decisão de design que já
tínhamos** — e reforça outra:

- **O IC largo não era só estética honesta — é empiricamente necessário.** O modelo
  é sobreconfiante; portanto a incerteza **precisa** ser mostrada com destaque, e as
  probabilidades pontuais (63% etc.) devem vir acompanhadas de um IC honesto e
  provavelmente **mais largo do que o probit ingênuo sugere**. Recomenda-se calcular
  o IC por **bootstrap / walk-forward** (a dispersão real das previsões
  out-of-sample), não pelo *delta method* sobre erros-padrão que o próprio modelo
  admite serem frágeis.
- **Regularização é necessária.** Para conter a separação perfeita: um probit
  **penalizado** (ridge/Firth) ou um *prior* fraco. Sem isso, as probabilidades
  extremas (99%) são artefato, não sinal.

### Especificação recomendada (pós-auditoria)

```
y*_t = β₁·tom_ata + β₂·desvio_inflacao + β₃·selic_lag + ε_t
```
Três features, todas significantes, sinais corretos. Sem `hiato`, sem interação
(por ora). Estimação **regularizada**. IC das probabilidades por **reamostragem**,
não por delta method.

> **Conclusão da Parte 1:** o motor é **teoricamente sólido na direção** (sinais e
> significância corretos; o tom da ata realmente informa) mas **modesto e
> sobreconfiante na magnitude** (poder out-of-sample limitado, risco de separação).
> O produto é honesto **se e somente se** o gráfico enfatizar a incerteza e as
> probabilidades forem regularizadas. Um gráfico que mostrasse "78% ± 2%" seria uma
> mentira que os próprios dados desmentem.

---

## PARTE 2 — Cenários em linguagem natural ("e se a guerra do Irã continuar?")

Decisão do Vitor: **o LLM estima a DIREÇÃO de cada feature; a MAGNITUDE vem do
histórico.** O LLM nunca produz a probabilidade — o probit recalcula. Esta é a
versão mais conservadora e defensável.

### O fluxo

```
1. Cliente (linguagem natural):  "e se a guerra do Irã continuar?"
                                        │
2. RAG interpreta o choque e mapeia para DIREÇÕES nas features:
     petróleo ↑ → câmbio ↑        (direção: +)
                → inflação import. ↑ (direção: + em desvio_inflacao)
                → prêmio de risco ↑  (direção: +)
                                        │
3. HISTÓRICO fornece a MAGNITUDE de cada direção:
     "choques geopolíticos análogos moveram o câmbio em ~+Δ típico"
     (percentis de movimentos passados — não um chute do LLM)
                                        │
4. As features deslocadas entram no ORDERED PROBIT (regularizado)
                                        │
5. O modelo recalcula P(corte/mant/alta) + IC  →  a curva se MOVE
                                        │
6. RAG narra: "sua premissa (guerra continua) desloca o câmbio e o prêmio de
    risco; pelo histórico de choques similares, isso eleva P(alta) de X% para Y%.
    Premissas usadas: [câmbio +Δ, prêmio +Δ]. Fontes: [...]. Edite se discordar."
```

### Por que essa arquitetura é defensável

- **O LLM fica preso ao que sabe fazer:** interpretar texto e mapear causalidade
  qualitativa (guerra→petróleo→câmbio). Ele **não** inventa números.
- **A magnitude é empírica:** vem da distribuição histórica de choques análogos, não
  da imaginação do modelo. Choque inédito sem análogo → o sistema **declara** que
  não tem base para a magnitude (não chuta).
- **Premissas visíveis e editáveis:** o cliente vê "câmbio +3%, prêmio +0,4pp" e
  pode ajustar. A cadeia narrativa→número é auditável ponta a ponta.
- **A fronteira número/porquê se mantém:** RAG traduz e narra; probit calcula.

### Contrato (esboço)

```
CenarioRequest:
  texto: str                       # "e se a guerra do Irã continuar?"

CenarioResponse:
  premissas: list[Premissa]        # o que o choque implica (editável)
  distribuicao: DistribuicaoCopom  # recalculada pelo probit (número + IC)
  narrativa: str                   # explicação citável (RAG)
  base_magnitude: str              # de onde veio cada magnitude (histórico)

Premissa:
  feature: str                     # "delta_cambio", "desvio_inflacao"
  direcao: "+" | "-"               # do RAG (o que ele sabe fazer)
  magnitude: float                 # do histórico (percentil de choques análogos)
  editavel: true
  fonte_direcao: str               # trecho/racional do RAG
  fonte_magnitude: str             # "p75 de choques geopolíticos 2018-2025"
```

### Riscos específicos dos cenários

1. **Choque sem análogo histórico** — o LLM dá a direção, mas o histórico não tem
   magnitude. **Regra:** o sistema mostra a direção e marca a magnitude como
   "sem base histórica — informe manualmente", em vez de inventar.
2. **Dupla contagem** — se o choque já está parcialmente nos dados correntes (o
   câmbio já subiu), aplicar o Δ de novo conta duas vezes. O RAG precisa checar o
   que já está no *baseline*.
3. **Combinações não-lineares** — dois choques simultâneos não somam linearmente. O
   probit é linear na latente; cenários compostos devem alertar sobre isso.
4. **A calibração ruim do modelo (Parte 1) contamina o cenário.** Se o probit é
   sobreconfiante, o *movimento* da curva sob um cenário também será exagerado. A
   regularização da Parte 1 é pré-requisito para os cenários serem confiáveis.

---

## Faseamento revisado (pós-auditoria)

| Fase | Entrega | Nota da auditoria |
|---|---|---|
| **1** | Camada de sinais (features) | `hiato` sai; foco em tom, desvio, selic_lag |
| **2** | Probit **regularizado** + IC por reamostragem | crítico: sem regularização, probabilidades são artefato |
| **3** | Gráfico com IC honesto (largo) | a auditoria confirma: enfatizar incerteza |
| **4** | RAG narrativo | — |
| **5** | **Cenários em linguagem natural** | direção (LLM) + magnitude (histórico); só após Fase 2 sólida |
| **6** | **Âncora de mercado** (curva DI) | valida o modelo contra o que o mercado precifica |

---

## Resumo em uma frase

A auditoria **aprovou a direção** do motor (o tom da ata realmente prevê a decisão;
sinais corretos e significantes) mas **reprovou a sobreconfiança** (poder
out-of-sample modesto, separação perfeita, log-loss ruim) — o que torna o **IC
honesto e a regularização não opcionais, mas obrigatórios**; e os **cenários em
linguagem natural** são viáveis e defensáveis desde que o LLM só dê a direção e a
magnitude venha do histórico, nunca da imaginação do modelo.
