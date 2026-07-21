# Design — Motor de Probabilidade da Decisão do Copom (número + narrativa)

> **🅿️ ROADMAP FUTURO — escopo parqueado (2026-07-21).**
> Esta é uma ideia de evolução, **fora do escopo atual**. Após a auditoria empírica
> (ver [`auditoria-teorica-e-cenarios.md`](auditoria-teorica-e-cenarios.md)), o
> motor mostrou-se **sobreconfiante e modesto out-of-sample** — não maduro para
> produto. O foco voltou ao que já está sólido: a **Sentimento COPOM API** e o
> **RAG básico do Copom**. Este documento fica preservado como registro de design
> para retomada futura.
>
> **Status original:** proposta de arquitetura (sem código).
> **Data:** 2026-07-20.

## 1. O problema

O público-alvo (mesa de *rates*) está **no meio do intervalo entre reuniões** e
precisa de uma leitura prospectiva: **qual a probabilidade de alta, manutenção ou
corte na próxima reunião**, e **como a conjuntura recente desloca essa
probabilidade**.

O RAG atual é **retrospectivo**: lê a última ata (o que o Copom já pensou). Mas a
decisão futura depende de tudo que aconteceu **desde** a última ata — IPCA, Focus
(revisado toda semana), câmbio, choques de oferta, ruído fiscal. Nada disso está
na ata anterior.

**Objetivo:** transformar o RAG de arquivo histórico em **motor prospectivo** que
entrega um **gráfico de distribuição** (alta / manutenção / corte) **com intervalo
de confiança**, atualizado conforme a conjuntura muda.

### Exemplo-guia (do Vitor)

> IPCA mais pressionado + quebra de safra + a última ata disse "data-dependente"
> → a probabilidade de alta sobe.

A estrutura desse raciocínio é o coração do design:

1. **A ata declara a regra de reação** ("somos data-dependentes"). Não é ruído — é
   o Copom dizendo *quanto* vai reagir a dados novos.
2. **Chega uma surpresa** (IPCA acima do esperado; choque de oferta).
3. **Regra × surpresa** → o viés se desloca para alta.

O *forward guidance* da ata **condiciona o peso** de cada dado novo. Essa é a ponte
entre o RAG (que lê a ata) e o modelo (que pondera as surpresas).

---

## 2. Princípio de arquitetura: separar "o número" do "porquê"

Decisão tomada: abordagem **híbrida**.

| Componente | Responsabilidade | Por quê |
|---|---|---|
| **Modelo estatístico** | Produz P(alta/mant/corte) + intervalo de confiança | Proveniência estatística — passa no comitê de modelos; o LLM **nunca** inventa o número |
| **RAG** | Recupera os *drivers* e escreve a narrativa que justifica cada barra | Explica e contesta o número com fontes citáveis |

Um LLM cuspindo "P(alta)=70% ± 8%" seria **falsa precisão** — inaceitável num
banco. A separação garante que a probabilidade tenha origem auditável e que o RAG
fique no seu papel legítimo: **interpretar**, não fabricar.

```
┌─────────── CAMADA DE SINAIS (ingestão datada) ───────────┐
│ Focus (revisões semanais) · IPCA · câmbio · hiato · Selic │
│  → SURPRESA = realizado − esperado(Focus)                 │
└───────────────┬───────────────────────────┬───────────────┘
                ▼                             ▼
   ┌────────────────────────┐   ┌──────────────────────────────┐
   │ MODELO (o NÚMERO)      │   │ RAG (o PORQUÊ)                │
   │ ordered probit:        │◀──│ regra de reação da última ata │
   │ surpresas + tom da ata │tom│ + trechos que justificam cada  │
   │ → P(3 estados) + IC    │   │   driver                       │
   └───────────┬────────────┘   └───────────────┬──────────────┘
               └───────────────┬─────────────────┘
                               ▼
              ┌──────────────────────────────────┐
              │ GRÁFICO: 3 barras + IC            │
              │ + narrativa por barra (fontes RAG)│
              └──────────────────────────────────┘
```

---

## 3. Camada de sinais — as *features* (surpresas)

O insumo do modelo **não** é o nível de cada variável, mas a **surpresa**:
`realizado − esperado`, onde o esperado é o Focus da véspera da reunião. É a
surpresa que move probabilidade — o esperado já está no preço.

| *Feature* | Definição | Sinal esperado |
|---|---|---|
| `surpresa_ipca` | IPCA realizado − Focus (véspera) | + → viés de alta |
| `surpresa_focus` | Revisão da expectativa Focus 12m no intervalo | + → alta (desancoragem) |
| `desvio_inflacao` | Expectativa 12m − meta | + → alta |
| `delta_cambio` | Variação do câmbio no intervalo | + → alta (repasse) |
| `hiato` | Hiato do produto | + → alta (pressão de demanda) |
| `tom_ata` | Score de tom da última ata (Sentimento COPOM) | + → alta |
| **Interação** | `tom_ata × surpresa_ipca` | captura o "data-dependente pesa mais" |

O termo de **interação** é o que formaliza o exemplo-guia: quando a ata é
hawkish/data-dependente, a surpresa de inflação pesa mais na probabilidade.

### O ecossistema de projetos que alimenta o motor

O motor **não coleta nada sozinho** — ele orquestra cinco projetos do portfólio:

| Projeto | O que fornece ao motor |
|---|---|
| **Sentimento COPOM** | `tom_ata` — o score hawkish-dovish da última ata (3 LLMs) |
| **Taylor_Sentimento** | features por reunião já montadas em `data/base_taylor.csv` |
| **Previsão Macro** | o **"esperado"** — forecast de IPCA/câmbio/Selic contra o qual se mede a surpresa; já consome o Sentimento COPOM |
| **COPOM RAG Service** | a regra de reação da ata + trechos citáveis (a narrativa) |
| **Warehouse Macro + IA** | a **camada de dados governada** — séries BCB/IBGE/FRED via Airflow+dbt (raw→staging→gold) com contratos de dados; serve dados confiáveis e versionados às fontes acima, em vez de cada projeto coletar solto |

O `base_taylor.csv` já traz 48 reuniões com `d_selic` (o alvo), `score_ensemble`
(tom), `ipca_e12m`, `meta_12m`, `desvio_inflacao`, `hiato`, `selic_lag` — a maior
parte das *features*.

### O que FALTA coletar/alinhar

- **Surpresa de Focus intra-intervalo** (a revisão semanal — o sinal mais vivo);
- **Câmbio** alinhado por reunião (`delta_cambio`) — o Previsão Macro já o modela;
- **Surpresa de IPCA** (realizado vs. Focus da véspera), para o *nowcast*.

À medida que o **Warehouse** amadurece, essas séries passam a vir dele (governadas
e testadas) em vez de coletas ad-hoc — reduzindo o risco de dado inconsistente
entre projetos.

---

## 4. O modelo — ordered probit (o "número")

**Especificação:** a decisão é ordenada (corte < manutenção < alta), então o
modelo natural é um **probit/logit ordenado**:

```
y*_t = β·X_t + ε_t          X_t = [surpresas, tom_ata, interação]
y_t  = corte    se y*_t ≤ κ₁
       manutenção se κ₁ < y*_t ≤ κ₂
       alta     se y*_t > κ₂
```

Saída: `P(corte)`, `P(manutenção)`, `P(alta)` — uma distribuição sobre 3 estados,
somando 1. O **intervalo de confiança** vem da incerteza dos parâmetros
(erros-padrão → simulação/*delta method*), propagada para as probabilidades.

### Honestidade sobre os dados

- **Amostra curta** (~48 reuniões, 3 estados) → IC **largo**. Isso é *correto*: a
  incerteza é real e o gráfico deve mostrá-la, não escondê-la. Um IC apertado com
  n=48 seria mentira.
- **Alternativa/robustez:** ancorar (ou validar) contra a **probabilidade implícita
  na curva de DI** (B3/Anbima). Se um dia plugarmos dados de mercado, o modelo
  passa a ter uma âncora externa e o RAG explica divergências modelo × mercado.
- **Validação:** a mesma disciplina do resto do portfólio — *walk-forward* (treina
  em t, prevê t+1), reportando acurácia de classificação e *log-loss*.

---

## 5. O RAG — a narrativa (o "porquê")

O RAG ganha responsabilidades novas, além de responder perguntas sobre atas:

1. **Extrair a regra de reação** da última ata: é data-dependente? quais gatilhos
   explícitos? qual o viés declarado? (isto vira, inclusive, a *feature* `tom_ata`
   e o condicionamento das interações).
2. **Recuperar, por driver, o trecho que o justifica** — para cada barra do
   gráfico, o RAG retriova as fontes (ata, comunicado, Focus datado) que sustentam
   aquele deslocamento de probabilidade.
3. **Narrar e contestar** — escrever o texto que acompanha o gráfico: "P(alta)
   subiu porque o Focus 2026 foi revisado de 3,8% para 4,0% [fonte] e a ata sinalizou
   data-dependência [fonte]; risco de baixa: atividade desacelerando [fonte]".

### Contrato de saída (estruturado, Pydantic)

```
DistribuicaoCopom:
  reuniao_alvo: str
  gerado_em: date
  estados: list[EstadoProb]        # 3 estados
  narrativa: str                   # síntese textual
  drivers: list[Driver]            # cada sinal + fonte + contribuição

EstadoProb:
  estado: "alta" | "manutencao" | "corte"
  prob: float                      # do MODELO
  ic_baixo: float; ic_alto: float  # do MODELO
  justificativa: str               # do RAG

Driver:
  nome: str                        # "surpresa_ipca", "tom_ata"...
  valor: float
  contribuicao: "alta" | "baixa" | "neutro"
  fonte: str                       # ata_279, focus_2026-07-14... (auditável)
```

Note que `prob` e `ic_*` vêm **do modelo**; `justificativa`, `narrativa` e `fonte`
vêm **do RAG**. A fronteira é explícita no próprio schema — é o que torna o produto
defensável.

---

## 6. O gráfico

- **Três barras** (corte / manutenção / alta) com **intervalo de confiança** (barra
  de erro ou faixa sombreada).
- **Interativo:** *tooltip* em cada barra com a `justificativa` e as `fontes` do RAG.
- **Comparativo temporal (fase 2):** a distribuição de *hoje* vs. a de logo após a
  ata — mostra o **movimento** que a conjuntura causou (o cerne do pedido).
- Servido pela API (`GET /probabilidade`) e renderizável na ficha de portfólio.

---

## 7. Faseamento proposto

| Fase | Entrega | Depende de |
|---|---|---|
| **0 — este doc** | Arquitetura revisada e aprovada | — |
| **1 — camada de sinais** | Base de *features*/surpresas por reunião (reusa `base_taylor.csv` + coleta câmbio/Focus semanal) | API BCB |
| **2 — modelo** | Ordered probit + IC + *walk-forward*; endpoint `GET /probabilidade` (números reais, narrativa stub) | Fase 1 |
| **3 — RAG narrativo** | Extração da regra de reação + drivers citáveis + narrativa por barra | RAG atual + Fase 2 |
| **4 — gráfico** | Visualização com IC + *tooltip* de fontes + comparativo temporal | Fases 2–3 |
| **5 — eval** | Gate: *log-loss*/acurácia do modelo + fidelidade da narrativa (LLM-as-judge sobre "a narrativa cita fonte para cada driver?") | Fases 2–4 |

---

## 8. Riscos e decisões em aberto

1. **Falsa precisão** — mitigada pela separação número/narrativa e por IC honesto
   (largo com n pequeno). **Não** deixar o LLM produzir o número.
2. **Amostra curta** — 48 reuniões limitam o probit. Decisão em aberto: ancorar em
   DI/mercado (mais robusto, exige dados B3/Anbima) vs. só modelo próprio.
3. **Vazamento temporal** — no *walk-forward*, garantir que nenhuma *feature* use
   informação posterior à reunião prevista (ex.: Focus **da véspera**, não o final).
4. **Choques não capturados** (quebra de safra, ruído fiscal) — nem toda surpresa
   está numa série. Aqui o **RAG cobre o que o modelo não vê**: recupera o texto do
   choque e sinaliza que o modelo pode subestimar. É a defesa contra o modelo cego.
5. **Fronteira de projetos** — este motor consome **cinco** projetos: Sentimento
   COPOM (tom), Taylor_Sentimento (features), Previsão Macro (o "esperado"), COPOM
   RAG Service (narrativa) e Warehouse Macro + IA (dados governados). Decisão: onde
   ele mora? Sugestão: um serviço novo que **orquestra** os cinco, mantendo cada um
   coeso — ver o diagrama do ecossistema em
   `docs/ecossistema-motor-probabilidade.excalidraw`.

---

## 9. Resumo em uma frase

Um **modelo ordenado** dá a distribuição de probabilidade da próxima decisão do
Copom **com intervalo de confiança honesto**; o **RAG** recupera os *drivers* da
conjuntura e escreve a narrativa citável que explica **por que** a probabilidade
está onde está — e se desloca quando a conjuntura muda. O número é do modelo; o
porquê é do RAG; a incerteza é mostrada, não escondida.
