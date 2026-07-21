# Roadmap — ideias parqueadas

Material de design e pesquisa **fora do escopo atual**, preservado para retomada
futura. Nada aqui está em produção.

## Motor de Probabilidade da Decisão do Copom

Ideia: um serviço que entrega a **distribuição de probabilidade** da próxima
decisão do Copom (alta / manutenção / corte) com intervalo de confiança, movida
por um ordered probit sobre as surpresas de conjuntura, com o RAG narrando o
porquê — e cenários em linguagem natural ("e se a guerra do Irã continuar?").

**Por que foi parqueado (2026-07-21):** a auditoria empírica sobre dados reais
mostrou que o modelo, com a amostra atual (~47 reuniões), é **sobreconfiante** e
tem **poder out-of-sample modesto** (log-loss pior que o chute uniforme; sinal de
separação perfeita). A hipótese central — a interação tom×inflação — **não é
estatisticamente significativa** (p=0,25). Levar isso a produto agora seria
construir sobre base frágil. O foco voltou ao que já está sólido e provado.

**O que fica sólido (fora daqui):** a Sentimento COPOM API (features `tom_ata` etc.
são reaproveitáveis) e o COPOM RAG Service.

### Arquivos

| Arquivo | O que é |
|---|---|
| [`design-motor-probabilidade-copom.md`](design-motor-probabilidade-copom.md) | Arquitetura do motor (número do modelo + porquê do RAG), o ecossistema de 5 projetos, contratos e faseamento |
| [`auditoria-teorica-e-cenarios.md`](auditoria-teorica-e-cenarios.md) | Auditoria empírica do ordered probit (o que passou, o que não passou) + design dos cenários em linguagem natural |
| `ecossistema-motor-probabilidade.excalidraw` | Diagrama dos 5 projetos convergindo no motor |
| `design-motor-probabilidade-copom.html` | Render Quarto do design doc |

### Para retomar, o que seria preciso

1. **Mais dados** — a amostra de ~47 reuniões é o gargalo; a interação e a
   calibração exigem mais história.
2. **Regularização** do probit (ridge/Firth) para conter a separação perfeita.
3. **IC por reamostragem** (bootstrap/walk-forward), não delta method.
4. **Âncora de mercado** (curva DI/Anbima) para validação externa.
