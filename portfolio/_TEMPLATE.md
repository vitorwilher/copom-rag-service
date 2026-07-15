# Template `portfolio/` — Estrutura Padrão para Projetos do Portfólio

Esta pasta é o **template** que será replicado em todos os projetos futuros do portfólio pessoal. O propósito é separar três camadas do mesmo projeto:

1. **Código e dados** (raiz do projeto) — implementação técnica do projeto.
2. **Divulgação no portfólio** (esta pasta) — documentação CRISP-DM em `.qmd` que vai para `Github_Portfolio/projetos/`.
3. **Repositório GitHub** (`README.md` adaptado) — README do repositório do projeto.

## Arquivos obrigatórios

| Arquivo | Função |
|---|---|
| `<slug-do-projeto>.qmd` | Documentação CRISP-DM completa, copiada para `Github_Portfolio/projetos/` |
| `README.md` | README adaptado do `.qmd`, usado no repositório GitHub do projeto |
| `<slug>_*.png` | Imagens de capa e diagramas referenciados no `.qmd` e no README |

## Subpasta opcional `research/` (quando o projeto vira paper acadêmico)

Quando um projeto também produz um *working paper* destinado à seção `research/` do site, criar esta subpasta dentro de `portfolio/`. **Premissa fundamental:** o repositório do projeto continua privado; apenas o **PDF público** (a edição do leitor, com `echo: false`) e o *changelog* vão para o site.

```
portfolio/
└── research/
    ├── <slug>.qmd                 # página HTML do paper (abstract, PDF, changelog, BibTeX, contato)
    ├── <slug>_publico.pdf         # PDF da edição do leitor — único artefato com o conteúdo do paper
    └── <slug>_indice.png          # imagem de capa
```

A página `<slug>.qmd` contém:

1. **Callout de "working paper"** convidando à crítica via e-mail.
2. **Botão de download do PDF**.
3. **Abstract bilíngue** (PT/EN).
4. **Histórico de versões** (changelog em prosa, mais legível que o `CHANGELOG.md` técnico).
5. **Como citar** (BibTeX).
6. **Contato para diálogo acadêmico**.
7. **Cross-link para a ficha técnica do projeto** em `projetos/<slug>.html`.
8. **Nota explícita** de que o código-fonte é privado.

Os arquivos são copiados para `Github_Portfolio/research/` (não em subpasta — o listing usa `*.qmd` no top-level).

**Sincronização a cada release do paper:** o PDF público e o histórico de versões precisam ser atualizados na página `research/` quando uma nova versão é fechada. Esse passo entra no *workflow* "fechar versão" do projeto.

## Convenções

- **`.qmd`**: kebab-case (`sentimento-copom.qmd`, `roi-diagnostico.qmd`).
- **YAML do `.qmd`**: deve ter `title`, `date`, `categories`, `description`, `image`, `execute: { eval: false, echo: true }`.
- **Estrutura CRISP-DM**: as 6 fases (Entendimento do Negócio, Entendimento dos Dados, Preparação, Modelagem, Avaliação, Deployment) viram seções numeradas no `.qmd`.
- **README.md**: versão enxuta do `.qmd` — abre com tese central, mostra arquitetura/estrutura, foca em "como rodar" e remove a longa argumentação acadêmica.

## Fluxo de divulgação

```
projeto/
└── portfolio/
    ├── <slug>.qmd ──────────────────────► Github_Portfolio/projetos/<slug>.qmd
    ├── README.md   ────────────────────► usado no repositório GitHub do projeto
    ├── *.png       ────────────────────► copiadas junto com o .qmd
    └── research/   (opcional)
        ├── <slug>.qmd ──────────────────► Github_Portfolio/research/<slug>.qmd
        ├── <slug>_publico.pdf ─────────► Github_Portfolio/research/<slug>_publico.pdf
        └── <slug>_indice.png ──────────► Github_Portfolio/research/<slug>_indice.png
```

## Como replicar em projetos futuros

1. Criar `portfolio/` na raiz do novo projeto.
2. Copiar `sentimento-copom.qmd` daqui como ponto de partida; adaptar as 6 fases CRISP-DM ao novo projeto.
3. Adaptar `README.md` para o novo escopo.
4. Adicionar imagens de capa.
5. Copiar `<slug>.qmd` + imagens para `Github_Portfolio/projetos/`.
6. Commit e push em `vitorwilher.github.io`.
