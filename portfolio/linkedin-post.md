# Post para LinkedIn — COPOM RAG Service

> **Como publicar:** cole o texto abaixo em um novo post e **anexe a imagem**
> `copom-rag-service-capa.png` (esta pasta). O LinkedIn só aceita imagens raster
> (PNG/JPG) — por isso o `.png`, e não o `.svg`. Para editar o diagrama, abra
> `copom-rag-service-arquitetura.excalidraw` em <https://excalidraw.com> e
> exporte um novo PNG (File → Export image → PNG, 2x).

---

## Versão principal (recomendada)

Todo mundo sabe montar um RAG. Poucos sabem provar que ele não regride. 👇

Construí o esqueleto do **COPOM RAG Service**: um RAG sobre as **atas do Copom** e
o **boletim Focus** do Banco Central, servido como **API** (FastAPI + Docker).
Você pergunta sobre política monetária em linguagem natural e recebe uma resposta
fundamentada — **com as fontes citadas**.

Mas o diferencial não é o retrieval. É a engenharia de qualidade em volta dele:

🔹 **Eval harness** — golden set de perguntas reais do Copom + LLM-as-judge +
um **gate de CI que FALHA (exit 1)** se a qualidade média cai abaixo de um
threshold. Qualidade virou contrato verificável, não "achismo".

🔹 **Observability** — custo, latência e tokens medidos por request.

🔹 **API + Docker** — `POST /ask` de pé com um comando.

Num domínio onde um número errado (Selic, meta de inflação, projeção) é uma falha
grave, um RAG não se prova por uma demo bonita — prova-se por **não regredir**.

Detalhe que eu adoro: enquanto o pipeline é stub, o gate **reprova de propósito**.
Um sistema vazio *tem* que falhar o teste. É exatamente o que se espera de um bom
test harness.

Projeto em desenvolvimento aberto — ficha técnica (CRISP-DM) e código no primeiro
comentário. 👇

\#RAG #LLM #MLOps #FastAPI #Python #Anthropic #Claude #AIEngineering #MonetaryPolicy #Copom #BancoCentral #DataScience #LLMOps

---

## Versão curta (alternativa)

RAG é commodity. O que separa protótipo de produção é **não deixar a qualidade
regredir em silêncio**.

No **COPOM RAG Service** (RAG sobre atas do Copom + Focus, servido como API), o
código-âncora não é o retrieval — é o **eval harness**: golden set +
LLM-as-judge + um **gate de CI que falha (exit 1)** quando a resposta piora.
Mais custo/latência/tokens por request e empacotamento em Docker.

Ficha técnica e código nos comentários. 👇

\#LLM #RAG #MLOps #AIEngineering #Python #Copom

---

## Primeiro comentário (para colar em ambas)

📄 Ficha técnica (CRISP-DM): https://vitorwilher.github.io/copom-rag-service/
💻 Código: https://github.com/vitorwilher/copom-rag-service
