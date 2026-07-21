"""Embeddings para o retrieval denso — provedor via `.env` ou fallback local.

Estratégia de degradação graciosa:

1. **OpenAI** (`OPENAI_API_KEY`) — `text-embedding-3-small`, se disponível;
2. **Google** (`GOOGLE_API_KEY`) — `text-embedding-004`, se disponível;
3. **Fallback local** — embedding determinístico por *hashing trick* + TF, sem
   dependência externa nem download de modelo. Suficiente para o retrieval denso
   funcionar em CI/offline; troque por um provedor para qualidade de produção.

Todos os vetores são retornados normalizados implicitamente pela similaridade de
cosseno no retriever, então a escala do fallback não importa.
"""

from __future__ import annotations

import hashlib
import os
import re

_TOKEN_RE = re.compile(r"[0-9a-zà-ÿ]+", re.IGNORECASE)
_LOCAL_DIM = 512  # dimensão do embedding local por hashing


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Gera embeddings para uma lista de textos.

    Tenta provedores em ordem de preferência e cai para o embedding local se
    nenhum estiver configurado ou se a chamada falhar.

    Args:
        texts: textos a codificar.

    Returns:
        Lista de vetores (um por texto).
    """
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _embed_openai(texts)
        except Exception:  # noqa: BLE001 — degrada para o próximo provedor
            pass
    if os.getenv("GOOGLE_API_KEY"):
        try:
            return _embed_google(texts)
        except Exception:  # noqa: BLE001
            pass
    return [_embed_local(t) for t in texts]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    """Embeddings via OpenAI `text-embedding-3-small`."""
    from openai import OpenAI

    client = OpenAI()
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


def _embed_google(texts: list[str]) -> list[list[float]]:
    """Embeddings via Google `text-embedding-004`."""
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    out: list[list[float]] = []
    for t in texts:
        r = genai.embed_content(model="models/text-embedding-004", content=t)
        out.append(r["embedding"])
    return out


def _embed_local(text: str) -> list[float]:
    """Embedding local determinístico via *hashing trick* + frequência de termos.

    Cada token é mapeado para um índice fixo por hash (md5) e acumula sua
    frequência. Não é semântico como um modelo neural, mas captura co-ocorrência
    de termos de forma estável e barata — o bastante para o denso complementar o
    BM25 no fallback offline.

    Args:
        text: texto a codificar.

    Returns:
        Vetor de dimensão `_LOCAL_DIM`.
    """
    vec = [0.0] * _LOCAL_DIM
    for tok in _TOKEN_RE.findall(text.lower()):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % _LOCAL_DIM] += 1.0
    return vec
