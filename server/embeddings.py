"""
Local embeddings via Ollama.

Design decisions this file implements (see design doc):
- Decision #34: qwen3-embedding:0.6b through Ollama's /api/embed, replacing
  Voyage — free, private, no rate limits
- Asymmetric retrieval: documents (code chunks) are embedded as-is; queries
  get Qwen3-Embedding's instruction prefix. Two separate functions so the
  two can't be mixed up by accident.
- Fixed-size batches: no API limits to pack against locally; batches exist
  only so indexing can report progress
"""

from collections.abc import Callable

import requests

from server.config import (
    OLLAMA_URL,
    EMBEDDING_MODEL,
    EMBED_NUM_CTX,
    EMBED_BATCH_SIZE,
    EMBED_QUERY_INSTRUCTION,
)

ProgressFn = Callable[[int, int], None]  # (batches_done, batches_total)


def _embed(texts: list[str]) -> list[list[float]]:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": texts,
                # Texts longer than the window are cut to their start rather
                # than failing; the full text is still stored in Chroma.
                "truncate": True,
                "options": {"num_ctx": EMBED_NUM_CTX},
            },
            timeout=(5, 300),
        )
    except requests.ConnectionError:
        raise RuntimeError(f"Ollama isn't running at {OLLAMA_URL}. Start it: brew services start ollama")
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama embed failed ({resp.status_code}): {resp.text}")
    return resp.json()["embeddings"]


def embed_documents(texts: list[str], on_batch: ProgressFn | None = None) -> list[list[float]]:
    """
    Embed code chunks for indexing. vectors[i] belongs to texts[i].
    `on_batch(done, total)` is called after each batch, for progress.
    """
    batches = [texts[i:i + EMBED_BATCH_SIZE] for i in range(0, len(texts), EMBED_BATCH_SIZE)]
    vectors: list[list[float]] = []
    for n, batch in enumerate(batches, start=1):
        vectors.extend(_embed(batch))
        if on_batch:
            on_batch(n, len(batches))
    return vectors


def embed_query(text: str) -> list[float]:
    """Embed a single user question for search."""
    return _embed([f"Instruct: {EMBED_QUERY_INSTRUCTION}\nQuery: {text}"])[0]
