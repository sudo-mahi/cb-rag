"""
Query-time retrieval: classify the question, search the routed layers.

Design decisions this file implements (see design doc):
- Section 10.1: cheap heuristic classifier (no LLM) returning a *set* of
  layers; no signal -> {function, file}
- Section 10.2: one Chroma search per routed layer, per-layer top_k
- Section 10.3: no score filtering yet — everything a routed layer returns
- Section 11.1: layers ordered broad -> narrow (repo, file, function)
"""

import re

from server.config import DEFAULT_PER_LAYER_K, LANG_MAP, REPO_SUMMARY_ID
from server.embeddings import embed_query
from server.indexer import get_collection

LAYER_ORDER = ["repo", "file", "function"]

_IDENTIFIER = re.compile(
    r"`[^`]+`"                      # backticked term
    r"|\b[a-z0-9]+_[a-z0-9_]+\b"    # snake_case
    r"|\b[a-z]+[A-Z]\w*\b"          # camelCase
    r"|\b[A-Z][a-z0-9]+[A-Z]\w*\b"  # PascalCase with 2+ humps
    r"|\b\w+\(\)"                   # call-like: foo()
)
_FILE_EXT = re.compile(
    r"\w\.(" + "|".join(ext.lstrip(".") for ext in LANG_MAP) + r")\b"
)
_FILE_WORDS = re.compile(r"\b(files?|modules?)\b", re.IGNORECASE)
_PATH_LIKE = re.compile(r"\w/\w")
_REPO_WORDS = re.compile(
    r"\b(architecture|structure[ds]?|overview|high[- ]level|design|"
    r"fits? together|flow between)\b",
    re.IGNORECASE,
)


def classify(question: str) -> list[str]:
    """Which layers to search, in broad -> narrow order."""
    layers = set()
    if _IDENTIFIER.search(question):
        layers.add("function")
    if _FILE_EXT.search(question) or _FILE_WORDS.search(question) or _PATH_LIKE.search(question):
        layers.add("file")
    if _REPO_WORDS.search(question):
        layers.add("repo")
    if not layers:
        layers = {"function", "file"}
    return [layer for layer in LAYER_ORDER if layer in layers]


def get_repo_summary(repo_name: str) -> dict | None:
    """
    The repo-level summary as a result dict (decision #41). Fetched by id,
    not by similarity: /ask always includes it.
    """
    r = get_collection(repo_name).get(ids=[REPO_SUMMARY_ID], include=["documents"])
    if not r["ids"]:
        return None
    return {
        "chunk_id": REPO_SUMMARY_ID, "text": r["documents"][0], "filepath": "",
        "start_line": 0, "end_line": 0, "layer": "repo", "kind": "summary", "score": 1.0,
    }


def search(
    repo_name: str,
    query: str,
    layers: list[str] | None = None,
    top_k: dict[str, int] | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Returns (layers_searched, results). `layers` bypasses the classifier;
    `top_k` overrides the per-layer defaults (Section 14.2).
    """
    layers = [l for l in LAYER_ORDER if l in layers] if layers else classify(query)
    k = {**DEFAULT_PER_LAYER_K, **(top_k or {})}

    collection = get_collection(repo_name)
    query_vector = embed_query(query)

    results = []
    for layer in layers:
        r = collection.query(
            query_embeddings=[query_vector],
            n_results=k[layer],
            where={"level": layer},
            include=["documents", "metadatas", "distances"],
        )
        for id_, text, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            results.append({
                "chunk_id": id_,
                "text": text,
                "filepath": meta["filepath"],
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
                "layer": layer,
                "kind": meta["kind"],
                "score": round(1 - dist, 4),  # cosine distance -> similarity
            })
    return layers, results
