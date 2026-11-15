"""
/complete: single-line inline suggestions from a local fill-in-the-middle model.

Design decisions this file implements (see design doc):
- Section 12.1: a function-level search from the code before the cursor,
  plus the current file's summary chunk; no question classifier
- Decision #43 (measured): Qwen repo-level prompt format, context =
  current file's outline + top-3 similar functions from OTHER files
  (33% exact vs 23% without context on 60 real lines)
- Context cached per (file, 25-line region), and the in-file prefix starts
  at a fixed line per region, so consecutive keystrokes send a prompt whose
  start is byte-identical: Ollama's prompt cache then reuses it (~0.3s
  while typing vs ~0.5-0.9s cold)
"""

import threading
import time
from collections import OrderedDict
from pathlib import Path

import requests

from server import registry
from server.config import (
    OLLAMA_URL,
    COMPLETE_MODEL,
    COMPLETE_NUM_CTX,
    COMPLETE_MAX_TOKENS,
    COMPLETE_KEEP_ALIVE,
    COMPLETE_REGION_LINES,
    COMPLETE_PREFIX_LINES,
    COMPLETE_SUFFIX_CHARS,
    COMPLETE_SNIPPET_CHARS,
    COMPLETE_K,
    COMPLETE_CACHE_SIZE,
)
from server.embeddings import embed_documents
from server.indexer import get_collection

_cache: OrderedDict[tuple, list[tuple[str, str]]] = OrderedDict()
_cache_lock = threading.Lock()


def _locate(filepath: str) -> tuple[str, str] | None:
    """(repo name, path relative to its root) for the deepest indexed repo containing the file."""
    path = Path(filepath).resolve()
    matches = [r for r in registry.list_repos() if path.is_relative_to(r["path"])]
    if not matches:
        return None
    repo = max(matches, key=lambda r: len(r["path"]))
    return repo["name"], str(path.relative_to(repo["path"]))


def _retrieve(repo: str, rel: str, query: str) -> list[tuple[str, str]]:
    """[(label, text)]: this file's outline, then similar functions from other files."""
    col = get_collection(repo)
    # Code -> code similarity is symmetric, so embed the query as a document
    # (the query instruction is worded for questions, not code).
    vec = embed_documents([query])[0]
    r = col.query(query_embeddings=[vec], n_results=COMPLETE_K + 10, where={"level": "function"},
                  include=["documents", "metadatas"])
    # Chunks from the current file are excluded: its live text is already in
    # the prompt, and the indexed copy may be stale.
    entries = [(m["filepath"], d[:COMPLETE_SNIPPET_CHARS])
               for d, m in zip(r["documents"][0], r["metadatas"][0]) if m["filepath"] != rel][:COMPLETE_K]
    outline = col.get(ids=[f"{rel}:<file>"], include=["documents"])
    if outline["ids"]:
        entries.insert(0, (f"{rel} (outline of this file)", outline["documents"][0][:COMPLETE_SNIPPET_CHARS]))
    return entries


def _context(repo: str, rel: str, region: int, query: str) -> tuple[list[tuple[str, str]], bool]:
    """Cached context for (file, region); returns (entries, was_cached)."""
    key = (repo, rel, region)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key], True
    entries = _retrieve(repo, rel, query)
    with _cache_lock:
        _cache[key] = entries
        while len(_cache) > COMPLETE_CACHE_SIZE:
            _cache.popitem(last=False)
    return entries, False


def complete(filepath: str, prefix: str, suffix: str, cursor_line: int) -> dict:
    """
    `prefix` / `suffix`: the whole document before / after the cursor.
    `cursor_line`: 0-based line of the cursor.
    """
    t0 = time.time()
    region = cursor_line // COMPLETE_REGION_LINES
    prefix_lines = prefix.splitlines(keepends=True)
    if prefix.endswith(("\n", "\r")):
        prefix_lines.append("")  # cursor at the start of an empty line
    # Fixed start per region: within a region the in-file prefix only grows
    # at its end, so the prompt before the cursor stays cacheable.
    start = max(0, region * COMPLETE_REGION_LINES - COMPLETE_PREFIX_LINES)
    file_prefix = "".join(prefix_lines[start:])
    file_suffix = suffix[:COMPLETE_SUFFIX_CHARS]

    located = _locate(filepath)
    entries, cached, context = [], False, "none"
    if located:
        repo, rel = located
        # Search query = the 15 lines before the cursor. Only used when the
        # region's context isn't cached yet; after that the region keeps it.
        query = "".join(prefix_lines[-15:])
        entries, cached = _context(repo, rel, region, query)
        context = "cached" if cached else "fresh"
        files = "".join(f"<|file_sep|>{label}\n{text}\n" for label, text in entries)
        head = f"<|repo_name|>{repo}\n{files}<|file_sep|>{rel}\n"
    else:
        head = ""
    t1 = time.time()

    prompt = f"{head}<|fim_prefix|>{file_prefix}<|fim_suffix|>{file_suffix}<|fim_middle|>"
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": COMPLETE_MODEL,
                "prompt": prompt,
                "raw": True,  # we build Qwen's FIM tokens ourselves (repo format)
                "stream": False,
                "keep_alive": COMPLETE_KEEP_ALIVE,
                "options": {
                    "temperature": 0,
                    "num_predict": COMPLETE_MAX_TOKENS,
                    "num_ctx": COMPLETE_NUM_CTX,
                    "stop": ["\n", "<|file_sep|>", "<|endoftext|>", "<|fim_pad|>"],
                },
            },
            timeout=(2, 30),
        )
    except requests.ConnectionError:
        raise RuntimeError(f"Ollama isn't running at {OLLAMA_URL}. Start it: brew services start ollama")
    resp.raise_for_status()
    completion = resp.json()["response"].rstrip()
    # Cursor after only indentation: the model sometimes re-indents, which
    # would double the indentation once inserted at the cursor.
    if not prefix_lines[-1].strip():
        completion = completion.lstrip()
    t2 = time.time()

    return {
        "completion": completion,
        "context": context,
        "ms": {"context": round((t1 - t0) * 1000), "generate": round((t2 - t1) * 1000),
               "total": round((t2 - t0) * 1000)},
    }
