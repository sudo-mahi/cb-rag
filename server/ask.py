"""
/ask generation: turn retrieved chunks into a prompt and stream Claude's answer.

Design decisions this file implements (see design doc):
- Section 11.1: broad -> narrow (repo context, then per-file summary, then
  functions)
- Section 11.2: grouped by file; `### name (lines X-Y)` headings double as
  the citation format
- Section 11.3: summaries labeled as auto-generated, never shown as code
  (driven by each chunk's `kind`)
- Decision #31: streamed; this module yields events, the daemon sends them
  as NDJSON
- Decision #33: answered by a local model through Ollama's /api/chat
"""

import json
from collections.abc import Iterator
from pathlib import Path

import requests

from server.config import OLLAMA_URL, ASK_MODEL, ASK_NUM_CTX, ASK_MAX_TOKENS, LANG_MAP

SYSTEM_PROMPT = """\
You answer questions about the user's own codebase. The user message contains \
context retrieved from the repository, followed by the question.

Base the answer on that context. When you refer to code, cite it as \
`path:start-end` using the line numbers in the headings. If the context \
doesn't contain what's needed to answer, say what's missing instead of \
guessing; general programming knowledge is fine for explaining what the \
code does, but not for claiming what it contains.

Sections marked "Summary (auto-generated)" are heuristic outlines of a file \
(its imports and signatures), not its source — use them for orientation, and \
don't quote them as code."""


def _code_block(text: str, filepath: str) -> str:
    lang = LANG_MAP.get(Path(filepath).suffix, "")
    return f"```{lang}\n{text}\n```"


def build_context(results: list[dict]) -> str:
    """
    Format retrieved chunks per Section 11.2. Files are ordered by their
    best-scoring chunk; within a file, the file-level chunk comes first,
    then functions in source order.
    """
    repo_chunks = [r for r in results if r["layer"] == "repo"]
    by_file: dict[str, list[dict]] = {}
    for r in sorted(results, key=lambda r: -r["score"]):
        if r["layer"] != "repo":
            by_file.setdefault(r["filepath"], []).append(r)

    parts = []
    if repo_chunks:
        parts.append("## Repository Context")
        parts += [f"_Summary (auto-generated):_\n{r['text']}" for r in repo_chunks]

    for filepath, chunks in by_file.items():
        parts.append(f"## File: {filepath}")
        for c in sorted(chunks, key=lambda c: (c["layer"] != "file", c["start_line"])):
            if c["kind"] == "summary":
                parts.append(f"_Summary (auto-generated):_\n{c['text']}")
            elif c["layer"] == "file":
                # File without functions: its real content (decision #25).
                parts.append(f"### Entire file (lines {c['start_line']}-{c['end_line']})\n"
                             + _code_block(c["text"], filepath))
            else:
                name = c["chunk_id"][len(filepath) + 1:]
                parts.append(f"### {name} (lines {c['start_line']}-{c['end_line']})\n"
                             + _code_block(c["text"], filepath))

    return "\n\n".join(parts)


def stream_answer(question: str, results: list[dict]) -> Iterator[dict]:
    """Yields {"type": "text", "text": ...} events, then {"type": "done", ...}."""
    if not results:
        yield {"type": "text", "text": "Nothing in the index matched this question."}
        yield {"type": "done", "stop_reason": "no_context"}
        return

    user_message = f"{build_context(results)}\n\n---\n\nQuestion: {question}"

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": ASK_MODEL,
                "stream": True,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "options": {"num_ctx": ASK_NUM_CTX, "num_predict": ASK_MAX_TOKENS},
            },
            stream=True,
            timeout=(5, 300),  # (connect, read between chunks)
        )
    except requests.ConnectionError:
        raise RuntimeError(f"Ollama isn't running at {OLLAMA_URL}. Start it: brew services start ollama")
    resp.raise_for_status()

    # Ollama streams NDJSON: {"message": {"content": "..."}, "done": false} ...
    # ending with a {"done": true, "done_reason": ...} line.
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        if "error" in chunk:
            raise RuntimeError(f"Ollama: {chunk['error']}")
        if text := chunk.get("message", {}).get("content"):
            yield {"type": "text", "text": text}
        if chunk.get("done"):
            yield {"type": "done", "stop_reason": chunk.get("done_reason")}
            return
