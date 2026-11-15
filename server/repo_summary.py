"""
Repo-level summary: one LLM-written overview per repo.

Design decisions this file implements (see design doc):
- Section 15 / decision #41: generated during indexing, with the local
  answer model, from (in priority order) the manifest description, README,
  folder tree, and one line per file (path + first line of its
  description), all within a fixed character budget
- Section 11.3: it's a paraphrase, so it's stored as kind "summary" and
  labeled as auto-generated wherever it's shown to a model
"""

import json
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath

import requests

from server.config import (
    OLLAMA_URL,
    ASK_MODEL,
    ASK_NUM_CTX,
    REPO_SUMMARY_README_CHARS,
    REPO_SUMMARY_TREE_CHARS,
    REPO_SUMMARY_INPUT_CHARS,
    REPO_SUMMARY_MAX_TOKENS,
)

SUMMARY_PROMPT = """\
You write the overview of a code repository for a developer seeing it for \
the first time. Using only the inputs provided, describe what the project \
does, its main components and how they fit together, how it is run or \
deployed, and where the key code lives (with file paths). Leave out \
anything the inputs don't state; don't guess. Under 300 words: one short \
paragraph, then bullet points."""


def _readme(root: Path) -> str:
    for p in sorted(root.iterdir()):
        if p.is_file() and p.name.lower().startswith("readme"):
            return p.read_text(errors="ignore")[:REPO_SUMMARY_README_CHARS]
    return ""


def _manifest_description(root: Path) -> str:
    """The one-line description a project gives itself, if it has one."""
    try:
        if (p := root / "pyproject.toml").is_file():
            return tomllib.loads(p.read_text()).get("project", {}).get("description", "")
        if (p := root / "package.json").is_file():
            return json.loads(p.read_text()).get("description", "")
        if (p := root / "Cargo.toml").is_file():
            return tomllib.loads(p.read_text()).get("package", {}).get("description", "")
    except (ValueError, OSError):
        pass  # malformed manifest: just skip this input
    return ""


def _tree(files: list[str]) -> str:
    """Folders up to 3 levels deep, with how many indexed files each holds."""
    counts: Counter[str] = Counter()
    for f in files:
        parts = PurePosixPath(f).parts[:-1]
        for depth in range(1, min(len(parts), 3) + 1):
            counts["/".join(parts[:depth])] += 1
    lines = [f"{'  ' * d.count('/')}{d.split('/')[-1]}/ ({n} files)" for d, n in sorted(counts.items())]
    top_level = sum(1 for f in files if "/" not in f)
    return (f"(root: {top_level} files)\n" + "\n".join(lines))[:REPO_SUMMARY_TREE_CHARS]


def _first_meaningful_line(text: str) -> str:
    """
    The first line of a file chunk that says something: a docstring's
    opening line, a Markdown title, a config file's first entry.
    """
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("File:"):
            continue
        if s.startswith(("Imports:", "Defines:")):
            return ""  # heuristic summary with no description
        s = s.removeprefix("Description:").removeprefix("Sections:")
        s = s.strip().strip("\"'#/*-` ").strip()
        if len(s) >= 3 and any(ch.isalpha() for ch in s):
            return s[:100]
    return ""


def build_inputs(repo_name: str, root: Path, files: list[str], file_chunks: dict[str, dict]) -> str:
    parts = [f"# Repository: {repo_name}"]
    if desc := _manifest_description(root):
        parts.append(f"## Manifest description\n{desc}")
    if readme := _readme(root):
        parts.append(f"## README (may be truncated)\n{readme}")
    parts.append(f"## Folder tree\n{_tree(files)}")

    header = "## Files (path: first line of its description)"
    used = sum(len(p) for p in parts) + len(header)
    file_lines = []
    for rel in files:
        chunk = file_chunks.get(rel)
        desc = _first_meaningful_line(chunk["text"]) if chunk else ""
        line = f"{rel}: {desc}" if desc else rel
        if used + len(line) + 1 > REPO_SUMMARY_INPUT_CHARS:
            file_lines.append(f"... and {len(files) - len(file_lines)} more files")
            break
        file_lines.append(line)
        used += len(line) + 1
    parts.append(header + "\n" + "\n".join(file_lines))
    return "\n\n".join(parts)


def generate(repo_name: str, root: Path, files: list[str], file_chunks: dict[str, dict]) -> str:
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": ASK_MODEL,
                "stream": False,
                "messages": [
                    {"role": "system", "content": SUMMARY_PROMPT},
                    {"role": "user", "content": build_inputs(repo_name, root, files, file_chunks)},
                ],
                "options": {"num_ctx": ASK_NUM_CTX, "num_predict": REPO_SUMMARY_MAX_TOKENS},
            },
            timeout=(5, 600),
        )
    except requests.ConnectionError:
        raise RuntimeError(f"Ollama isn't running at {OLLAMA_URL}. Start it: brew services start ollama")
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama chat failed ({resp.status_code}): {resp.text}")
    return resp.json()["message"]["content"].strip()
