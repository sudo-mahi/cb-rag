"""
Central configuration for cb-rag.
Every constant referenced by more than one file lives here — nothing
hardcoded or duplicated across chunker.py, indexer.py, daemon.py, cli.py.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# API keys from <project root>/.env into the environment. Nothing uses one
# right now (everything runs locally via Ollama), but any future API client
# will pick its key up from here, since every module imports config first.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Runtime data locations (Section 3: separate from project code) ──────────
CB_RAG_HOME = Path(os.path.expanduser("~/.cb-rag"))
CHROMA_DB_PATH = CB_RAG_HOME / "db"
REGISTRY_DB_PATH = CB_RAG_HOME / "registry.sqlite3"
LOGS_DIR = CB_RAG_HOME / "logs"

# Ensure these exist at import time — every entrypoint (daemon, CLI, indexer)
# imports this module, so this guarantees the dirs exist before anything
# tries to write to them.
CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Daemon ────────────────────────────────────────────────────────────────
DAEMON_HOST = "127.0.0.1"
DAEMON_PORT = 8787

# ── Models (Section 4: why each was chosen) ──────────────────────────────
EMBEDDING_MODEL = "qwen3-embedding:0.6b"  # local via Ollama (decision #34); outputs 1024 dims.
                                          # Changing it = full re-embed: each collection records
                                          # the model that built it and refuses queries from another.
EMBED_NUM_CTX = 8192     # tokens per text; longer chunks are embedded from their start
EMBED_BATCH_SIZE = 32    # texts per request — only for progress reporting, no API limit
# Qwen3-Embedding is asymmetric: queries get this instruction, documents don't.
EMBED_QUERY_INSTRUCTION = (
    "Given a question about a codebase, retrieve the code and file summaries that answer it"
)
# /ask runs on a local model via Ollama (decision #33): free, private, no
# rate limits. Largest coding model that fits 16 GB RAM; swap here.
OLLAMA_URL = "http://localhost:11434"
ASK_MODEL = "qwen2.5-coder:7b"
ASK_NUM_CTX = 16384     # Ollama's default window is small and silently drops the
                        # START of an over-long prompt (system prompt + top results)
ASK_MAX_TOKENS = 2048   # cap on the generated answer
COMPLETE_MODEL = "claude-haiku-5-5"  # Phase 2, not built yet; revisit with /complete

# ── Chunking / language support (Section 2, Section 16.1 step 5) ──────────
# Maps file extension -> tree-sitter-language-pack language identifier
LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".rb": "ruby",
}

# tree-sitter node types that count as "chunk-worthy" per language family.
# Not every language calls a function the same node type name.
CHUNKABLE_NODE_TYPES = {
    "function_definition",
    "class_definition",
    "method_definition",
    "function_declaration",
    "method_declaration",
}

# Top-level import node types, listed in file-summary chunks (Section 6).
IMPORT_NODE_TYPES = {
    "import_statement",        # python, javascript, typescript
    "import_from_statement",   # python
    "future_import_statement", # python
    "import_declaration",      # go, java
    "use_declaration",         # rust
    "preproc_include",         # c, cpp
}

# Files with no function chunks get their real content (capped) as the
# file-level chunk instead of a summary (decision #25).
FILE_FALLBACK_MAX_CHARS = 6000
FILE_DESCRIPTION_MAX_CHARS = 1000  # cap on the leading docstring/comment

# ── Default ignore patterns (Section 16.2) ─────────────────────────────────
DEFAULT_IGNORE_DIRS = {
    ".git", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", ".next", "target", "vendor", ".idea", ".vscode",
}
DEFAULT_IGNORE_FILE_SUFFIXES = (".lock", ".min.js")

# Name of the optional per-repo override file (Section 16.2)
RAGXIGNORE_FILENAME = ".ragxignore"

# ── Retrieval defaults (Section 10.2) ──────────────────────────────────────
DEFAULT_PER_LAYER_K = {
    "function": 6,
    "file": 3,
    "repo": 1,
}

# ── Repo-summary refresh threshold (Section 15.3) ──────────────────────────
REPO_SUMMARY_REFRESH_THRESHOLD = 0.10  # regenerate if >10% of files changed