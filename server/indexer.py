"""
Repo indexer: walk a repo, chunk every code file, embed only what changed,
and sync the result into Chroma.

Design decisions this file implements (see design doc):
- Decision #22: one Chroma collection per repo, so chunk ids stay
  filepath:qualified_name with no cross-repo collisions
- Decision #23: in git repos, `git ls-files` picks the files (respects the
  repo's .gitignore); non-git folders fall back to the default ignore lists
- Section 9: per-chunk hash diffing — added/modified chunks are embedded,
  removed ones deleted, unchanged ones skipped
- Function + file layers from the chunker; repo layer = one LLM summary
  per repo (decision #41), outside the per-chunk diff
"""

import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError

from server.chunker import chunk_file, text_lang
from server.config import (
    CHROMA_DB_PATH,
    EMBEDDING_MODEL,
    LANG_MAP,
    DEFAULT_IGNORE_DIRS,
    DEFAULT_IGNORE_FILE_SUFFIXES,
    REPO_SUMMARY_ID,
    REPO_SUMMARY_REFRESH_THRESHOLD,
)
from server.embeddings import embed_documents, ProgressFn
from server import registry, repo_summary


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

_chroma = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))


def collection_name(repo_name: str) -> str:
    """
    Chroma collection names must be 3-512 chars of [a-zA-Z0-9._-] and
    start/end with a letter or digit.
    """
    name = re.sub(r"[^a-zA-Z0-9._-]", "-", repo_name).strip("._-")
    return name.ljust(3, "0")


class EmbeddingModelMismatch(RuntimeError):
    pass


def _create_collection(name: str):
    # embedding_function=None: we always pass our own vectors, so Chroma
    # must never fall back to its default local model.
    return _chroma.create_collection(
        name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": EMBEDDING_MODEL},
    )


def get_collection(repo_name: str, rebuild_on_model_change: bool = False):
    """
    Vectors from different embedding models aren't comparable, and mixing
    them fails silently (meaningless scores). So each collection records the
    model that built it (decision #35): indexing rebuilds a stale collection
    from scratch; searching refuses it.
    """
    name = collection_name(repo_name)
    try:
        collection = _chroma.get_collection(name, embedding_function=None)
    except NotFoundError:
        return _create_collection(name)

    built_with = (collection.metadata or {}).get("embedding_model", "unknown (pre-#35)")
    if built_with == EMBEDDING_MODEL:
        return collection
    if rebuild_on_model_change:
        _chroma.delete_collection(name)
        return _create_collection(name)
    raise EmbeddingModelMismatch(
        f"Repo '{repo_name}' was indexed with {built_with}, but the configured "
        f"embedding model is {EMBEDDING_MODEL}. Re-index it."
    )


def _is_indexable(rel_path: str) -> bool:
    parts = Path(rel_path).parts
    if any(p in DEFAULT_IGNORE_DIRS for p in parts[:-1]):
        return False
    if rel_path.endswith(DEFAULT_IGNORE_FILE_SUFFIXES):
        return False
    return Path(rel_path).suffix in LANG_MAP or text_lang(Path(rel_path).name) is not None


def list_files(root: Path) -> list[str]:
    """Repo-relative paths of every file to index, sorted."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        candidates = [p for p in result.stdout.split("\0") if p]
    else:
        # Not a git repo: walk the tree, pruning ignored dirs in place so
        # os.walk never descends into them.
        candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORE_DIRS]
            for f in filenames:
                candidates.append(str(Path(dirpath, f).relative_to(root)))

    # --cached still lists files deleted on disk but not yet committed.
    return sorted(
        p for p in candidates
        if _is_indexable(p) and (root / p).is_file()
    )


def index_repo(
    path: str,
    repo_name: str | None = None,
    on_batch: ProgressFn | None = None,
    on_phase: Callable[[str], None] | None = None,
) -> dict:
    root = Path(path).expanduser().resolve()
    repo_name = repo_name or root.name
    # Before touching Chroma: raises RepoNameConflict if this name belongs
    # to a different folder.
    registry.register_repo(repo_name, str(root))
    collection = get_collection(repo_name, rebuild_on_model_change=True)

    # 1. Chunk every file (local and fast — only embedding costs money).
    current: dict[str, dict] = {}
    skipped: list[str] = []
    files = list_files(root)
    for rel in files:
        try:
            chunks = chunk_file(str(root / rel), rel_path=rel)
        except (UnicodeDecodeError, OSError) as e:
            skipped.append(f"{rel} ({type(e).__name__})")
            continue
        for c in chunks:
            c["filepath"] = rel
            current[c["chunk_id"]] = c

    # 2. What's already in Chroma for this repo: id -> stored hash. The repo
    # summary isn't a file chunk, so it stays out of the diff (or every run
    # would delete it).
    existing = collection.get(include=["metadatas"])
    stored = {
        id_: meta for id_, meta in zip(existing["ids"], existing["metadatas"])
        if id_ != REPO_SUMMARY_ID
    }
    has_summary = REPO_SUMMARY_ID in existing["ids"]

    # 3. Diff (Section 9).
    to_embed = [
        c for id_, c in current.items()
        if stored.get(id_, {}).get("content_hash") != c["content_hash"]  # added or modified
    ]
    to_delete = [id_ for id_ in stored if id_ not in current]
    changed_files = {c["filepath"] for c in to_embed} | {stored[id_]["filepath"] for id_ in to_delete}

    # 4. Embed only what changed, then write.
    if to_embed:
        vectors = embed_documents([c["text"] for c in to_embed], on_batch=on_batch)
        max_batch = _chroma.get_max_batch_size()
        for i in range(0, len(to_embed), max_batch):
            batch = to_embed[i:i + max_batch]
            collection.upsert(
                ids=[c["chunk_id"] for c in batch],
                embeddings=vectors[i:i + max_batch],
                documents=[c["text"] for c in batch],
                metadatas=[{
                    "filepath": c["filepath"],
                    "level": c["level"],
                    "kind": c["kind"],
                    "start_line": c["start_line"],
                    "end_line": c["end_line"],
                    "content_hash": c["content_hash"],
                    "node_type": c["node_type"],
                } for c in batch],
            )
    if to_delete:
        collection.delete(ids=to_delete)

    # 5. Repo summary (Section 15.3): on first index, or once >10% of files
    # have changed since it was written. A failure here doesn't fail the
    # index — the chunks are already written; it retries next run.
    changed_since = registry.get_repo(repo_name)["files_changed_since_summary"] + len(changed_files)
    summary_status = "kept"
    if not has_summary or (files and changed_since / len(files) > REPO_SUMMARY_REFRESH_THRESHOLD):
        if on_phase:
            on_phase("summarizing")
        try:
            file_chunks = {c["filepath"]: c for c in current.values() if c["level"] == "file"}
            text = repo_summary.generate(repo_name, root, files, file_chunks)
            collection.upsert(
                ids=[REPO_SUMMARY_ID],
                embeddings=embed_documents([text]),
                documents=[text],
                metadatas=[{
                    "filepath": "", "level": "repo", "kind": "summary",
                    "start_line": 0, "end_line": 0,
                    "content_hash": _hash(text), "node_type": "repo",
                }],
            )
            summary_status, changed_since = "generated", 0
        except Exception as e:
            summary_status = f"failed: {type(e).__name__}: {e}"

    # Chroma first, registry after (Section 8): a crash in between just
    # leaves last_indexed_at stale, never claiming work that didn't happen.
    registry.record_index(
        repo_name, file_count=len(files), chunk_count=len(current),
        files_changed_since_summary=changed_since,
        summary_regenerated=summary_status == "generated",
    )

    return {
        "repo": repo_name,
        "collection": collection.name,
        "files": len(files),
        "chunks": len(current),
        "embedded": len(to_embed),
        "deleted": len(to_delete),
        "unchanged": len(current) - len(to_embed),
        "skipped": skipped,
        "repo_summary": summary_status,
    }


if __name__ == "__main__":
    # Temporary entrypoint for testing until the daemon/CLI exist:
    #   uv run python3 -m server.indexer <path> [repo_name]
    print(index_repo(*sys.argv[1:]))
