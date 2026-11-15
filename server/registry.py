"""
SQLite registry of indexed repos.

Design decisions this file implements (see design doc):
- Decision #26: `repos` table only for now; the per-file table from
  Section 8 is deferred until a feature needs it (chunk-level facts
  already live in Chroma)
- Decision #27: a repo name already registered for a different path is
  refused, never shared — two folders in one collection would make the
  diff delete the other repo's chunks
- A fresh connection per call, so the daemon's background indexing
  thread never shares a connection with request handlers
"""

import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from server.config import REGISTRY_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS repos (
    name            TEXT PRIMARY KEY,
    path            TEXT NOT NULL,   -- absolute path to the repo root
    registered_at   TEXT NOT NULL,   -- ISO timestamp, UTC
    last_indexed_at TEXT,            -- NULL until the first index completes
    file_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    summary_generated_at        TEXT,                        -- repo summary (Section 15)
    files_changed_since_summary INTEGER NOT NULL DEFAULT 0   -- for the 10% refresh rule
)
"""

# Columns added after the table first shipped: (name, SQL type + default).
# CREATE TABLE IF NOT EXISTS doesn't touch an existing table, so add them.
_ADDED_COLUMNS = [
    ("summary_generated_at", "TEXT"),
    ("files_changed_since_summary", "INTEGER NOT NULL DEFAULT 0"),
]


class RepoNameConflict(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(REGISTRY_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    have = {row["name"] for row in conn.execute("PRAGMA table_info(repos)")}
    for name, decl in _ADDED_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE repos ADD COLUMN {name} {decl}")
    return conn


def get_repo(name: str) -> dict | None:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM repos WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def list_repos() -> list[dict]:
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM repos ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def register_repo(name: str, path: str) -> None:
    """Register on first index; no-op if already registered for this path."""
    existing = get_repo(name)
    if existing:
        if existing["path"] != path:
            raise RepoNameConflict(
                f"Repo name '{name}' is already registered for {existing['path']}. "
                f"Pass a different name to index {path}."
            )
        return
    with closing(_connect()) as conn, conn:  # inner `conn` = commit on success
        conn.execute(
            "INSERT INTO repos (name, path, registered_at) VALUES (?, ?, ?)",
            (name, path, _now()),
        )


def record_index(
    name: str,
    file_count: int,
    chunk_count: int,
    files_changed_since_summary: int,
    summary_regenerated: bool,
) -> None:
    """Call only after the Chroma writes succeeded (Section 8 sync discipline)."""
    now = _now()
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE repos SET last_indexed_at = ?, file_count = ?, chunk_count = ?, "
            "files_changed_since_summary = ? WHERE name = ?",
            (now, file_count, chunk_count, files_changed_since_summary, name),
        )
        if summary_regenerated:
            conn.execute("UPDATE repos SET summary_generated_at = ? WHERE name = ?", (now, name))
