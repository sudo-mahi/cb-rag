"""
FastAPI daemon: holds Chroma and talks to Ollama, so the CLI stays a thin HTTP client.

Design decisions this file implements (see design doc):
- Section 14: /health, /repos, /index, /index/status, /query, /ask, plus
  /summary (decision #41) and /complete (Phase 2, decision #43)
- Decision #28: a second /index for a repo that's already indexing gets
  the running job back, so two jobs never diff one collection at once
- Decision #29: /index/status reports embedding batch progress
- Endpoints are plain `def`: Chroma and Ollama calls block, so FastAPI runs
  them in its thread pool instead of stalling the event loop

Run (from the project root):
    uv run uvicorn server.daemon:app --host 127.0.0.1 --port 8787
"""

import json
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server import registry
from server.ask import fit_to_context, stream_answer
from server.complete import complete
from server.indexer import index_repo
from server.retrieval import get_repo_summary, search

app = FastAPI(title="cb-rag")

# In memory: lost on restart, which is fine — indexing can always be re-run.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

Layer = Literal["function", "file", "repo"]


class IndexRequest(BaseModel):
    path: str
    repo: str | None = None


class QueryRequest(BaseModel):
    repo: str
    query: str
    layers: list[Layer] | None = None
    top_k: dict[Layer, int] | None = None


class AskRequest(BaseModel):
    repo: str
    question: str
    top_k: dict[Layer, int] | None = None


class CompleteRequest(BaseModel):
    filepath: str        # absolute path of the file being edited
    prefix: str          # whole document before the cursor
    suffix: str          # whole document after the cursor
    cursor_line: int     # 0-based


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/repos")
def repos():
    return {"repos": registry.list_repos()}


def _run_index_job(job: dict, path: str) -> None:
    def on_batch(done: int, total: int):
        job["batches_done"], job["batches_total"] = done, total

    def on_phase(phase: str):
        job["phase"] = phase

    try:
        job["result"] = index_repo(path, job["repo"], on_batch=on_batch, on_phase=on_phase)
        job["status"] = "done"
    except Exception as e:
        job["error"] = f"{type(e).__name__}: {e}"
        job["status"] = "failed"


@app.post("/index")
def start_index(req: IndexRequest):
    root = Path(req.path).expanduser().resolve()
    if not root.is_dir():
        raise HTTPException(400, f"Not a directory: {root}")
    repo_name = req.repo or root.name

    # Check the name up front so a clash is a 409 now, not a failed job later.
    try:
        registry.register_repo(repo_name, str(root))
    except registry.RepoNameConflict as e:
        raise HTTPException(409, str(e))

    with _jobs_lock:
        for job in _jobs.values():
            if job["repo"] == repo_name and job["status"] == "running":
                return {"job_id": job["job_id"], "status": "already_running"}

        job = {
            "job_id": f"idx_{uuid.uuid4().hex[:6]}",
            "repo": repo_name,
            "status": "running",
            "phase": "indexing",    # -> "summarizing" while the repo summary is written
            "batches_done": 0,
            "batches_total": None,  # known once chunking + diffing finish
            "result": None,
            "error": None,
        }
        _jobs[job["job_id"]] = job

    threading.Thread(target=_run_index_job, args=(job, str(root)), daemon=True).start()
    return {"job_id": job["job_id"], "status": "started"}


@app.get("/index/status")
def index_status(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id: {job_id}")
    return job


@app.post("/query")
def query(req: QueryRequest):
    if registry.get_repo(req.repo) is None:
        raise HTTPException(404, f"Unknown repo: {req.repo}. Index it first.")
    layers, results = search(req.repo, req.query, req.layers, req.top_k)
    return {"layers_searched": layers, "results": results}


@app.post("/complete")
def complete_endpoint(req: CompleteRequest):
    """
    Single-line inline suggestion (Section 12, decision #43). The repo is
    inferred from the file's path; files outside indexed repos still get a
    suggestion, just without repo context.
    """
    return complete(req.filepath, req.prefix, req.suffix, req.cursor_line)


@app.get("/summary")
def summary(repo: str):
    """The stored repo-level summary, so it can be read and checked (#41)."""
    info = registry.get_repo(repo)
    if info is None:
        raise HTTPException(404, f"Unknown repo: {repo}. Index it first.")
    s = get_repo_summary(repo)
    return {"repo": repo, "summary": s["text"] if s else None, "generated_at": info["summary_generated_at"]}


@app.post("/ask")
def ask(req: AskRequest):
    """
    Streams NDJSON (decision #31), one event per line:
      {"type": "sources", "layers_searched": [...], "sources": [...]}
      {"type": "text", "text": "..."}            (repeated)
      {"type": "done", "stop_reason": "..."}     or {"type": "error", "error": "..."}
    """
    if registry.get_repo(req.repo) is None:
        raise HTTPException(404, f"Unknown repo: {req.repo}. Index it first.")

    def events():
        # The 200 status is already sent once streaming starts, so failures
        # from here on are reported in-band as an error event.
        try:
            layers, results = search(req.repo, req.question, top_k=req.top_k)
            # The repo summary is always included at the top (#41) — fetched
            # by id, so drop it from search results to avoid sending it twice.
            results = [r for r in results if r["layer"] != "repo"]
            pinned = [s] if (s := get_repo_summary(req.repo)) else []
            # Trim before announcing sources, so they match what the model sees.
            results, dropped = fit_to_context(req.question, results, pinned)
            results = pinned + results
            yield {
                "type": "sources",
                "layers_searched": layers,
                "dropped": dropped,
                "sources": [
                    {k: r[k] for k in ("filepath", "start_line", "end_line", "layer", "kind", "score")}
                    for r in results
                ],
            }
            yield from stream_answer(req.question, results)
        except Exception as e:
            yield {"type": "error", "error": f"{type(e).__name__}: {e}"}

    return StreamingResponse(
        (json.dumps(e) + "\n" for e in events()),
        media_type="application/x-ndjson",
    )
