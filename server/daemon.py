"""
FastAPI daemon: holds Chroma + API clients so the CLI stays a thin HTTP client.

Design decisions this file implements (see design doc):
- Section 14: /health, /repos, /index, /index/status, /query, /ask
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
from server.indexer import index_repo
from server.retrieval import search

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/repos")
def repos():
    return {"repos": registry.list_repos()}


def _run_index_job(job: dict, path: str) -> None:
    def on_batch(done: int, total: int):
        job["batches_done"], job["batches_total"] = done, total

    try:
        job["result"] = index_repo(path, job["repo"], on_batch=on_batch)
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
            # Trim before announcing sources, so they match what the model sees.
            results, dropped = fit_to_context(req.question, results)
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
