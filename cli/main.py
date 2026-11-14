"""
ragx: thin terminal client over the cb-rag daemon.

Design decisions this file implements (see design doc):
- Section 3: no indexing/search logic here; every command is an HTTP call
  to the daemon on localhost
- Decision #38: command name `ragx`; repo inferred from the current folder
  (deepest registered repo containing it), `--repo` overrides; daemon not
  running -> error showing the exact start command

Install (from the project root):  uv tool install --editable .
"""

import json
import time
from pathlib import Path

import click
import requests

from server.config import DAEMON_HOST, DAEMON_PORT

DAEMON_URL = f"http://{DAEMON_HOST}:{DAEMON_PORT}"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_START_HINT = (
    "The cb-rag daemon isn't running. Start it in another terminal:\n"
    f"  cd {_PROJECT_ROOT} && uv run uvicorn server.daemon:app "
    f"--host {DAEMON_HOST} --port {DAEMON_PORT}"
)


def _request(method: str, path: str, **kwargs) -> requests.Response:
    try:
        resp = requests.request(method, f"{DAEMON_URL}{path}", timeout=(3, 600), **kwargs)
    except requests.ConnectionError:
        raise click.ClickException(_START_HINT)
    if resp.status_code >= 400:
        try:
            detail = resp.json()["detail"]
        except (ValueError, KeyError):
            detail = resp.text
        raise click.ClickException(str(detail))
    return resp


def _resolve_repo(repo: str | None) -> str:
    """--repo if given, else the deepest registered repo containing the cwd."""
    if repo:
        return repo
    cwd = Path.cwd().resolve()
    matches = [
        r for r in _request("GET", "/repos").json()["repos"]
        if cwd.is_relative_to(r["path"])
    ]
    if not matches:
        raise click.ClickException(
            f"{cwd} isn't inside an indexed repo. Pass --repo NAME, "
            "or run `ragx index` in the repo first (`ragx repos` lists them)."
        )
    return max(matches, key=lambda r: len(r["path"]))["name"]


def _loc(s: dict) -> str:
    return f"{s['filepath']}:{s['start_line']}-{s['end_line']}"


@click.group()
def cli():
    """Ask questions about your own code."""


@cli.command()
@click.argument("path", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--name", help="Repo name (default: the folder name).")
def index(path, name):
    """Index (or re-index) the repo at PATH, default the current folder."""
    body = {"path": str(Path(path).resolve()), "repo": name}
    job = _request("POST", "/index", json=body).json()
    if job["status"] == "already_running":
        click.echo("Already indexing this repo; following that job.")

    while True:
        s = _request("GET", "/index/status", params={"job_id": job["job_id"]}).json()
        if s["batches_total"]:
            click.echo(f"\rEmbedding: batch {s['batches_done']}/{s['batches_total']}", nl=False)
        if s["status"] != "running":
            break
        time.sleep(0.5)
    click.echo()

    if s["status"] == "failed":
        raise click.ClickException(s["error"])
    r = s["result"]
    click.echo(
        f"Indexed {r['repo']}: {r['files']} files, {r['chunks']} chunks "
        f"({r['embedded']} embedded, {r['unchanged']} unchanged, {r['deleted']} deleted)"
    )
    for skipped in r["skipped"]:
        click.echo(click.style(f"  skipped {skipped}", fg="yellow"))


@cli.command()
@click.argument("question")
@click.option("--repo", help="Repo name (default: inferred from the current folder).")
def ask(question, repo):
    """Ask a question; the answer streams as it's written."""
    repo = _resolve_repo(repo)
    resp = _request("POST", "/ask", json={"repo": repo, "question": question}, stream=True)

    sources = []
    click.echo(click.style(f"[{repo}] thinking...", dim=True), nl=False)
    started = False
    for line in resp.iter_lines():
        if not line:
            continue
        event = json.loads(line)
        if event["type"] == "sources":
            sources = event["sources"]
        elif event["type"] == "text":
            if not started:
                click.echo("\r\033[K", nl=False)  # clear the "thinking..." line
                started = True
            click.echo(event["text"], nl=False)
        elif event["type"] == "error":
            click.echo()
            raise click.ClickException(event["error"])
    click.echo("\n")

    if sources:
        click.echo(click.style("Sources:", bold=True))
        for s in sources:
            click.echo(click.style(f"  {_loc(s)}  ({s['layer']})", dim=True))


@cli.command()
@click.argument("query")
@click.option("--repo", help="Repo name (default: inferred from the current folder).")
@click.option("--layer", "layers", multiple=True, type=click.Choice(["function", "file", "repo"]),
              help="Search only this layer (repeatable). Default: the classifier decides.")
def q(query, repo, layers):
    """Search only, no LLM: show which chunks match and their scores."""
    repo = _resolve_repo(repo)
    body = {"repo": repo, "query": query, "layers": list(layers) or None}
    data = _request("POST", "/query", json=body).json()
    click.echo(click.style(f"[{repo}] layers searched: {', '.join(data['layers_searched'])}", dim=True))
    for r in data["results"]:
        click.echo(f"  {r['score']:.3f}  {r['layer']:8}  {_loc(r)}")


@cli.command()
def repos():
    """List indexed repos."""
    rows = _request("GET", "/repos").json()["repos"]
    if not rows:
        click.echo("No repos indexed yet. Run `ragx index` inside one.")
    for r in rows:
        click.echo(
            f"{r['name']:24} {r['file_count']:5} files  {r['chunk_count']:6} chunks  "
            f"last indexed {r['last_indexed_at'] or 'never'}\n{'':24} {r['path']}"
        )
