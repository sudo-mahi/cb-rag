# cb-rag

A personal codebase RAG that runs entirely on your machine. Index every repo
you work on, then ask questions about your code from the terminal, with
answers grounded in (and citing) your actual source files.

```
$ cd ~/code/my-service
$ ragx ask "how does the dataflow work?"
The pipeline runs as a Step Functions state machine on ECS Fargate:
Prepare splits the site's inverters into batches, ProcessBatches runs them
in parallel (ItemsPath $.prepare.batches), and Aggregate rolls the results
up with rollup.run_site_rollup ...

Sources:
  repo summary (ragx summary)  (repo)
  docs/FLOW.md:1-120  (file)
  template.yaml:1-100  (function)
  ...
```

- **Fully local.** Embeddings, answers and completions all run on local
  models through [Ollama](https://ollama.com). No API keys, no code leaves
  your machine, works offline.
- **Understands code structure.** Files are split at function and class
  boundaries with tree-sitter, not at arbitrary character counts.
- **Three layers of context.** Function chunks for "how does X work",
  file summaries for "where is X", and a repo summary for "what is this
  project", combined broad to narrow in every prompt.
- **Docs and infrastructure too.** READMEs, Markdown docs, YAML/JSON/TOML
  config, Dockerfiles and shell scripts are indexed alongside the code.
- **Cheap re-indexing.** Every chunk is content-hashed; re-indexing only
  embeds what actually changed.
- **Inline completion engine.** A fill-in-the-middle endpoint that uses
  similar code from elsewhere in the repo (an editor integration is not
  built yet).

## How it works

```
┌──────────────┐   HTTP    ┌───────────────────────────────┐   ┌──────────────┐
│  ragx (CLI)  │ ────────▶ │  daemon (FastAPI, :8787)      │──▶│ Ollama       │
└──────────────┘           │  index · search · ask ·       │   │ (local       │
                           │  complete                     │   │  models)     │
                           └──────────────┬────────────────┘   └──────────────┘
                                          │
                              ~/.cb-rag/  ├─ db/              Chroma vectors
                                          └─ registry.sqlite3 indexed repos
```

**Indexing** (`ragx index`) lists the repo's files (respecting its
`.gitignore`), chunks each one, embeds only new or changed chunks, stores
them in a per-repo Chroma collection, and writes a repo summary on the
first index or once more than 10% of files have changed.

**Asking** (`ragx ask`) picks which layers to search from the question,
retrieves the most relevant chunks, builds a prompt grouped by file with the
repo summary on top, and streams the local model's answer. Code the answer
quotes is checked against the retrieved sources, and a warning is shown if
any of it doesn't appear there.

**Completing** (`/complete`) fills in the line at the cursor using the code
around it plus similar functions from other files, cached per region so
suggestions stay fast while you type.

All design decisions and the reasoning behind them (43 so far, with
measurements) are in [`docs/DESIGN.md`](docs/DESIGN.md).

## Requirements

- macOS or Linux; developed on an Apple Silicon Mac with 16 GB RAM
- Python 3.14 and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) running locally, with these models:

| Model | Size | Used for |
|---|---|---|
| `qwen3-embedding:0.6b` | 0.6 GB | Embeddings (indexing and search) |
| `qwen2.5-coder:7b` | 4.7 GB | Answers (`ask`) and repo summaries |
| `qwen2.5-coder:1.5b` | 1 GB | Inline completion (`complete`) |

## Setup

```bash
# 1. Ollama and the models
brew install ollama
brew services start ollama
ollama pull qwen3-embedding:0.6b
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:1.5b

# 2. The project
git clone https://github.com/sudo-mahi/cb-rag.git
cd cb-rag
uv sync

# 3. The ragx command, available from any folder
#    (editable: code changes apply without reinstalling;
#     pin 3.14 because some dependencies have no 3.15 wheels yet)
uv tool install --editable . --python 3.14
```

## Usage

Start the daemon in its own terminal and leave it running:

```bash
cd cb-rag
uv run uvicorn server.daemon:app --host 127.0.0.1 --port 8787
```

Then, from anywhere:

```bash
cd ~/code/my-service
ragx index                       # index (or re-index) the repo you're in
ragx ask "what is this repo about?"
ragx ask "how are failed batches retried?"
ragx q "mongo connection setup"  # search only: matching chunks and scores
ragx summary                     # read the repo summary every answer starts from
ragx repos                       # list indexed repos
ragx complete src/app.py 42 --typed 6   # try inline completion on line 42
```

Inside a repo, commands use that repo automatically; use `--repo NAME` to
target another one. If the daemon isn't running, every command prints the
exact command to start it.

### HTTP API

The CLI is a thin client; the daemon on `localhost:8787` can be called
directly (interactive docs at `http://localhost:8787/docs`):

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `GET /repos` | Indexed repos |
| `POST /index`, `GET /index/status` | Start indexing in the background, poll its progress |
| `POST /query` | Search only, no LLM |
| `POST /ask` | Streamed answer (NDJSON: `sources`, `text`…, optional `warning`, `done`) |
| `GET /summary` | The stored repo summary |
| `POST /complete` | Single-line fill-in-the-middle suggestion |

## Configuration

All tunables live in [`server/config.py`](server/config.py): model names,
context window sizes, retrieval depth per layer, chunking limits, the
repo-summary refresh threshold, and completion cache settings. Changing the
embedding model is safe: each collection records the model that built it
and is rebuilt on the next `ragx index`.

Runtime data lives in `~/.cb-rag/` (vector DB, repo registry), separate
from the project.

## What gets indexed

- **Code, chunked by function and class:** Python, JavaScript/JSX,
  TypeScript/TSX, Go, Java (methods), C/C++
- **As text:** `.md` (split by heading), `.yaml`/`.yml`, `.json`, `.toml`,
  `.sh`, `Dockerfile*`
- **Skipped:** anything in the repo's `.gitignore`, plus `node_modules`,
  virtualenvs, build output, lockfiles and minified JS

## Known limitations

- Answers come from a 7B local model: good at explaining code it's shown,
  but it can still get details wrong. Check the cited lines for anything
  important.
- On a 16 GB machine, memory is the main constraint; with many apps open,
  answers can slow down noticeably.
- Rust, Ruby, Java/TypeScript classes and JavaScript arrow functions aren't
  chunked by function yet (their files are still covered by file-level
  chunks).
- A class and its methods are chunked separately, so both can appear in
  one prompt.
- Completion can mangle a word when the cursor stops mid-word.
- No editor integration yet; `/complete` is usable via `ragx complete` or
  the HTTP API.

## License

[MIT](LICENSE)
