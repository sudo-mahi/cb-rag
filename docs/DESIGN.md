# ragx — Personal Codebase RAG — Technical Design Document

**Status:** Living document. Updated as decisions are made or revisited.
**Last updated:** 2026-08-05
**Owner:** you

---

## 1. Purpose

A personal, always-on RAG system that indexes every repo you work on, answers
natural-language questions about your code from the terminal (`ragx ask`), and
provides inline code completions in VSCode — grounded in your actual codebase,
not generic training data.

---

## 2. Core Idea (Why RAG, not full-context)

Stuffing an entire repo into an LLM prompt doesn't scale — too many tokens,
too much irrelevant code, slower and more expensive per question.

RAG splits the problem in two:
- **Search** — pre-process the repo once into searchable pieces (chunks),
  find only the pieces relevant to a given question.
- **Generate** — hand just those relevant pieces to the LLM to reason over.

Everything in this design exists to make the **search** half accurate.

---

## 3. System Architecture

Three logical components, sharing one persistent brain:

```
┌─────────────────────────────────────────────────────┐
│                     YOUR MACHINE                      │
│                                                         │
│  ┌──────────────┐        ┌──────────────────────┐    │
│  │   INDEXER      │──────▶│   VECTOR DATABASE      │   │
│  │ (batch job,    │ writes│  (Chroma, on disk)     │   │
│  │  runs on       │       │  ~/.codebase-rag/db    │   │
│  │  demand/watch) │       └───────────┬────────────┘   │
│  └──────────────┘                    │ reads           │
│         │                              ▼                 │
│         │                ┌──────────────────────────┐   │
│         │                │   DAEMON (FastAPI)        │   │
│         └───────────────▶│   localhost:8787          │   │
│                          │   - always running         │   │
│                          │   - holds DB + API clients │   │
│                          └───────┬──────────┬─────────┘   │
│                                  │           │             │
│                       HTTP       │           │  HTTP       │
│                                  ▼           ▼             │
│                          ┌───────────┐  ┌──────────────┐  │
│                          │  ragx CLI  │  │ VSCode        │  │
│                          │ (terminal) │  │ extension     │  │
│                          └───────────┘  │ (inline ghost │  │
│                                          │  text)         │  │
│                                          └──────────────┘  │
└─────────────────────────────────────────────────────────┘
             │                                    │
             ▼                                    ▼
     Anthropic API (answers)          Voyage API (embeddings)
```

**Rationale:** the expensive parts (loading models, holding DB connections)
happen once in the daemon. CLI and editor plugin are thin, fast HTTP clients.
This is what makes inline suggestions feel instant instead of laggy.

---

## 4. Technology Choices

| Concern | Choice | Why |
|---|---|---|
| Vector DB | **Chroma** (embedded) | No server to run, disk-persisted, scales fine to 100K–1M+ vectors for a personal multi-repo setup. LanceDB is a viable swap-in alternative. Cloud vector DBs (Pinecone etc.) are overkill for single-machine, single-user use. |
| Embedding model | **Local `qwen3-embedding:0.6b` via Ollama** (was voyage-code-3; see decision #34) | Embeds by semantic/functional similarity, not surface text similarity, which directly determines retrieval quality. Local: free, private, no rate limits. |
| LLM (Q&A, `/ask`) | **Local `qwen2.5-coder:7b` via Ollama** (was Claude Sonnet; see decision #33) | Free and private; latency less critical since user is waiting on a considered answer. |
| LLM (inline `/complete`) | **Claude Haiku** | Fires on every typing pause — must return in <300–500ms, so speed > depth here. |
| Chunking | **tree-sitter** (AST-based) | Splits at function/class boundaries, never mid-function. Fixed-size/char-based chunking was rejected — it breaks code semantics. |
| Change detection | **git as pre-filter, content hash as source of truth** | Git alone misses uncommitted changes, untracked files, branch switches, and non-git folders. Hashing file/chunk contents is the real source of truth; git status is just a cheap way to narrow which files to hash first. |
| Registry / metadata store | **SQLite**, one shared DB across all repos, `repo_name` column | JSON doesn't scale (full read/write on every touch). SQLite gives fast indexed lookups and free cross-repo querying later, without restructuring. |
| CLI framework | **Click** | Subcommands, options, help text with minimal boilerplate. |
| API framework (daemon) | **FastAPI** | Async-native, auto docs at `/docs`, minimal boilerplate for a small local API. |

---

## 5. Chunking Strategy — Three Layers

Different question types need different granularity. No single chunk size is
correct for "all types of questions," so we store three layers, all embedded
into the same Chroma DB, tagged by `level`:

1. **Function/class-level** ("detail layer") — one chunk per function/class,
   extracted via tree-sitter AST parsing. Answers narrow "how does X work"
   questions.
2. **File-level summary** ("neighborhood layer") — one chunk per file,
   describing what the file is responsible for. Answers "what does this file
   do / where would I find X."
3. **Repo-level summary** ("city layer") — one chunk per repo, describing
   overall architecture. Answers "how is this project structured."

Retrieval doesn't need one perfect chunk — it needs the right *set* of
chunks; the LLM synthesizes across them.

---

## 6. File-Level Summaries — Generation Strategy

**Decision: Hybrid (cheap-by-default, LLM-upgrade-on-demand).**

Options considered:
- **A — LLM writes a summary for every file at index time.** Most accurate,
  but costs one LLM call per file, on every full re-index. Expensive at
  scale across many repos; also introduces its own staleness tracking need.
- **B — Cheap heuristic summary, no LLM at all.** Free, instant, built from
  signals already available at zero cost:
  - File path + name
  - Function/class names in the file (already extracted during chunking)
  - Existing docstrings / top-of-file comments
  - Import statements
  These get concatenated into one descriptive string and embedded.
  Weaker semantically — can miss a file's "real" purpose if it's not
  obvious from names/docstrings.
- **C — Hybrid.** Cheap heuristic (Option B) generated for every file by
  default. A real LLM-written summary is generated **only** for files that
  actually contributed to a full `/ask` answer (not just any `/query` search
  hit) — a tighter bar than "any search match." Once generated, it's cached
  and reused until the file's content changes.

**Chosen: Option C.** Defers LLM cost to files you actually work with,
rather than paying upfront for an entire codebase including code you never
touch.

---

## 7. Change Detection

**Decision: Git as a fast pre-filter; content hash as the source of truth.**

Git alone (`git diff`, `git status`) was considered insufficient because it
misses:
- Uncommitted in-progress edits
- Untracked new files (partially recoverable via `git status --porcelain`)
- Branch switches (many files appear "changed" with no real edit)
- Folders that aren't git repos at all

**Actual mechanism:**
- On each index run, use `git status --porcelain` (when available) as a
  cheap candidate filter — only bother hashing files git flags as possibly
  touched.
- For each candidate (or all files, on first index / non-git folders),
  compute a content hash and compare against the stored hash.
- Hash mismatch → file is genuinely dirty → re-chunk / re-embed.

---

## 8. Registry (SQLite) — `files` Table

**Update (decisions #26, #27): only a `repos` table is built for now.**

```sql
CREATE TABLE repos (
    name            TEXT PRIMARY KEY,
    path            TEXT NOT NULL,   -- absolute repo root
    registered_at   TEXT NOT NULL,
    last_indexed_at TEXT,            -- NULL until first index completes
    file_count      INTEGER NOT NULL DEFAULT 0,
    chunk_count     INTEGER NOT NULL DEFAULT 0
);
```

The `files` table below is deferred until a feature needs it (LLM-upgraded
file summaries, Section 6 Option C). Today it would mostly duplicate
Chroma: per-chunk hashes already skip re-embedding (the costly part), and
the file chunk's text already holds the summary. A repo name registered
for a different path is refused (#27). The original design follows.

One shared SQLite database across all indexed repos.

```sql
CREATE TABLE files (
    repo_name       TEXT NOT NULL,
    filepath        TEXT NOT NULL,   -- relative to repo root, not absolute
    content_hash    TEXT NOT NULL,   -- whole-file hash (staleness pre-check)
    summary_level   TEXT NOT NULL DEFAULT 'cheap',  -- 'cheap' or 'llm'
    summary_text    TEXT,             -- cached summary (whichever level)
    last_indexed_at TEXT NOT NULL,   -- ISO timestamp
    PRIMARY KEY (repo_name, filepath)
);
```

**Design decisions & rationale:**
- **Composite key `(repo_name, filepath)`**, not a synthetic `id` — this
  *is* the natural identity of a row; avoids a meaningless auto-increment on
  top of an already-unique lookup key.
- **`filepath` stored relative to repo root**, not absolute — keeps rows
  stable if a repo folder is moved on disk. Absolute path reconstructed via
  the repo registry's known root path when needed.
- **No `chunk_ids` column.** Considered storing a JSON array of a file's
  chunk IDs for fast lookup at delete-time, but rejected in favor of
  querying Chroma live (`collection.get(where={"filepath": ...})`) at the
  point of use. Rationale: chunk existence is already tracked in Chroma;
  storing it a second time in SQLite creates a sync obligation with no
  correctness benefit, and a real failure mode if the two ever drift
  (crash mid-write, manual edits, future code that touches Chroma
  directly). The live-query cost is milliseconds on a local embedded DB —
  negligible. **Principle applied:** when data is fully derivable from a
  system already trusted as source of truth, don't duplicate it elsewhere
  just for convenience.
- **`summary_level` as an explicit column**, not inferred from whether
  `summary_text` is null — the cheap summary is always present, so
  inferring "was this upgraded" from nullability wouldn't work; an
  explicit flag makes it a direct, readable query.
- **`last_indexed_at`** — not required for staleness logic (the hash
  handles that), but cheap to add and useful for health-check tooling
  (e.g. "which files haven't been touched in 60 days") and debugging.

**Sync discipline (correctness, not scale):** any operation touching both
Chroma and this registry must write to Chroma first, and only update the
registry after that write succeeds. Worst case on a crash mid-operation is
a file gets redundantly re-indexed next run (safe, wasted work) — never a
registry row falsely claiming to be current when Chroma doesn't match it.

---

## 9. Chunk Identity & Per-Chunk Hashing

**Decision: Hash and identify chunks individually, not the whole file at
once — stored as Chroma metadata, not in SQLite.**

**Problem with whole-file hashing:** editing one function in a 15-function
file would mark the entire file dirty, forcing re-embedding of 14 unchanged
functions — wasted API calls and time, especially on files touched
repeatedly during a work session.

**Fix:** hash per chunk (per function/class). Only chunks whose hash
actually changed get re-embedded; unchanged ones are left alone in Chroma.

**Where the hash lives:** inside Chroma's existing per-chunk metadata dict
(alongside `filepath`, `start_line`, `end_line`) — not a second SQLite
table. This preserves the same principle from Section 8: chunk-level facts
live with the chunks (Chroma); file-level facts live in SQLite (registry).
No overlapping source of truth between the two stores.

**Chunk identity — not based on line numbers.** Initially considered
`chunk_id = filepath:start_line:end_line`, but this breaks under reordering
— e.g. adding a function above an existing one shifts every subsequent
function's line numbers, making unchanged code look "new" and triggering
unnecessary re-embeds.

**Fix:** identity based on *what* the chunk is, not *where* it sits:
```
chunk_id = filepath:qualified_name
  e.g.  src/auth/jwt.py:validate_token
  e.g.  src/auth/jwt.py:TokenManager.refresh
```
`start_line` / `end_line` are still stored as metadata (useful for jumping
to the chunk in an editor) but are no longer used as identity. Rare true
name collisions (e.g. dynamically generated duplicate names) fall back to
an appended occurrence index (`:1`, `:2`) as a last resort.

**Diffing algorithm on file change:**
```
1. Parse current file → chunks, each with id = filepath:qualified_name
2. Fetch existing chunk ids + hashes for this file from Chroma
3. new_ids - existing_ids  → added functions → embed
4. existing_ids - new_ids  → removed functions → delete from Chroma
5. ids in both, hash differs → modified → re-embed
6. ids in both, hash same → unchanged → skip entirely
```

---

## 10. Query-Time Retrieval Strategy

**Decision: Classify the question, then route to relevant layer(s)
(Option 3 of four considered).**

Options considered:

| | Handles broad Qs | Handles narrow Qs | Extra cost | Complexity |
|---|---|---|---|---|
| 1. Merge all layers, single ranked search | Poor — dense function chunks drown out summaries | Good | None | Lowest |
| 2. Fixed split across layers (e.g. 3/2/1) | OK, forced inclusion | OK, forced dilution | None | Low |
| **3. Classify question → route to layer(s)** | **Good — targeted** | **Good — targeted** | **Small (classification step)** | **Medium** |
| 4. Hierarchical sequential search (repo→file→function) | Best — mirrors human exploration | Good | Higher (sequential round trips) | Highest |

**Why not Option 1:** embedding similarity tends to favor dense, specific
text (function code) over broad/abstract text (summaries), so summary
layers rarely win the ranking even for broad questions — the three-layer
system would effectively go unused.

**Why not Option 4 (yet):** adds sequential-search latency to *every*
query, including the common case of narrow, specific questions, to better
serve the less common case of broad architectural questions. More to build,
more places for an early bad result to cascade. Revisit only if Option 3
proves insufficient in practice.

**Chosen: Option 3.**

### 10.1 Classifier design

Start with a **cheap heuristic classifier** (no LLM call), upgrade to a
small LLM classification call later only if misrouting becomes a real,
observed problem.

Output is a **set** of layers to search, not a single label:

- Question contains an identifier-like token (camelCase/snake_case, or a
  backticked term) → include `function`
- Question mentions a file extension, "file," "module," or a path-like
  string → include `file`
- Question uses broad/structural vocabulary ("architecture," "structure,"
  "overview," "how does this fit together," "high level," "design," "flow
  between") → include `repo`
- No signals matched → default to `{function, file}` (skip `repo` by
  default to keep default-case context tight)

### 10.2 Search execution

```
layers = classify(question)
results = []
for layer in layers:
    results += chroma_search(query, filter={"level": layer}, top_k=per_layer_k[layer])
```

`top_k` differs per layer, since layers don't have equally many meaningful
candidates:
```
per_layer_k = {"function": 6, "file": 3, "repo": 1}
```

### 10.3 Result filtering — **deferred decision**

For now: **include everything each routed layer's search returns**,
unfiltered by individual similarity score, since routing already decided
that layer is relevant. Revisit with a relevance floor per layer (dropping
weak matches even within a correctly-routed layer) once real usage and
token-cost data suggests it's needed. **Explicitly not solved yet — do not
treat as final.**

---

## 11. Prompt Construction from Retrieved Context

**Decision: group retrieved chunks by file, ordered broad → narrow, with
summaries explicitly labeled as summaries (never presented as verbatim
code).**

### 11.1 Why ordering matters

LLMs read top-to-bottom, and earlier content frames how later content gets
interpreted — giving repo/file context before a specific function primes
correct interpretation of that function's role (e.g. why a timeout is
short, because it's part of a login flow). So layers are ordered
**repo-summary → file-summary → function chunks**, not raw search-score
order.

This only matters when multiple layers were actually routed to (Section
10.1) — for a narrow question routed to `{function}` only, there's no
broader context to order against, so the question is moot in the common
case.

### 11.2 Format — grouped by file, not by layer or raw score order

```
## Repository Context
[repo-summary chunk, if present]

## File: src/auth/jwt.py
[file-summary chunk, if present]

### validate_token (lines 12-28)
```python
def validate_token(...):
    ...
```

### TokenManager.refresh (lines 45-60)
```python
def refresh(...):
    ...
```
```

**Why grouped by file:** if multiple function chunks came from the same
file, the file's summary is shown once, above all of them — giving shared
context without repeating it per-function (wasteful) or omitting it
(loses the connection between a function and its file's purpose). The
heading structure (`## File:`, `### function_name (lines X-Y)`) also
doubles as the citation format — the LLM can reference `validate_token in
src/auth/jwt.py, lines 12-28` directly from what it's already looking at,
rather than cross-referencing a separate metadata list.

### 11.3 Summaries must be explicitly labeled, never presented as code

Repo-level and file-level summary chunks are AI-generated (or heuristic)
paraphrases, not verbatim source. They must be visibly marked as such in
the prompt (e.g. a `_Summary (auto-generated):_` prefix) so the LLM never
treats a summary's paraphrase as if it were exact code — and so that if
raw context is ever surfaced to the user for debugging, the distinction is
immediately visible there too.

### 11.4 Known open edge case — not solved yet

A file-summary chunk (built from the cheap heuristic in Section 6, which
includes function names) can "name-drop" a function that a function-level
search *also* returned in full. This isn't true redundancy — one is a
name-drop, the other is the real function body — but it's worth watching
once real output is seen, rather than solving preemptively. Tracked in
Section 12 open items.

---

## 12. Inline Completion (`/complete`) — Retrieval & Trigger Design

**Status: designed, but explicitly deferred to Phase 2 (see Section 13).
Recorded now so the design isn't lost, not because it's being built next.**

### 12.1 Retrieval strategy — fixed combination, no classification

Unlike `/ask`, `/complete` doesn't need the classify-then-route step
(Section 10) — the cursor position already tells us exactly where we are,
so there's no ambiguity about scope to classify away. Every `/complete`
request always combines exactly two things:

1. **A live function-level search**, queried using the last ~10-15 lines
   of `prefix` (or the current enclosing function, if a lightweight parse
   can detect it) — finds similar logic elsewhere in the repo (reuse /
   correctness).
2. **The current file's cheap summary chunk, always attached** — a direct
   lookup by filepath, not a similarity search, since we already know
   which file the cursor is in. Gives file style/convention context
   without spending a search on it.

No classification step in this path, deliberately — it would eat into the
<300–500ms latency budget (Section 4) for a call that fires on every
typing pause, and there's nothing to classify since the scope is already
known.

### 12.2 Trigger logic — debounce + trigger characters, combined

Neither mechanism alone is sufficient:
- **Debounce alone** (fire after N ms of no keystrokes) can fire mid-word,
  since natural typing isn't perfectly even-paced.
- **Trigger characters alone** (fire after `.`, `(`, newline) miss the
  common case of pausing mid-line to think, with no trigger character
  nearby.

**Combined mechanism** — trigger characters short-circuit the debounce
timer rather than running in parallel with it:

```
on each keystroke:
    reset the debounce timer
    if character just typed is a trigger char (., (, newline, etc.):
        fire immediately (bypass debounce)
    else:
        schedule fire after debounce_ms of no further keystrokes
```

Default `debounce_ms = 300` (in line with typical inline-completion tools).

### 12.3 Request cancellation — required, not optional

If a new trigger fires while a previous `/complete` request is still
in-flight, the previous request's `prefix` no longer matches what's on
screen by the time it returns — showing it as a suggestion risks an
actively wrong completion. **Every new trigger must cancel any in-flight
`/complete` request before firing a new one** (client-side via
`AbortController` in the VSCode extension; the daemon simply finishes the
wasted computation and the client discards the result).

---

## 13. Build Phasing

**Decision: two phases. Phase 2 does not start until Phase 1 is proven
working.**

- **Phase 1 — `/ask` end-to-end.** Indexing pipeline (chunking, registry,
  change detection), query-time retrieval (classify → route → search),
  prompt construction (Section 11), the daemon, and the CLI. No VSCode
  extension, no inline completions. Success criterion: `ragx ask` gives
  accurate, well-cited answers against a real repo, reliably, before
  anything in Phase 2 is started.
- **Phase 2 — `/complete` + inline UI.** The retrieval/trigger design in
  Section 12, plus the VSCode `InlineCompletionItemProvider` integration,
  ghost-text rendering, and accept/reject UX. Explicitly gated on Phase 1
  working well — if `/ask`'s retrieval quality or indexing reliability
  turns out to need rework, that rework happens before Phase 2 begins, not
  in parallel with it.

**Rationale:** `/complete` is a harder problem in some ways (tighter
latency budget, request cancellation, editor integration) and depends on
the same indexing/retrieval foundation as `/ask`. Validating that
foundation on the simpler, non-latency-critical `/ask` path first means
any fixes to chunking, retrieval, or the registry happen once, before
`/complete` is built on top of them — rather than discovering a foundational
issue after both are already built and needing to redo both.

---

## 14. Daemon API Contract

**Status: resolves open item #3. Part of Phase 1 scope.**

All endpoints are local HTTP on the daemon (`localhost:8787`, per Section
3). Every endpoint below is designed to be a thin wrapper over logic
already specified in earlier sections — nothing new is being decided here
beyond the wire format.

### 14.1 `POST /ask` — main Q&A endpoint

```
Request:
{
  "repo": "backend-api",
  "question": "how does JWT refresh work?",
  "top_k": {"function": 8, "file": 4, "repo": 1}   // optional, overrides Section 10.2 defaults
}

Response:
{
  "answer": "...",
  "sources": [
    {"filepath": "src/auth/jwt.py", "start_line": 45, "end_line": 60, "layer": "function"}
  ],
  "layers_searched": ["function", "file"]
}
```

- `sources` is structured, not left embedded in prose — the CLI needs to
  print citations cleanly, and structured sources let citations be checked
  against what was actually retrieved (Section 11.2's citation format
  reads directly off this).
- `layers_searched` exposes which layers the Section 10.1 classifier
  picked — debugging visibility into misrouting without separate logging.
- `top_k` is optional and overrides the `per_layer_k` defaults from
  Section 10.2 per-request; omitted, the daemon uses its configured
  defaults.

**Update (decisions #30-#32) — as built, the response streams as NDJSON:**

```
{"type": "sources", "layers_searched": ["file", "function"],
 "sources": [{"filepath": "...", "start_line": 45, "end_line": 60, "layer": "function", "kind": "code", "score": 0.61}]}
{"type": "text", "text": "The refresh flow..."}          // repeated
{"type": "done", "stop_reason": "end_turn"}              // or {"type": "error", "error": "..."}
```

### 14.2 `POST /query` — raw search, no LLM synthesis

```
Request:
{
  "repo": "backend-api",
  "query": "jwt validation",
  "layers": ["function", "file"],   // optional — omit to use the Section 10.1 classifier
  "top_k": {"function": 8, "file": 4, "repo": 1}   // optional, same override as /ask
}

Response:
{
  "results": [
    {"text": "...", "filepath": "...", "start_line": 45, "end_line": 60, "layer": "function", "score": 0.82}
  ]
}
```

- Exists separately from `/ask` as the inspection/debugging path — shows
  retrieval output before the LLM touches it, needed to gather the
  real-usage data Section 10.3 says is required before setting a
  relevance-floor filter. `ragx q` maps directly to this.
- `layers` lets a caller bypass the classifier entirely and force a
  specific layer — useful when checking whether the classifier misrouted
  by manually querying the layer it *didn't* pick.

### 14.3 `POST /index` — fire-and-forget indexing

```
Request:
{
  "path": "/Users/you/code/backend-api",
  "repo": "backend-api"   // optional, defaults to basename(path)
}

Response:
{
  "job_id": "idx_a1b2c3",
  "status": "started"
}
```

### 14.4 `GET /index/status?job_id=idx_a1b2c3`

```
Response:
{
  "job_id": "idx_a1b2c3",
  "status": "running" | "done" | "failed",
  "files_processed": 142,
  "files_total": 310,
  "chunks_indexed": 890,
  "error": null
}
```

**Update (decisions #28, #29) — as built:**

```
{
  "job_id": "idx_a1b2c3",
  "repo": "backend-api",
  "status": "running" | "done" | "failed",
  "batches_done": 3,
  "batches_total": 12,       // null until chunking + diffing finish
  "result": {...},           // index_repo stats once done
  "error": null              // "ExceptionType: message" once failed
}
```

`files_processed` / `files_total` were replaced: chunking finishes in
seconds, embedding is the slow part, so file counts would sit at 100%
while the real work runs. A second `POST /index` for a repo with a running
job returns that job (`"status": "already_running"`). A name clash
(#27) returns 409 immediately.

- Fire-and-forget plus poll rather than a blocking `/index` call — a large
  repo can take minutes to index (Section 7/9), and a request blocking
  that long is fragile (timeouts, no progress feedback).

### 14.5 `GET /repos` — list registered repos

```
Response:
{
  "repos": [
    {"name": "backend-api", "path": "/Users/you/code/backend-api", "last_indexed_at": "2026-08-05T10:00:00Z", "file_count": 310}
  ]
}
```

- The read path over the registry from Section 8 — without it, nothing
  outside SQLite itself can answer "what repos does this tool know about."

### 14.6 `GET /health`

```
Response: {"status": "ok"}
```

- Trivial, but explicit — gives `make status` something to curl.

---

## 15. Repo-Level Summary Generation

**Status: resolves the repo-level summary open item from Phase 1.**

**Decision: unconditional LLM generation at first index (not the
cheap-first pattern used for file-level summaries), refreshed only when
file-level summaries have changed significantly.**

### 15.1 Why this differs from the file-level approach (Section 6)

Section 6's cheap-by-default/LLM-on-demand pattern exists because there
are potentially hundreds of files per repo, most never touched by any
`/ask` query — laziness avoids paying for summaries that are never used.

That economic argument doesn't apply here: **there is exactly one
repo-level summary per repo.** Paying for one LLM call per repo, once, is
trivial regardless of repo size — there's nothing to be lazy about when
there's only one thing to generate. So the right default flips: generate
it with an LLM unconditionally, at first index.

### 15.2 What it's built from

- The directory tree (top 2–3 levels)
- `README.md`, if present — often already *is* a project summary, written
  by a human
- The description field from `package.json` / `pyproject.toml` /
  `Cargo.toml`, if present
- The set of file-level summaries already generated in the same indexing
  pass (Section 6) — concatenated, giving the LLM a bottom-up view to
  synthesize alongside the top-down README/tree signals

### 15.3 Refresh trigger

Unlike file-level summaries (refreshed per-file on content-hash mismatch,
Section 9), regenerating the repo-level summary on every single file edit
would be wasteful — one function changing rarely changes what the project
*as a whole* is about. Trigger: regenerate when the set of file-level
summaries has changed by more than a threshold (e.g. >10% of files
touched since the repo-summary was last generated) — checked cheaply by
comparing `last_indexed_at` timestamps already in the registry (Section 8)
against when the repo-summary chunk was last written; no new
infrastructure needed.

---

## 16. First-Time Repo Registration — Bootstrap Flow

**Status: resolves the repo-registration open item from Phase 1.**

### 16.1 Sequence

```
1. User runs: ragx index (or POST /index with a path)
2. Daemon checks registry (Section 8) — is this repo_name already known?
   - No  → this is first-time registration
   - Yes → this is a re-index, use existing registry rows for change detection (Section 7)
3. First-time only: write a new row to the repos table (name, path, registered_at)
4. Walk the repo, applying default ignore patterns (Section 16.2)
5. For every non-ignored file: chunk (Section 2), embed, store in Chroma
6. Generate file-level cheap summaries for every file (Section 6, Option B — free, no LLM)
7. Generate the ONE repo-level summary (Section 15 — the only LLM call in
   first-time indexing beyond per-file embeds)
8. Mark job as "done" in /index/status (Section 14.4)
```

### 16.2 Default ignore patterns

Hardcoded, standard set — no novel design needed:

```
.git, node_modules, venv, .venv, __pycache__, dist, build, .next,
target (Rust/Java), vendor (Go), .idea, .vscode, *.lock, *.min.js
```

**Update (decision #23):** in git repos the file list comes from
`git ls-files --cached --others --exclude-standard`, so the repo's own
`.gitignore` is respected; the defaults above still apply on top.
Non-git folders use the defaults only. `.ragxignore` below is deferred
until a repo actually needs it.

**Override mechanism:** an optional `.ragxignore` file at repo root,
`.gitignore` syntax — covers the minority of repos with unusual
build-output folder names without inventing new syntax to learn, and is
entirely skippable if never needed.

---

## 17. Open / Not Yet Designed

Tracked here so nothing gets lost. Update this section as items get
resolved — move them into a numbered section above once decided.

**Resolved items:**

- [x] ~~#1 — Prompt construction from retrieved context~~ — **resolved,
      see Section 11.**
- [x] ~~#2 — Inline completion (`/complete`) retrieval strategy~~ —
      **designed, see Section 12.1–12.2.** (Design resolved; build itself
      deferred to Phase 2 per Section 13.)
- [x] ~~#3 — Daemon API contract~~ — **resolved, see Section 14.**
- [x] ~~Repo-level summary generation~~ — **resolved, see Section 15.**
- [x] ~~Registering a repo for the first time~~ — **resolved, see Section 16.**

**✅ Phase 1 design is now complete.** Every item required before Phase 1
build could start (per Section 13) is resolved. The one remaining Phase 1
item below is a "watch and see" item, not a blocker — it doesn't require a
design decision until real output surfaces the problem.

**Phase 1 items remaining (non-blocking):**

- [ ] File-summary chunks name-dropping a function that a function-level
      chunk also returns in full — not true redundancy, but worth watching
      once real output is seen (see Section 11.4).

**Phase 2 items (explicitly deferred — see Section 13):**

- [ ] **#4 — VSCode extension mechanics** — request cancellation mechanism
      is designed (Section 12.3); still open: accept/reject keybinding UX,
      ghost-text rendering details.
- [ ] Making the daemon persistent across reboots (`launchd`/`systemd`) —
      deferred until the manual-start version is proven stable (applies to
      both phases, but only becomes urgent once Phase 2 is running
      continuously alongside editor use).

---

## 18. Decision Log

Chronological record of decisions made and why, for future-you to audit.

| # | Decision | Rejected alternatives | Reason |
|---|---|---|---|
| 1 | RAG = search (find relevant code) + generate (LLM reasons over it) | Stuffing full repo into prompt | Doesn't scale with real repo sizes |
| 2 | AST-based chunking via tree-sitter | Whole-file chunks; fixed-char chunks | Whole-file too coarse/noisy; fixed-char breaks mid-function |
| 3 | Three-layer chunking (function/file/repo) | Single granularity | Different question types need different granularity |
| 4 | File summaries: cheap heuristic default + LLM upgrade on real `/ask` use | Always-LLM; always-cheap | Balances cost vs accuracy; defers spend to code actually used |
| 5 | Change detection: git pre-filter + content hash as truth | Git history alone | Git misses uncommitted changes, untracked files, branch switches, non-git folders |
| 6 | Registry: SQLite, shared DB, `repo_name` column | Per-repo SQLite files; JSON | SQLite scales; single DB enables free cross-repo queries later |
| 7 | Drop `chunk_ids` column, derive from Chroma live | Storing chunk ID list in SQLite | Avoids duplicate source of truth and drift risk; query cost is negligible |
| 8 | Per-chunk hashing, stored in Chroma metadata | Whole-file hashing | Avoids re-embedding unchanged functions when only one function in a file changes |
| 9 | Chunk identity = `filepath:qualified_name` | `filepath:start_line:end_line` | Line-based IDs break under reordering/insertion above a function |
| 10 | Retrieval: classify question → route to layer(s) (Option 3) | Merge-all search; fixed-split search; full hierarchical search | Best precision-to-complexity tradeoff; hierarchical deferred as premature |
| 11 | Include all results from routed layers, unfiltered by score, for now | Relevance-floor filtering per layer | No real usage data yet to set a sane threshold; revisit later |
| 12 | Prompt context: grouped by file, ordered repo→file→function, summaries explicitly labeled | Raw score-order concatenation; ungrouped/unlabeled chunks | Framing-first ordering aids LLM interpretation; grouping avoids repeating/losing file context; explicit labeling prevents summary paraphrase being mistaken for verbatim code |
| 13 | `/complete` retrieval: fixed function-search + always-attached file summary, no classification step | Classify-then-route (same as `/ask`) | Cursor position already gives known scope — nothing to classify; avoids eating into the <300–500ms latency budget |
| 14 | `/complete` trigger: debounce + trigger-characters combined, with mandatory request cancellation | Debounce alone; trigger characters alone | Debounce alone can fire mid-word; triggers alone miss mid-line pauses; cancellation prevents stale in-flight requests from showing outdated suggestions |
| 15 | Build phasing: Phase 1 = `/ask` end-to-end only; Phase 2 (`/complete` + VSCode UI) gated on Phase 1 working | Building both in parallel | Validates the shared indexing/retrieval foundation once, on the simpler non-latency-critical path, before building the harder inline-completion layer on top of it |
| 16 | Daemon API: 6 endpoints (`/ask`, `/query`, `/index`, `/index/status`, `/repos`, `/health`); `top_k` optional override on `/ask` and `/query` | Fixed internal-only `top_k`; blocking `/index` | Optional `top_k` enables debugging/tuning retrieval without redeploying; fire-and-forget + poll avoids fragile long-blocking HTTP calls for large-repo indexing |
| 17 | Repo-level summary: unconditional LLM generation at first index, refreshed at >10% file-change threshold | Cheap-first/LLM-on-demand (same pattern as file-level, Section 6) | Only one repo-summary exists per repo, so the "avoid paying for unused summaries" economics that justified laziness at file-level don't apply — one LLM call per repo is trivial regardless of repo size |
| 18 | First-time repo bootstrap: hardcoded default ignore patterns + optional `.ragxignore` override (`.gitignore` syntax) | Fully hardcoded with no override; fully user-configured from scratch | Hardcoded defaults cover the vast majority of repos with zero setup; `.ragxignore` handles the minority with unusual build-output folders without inventing new syntax |
| 19 | Embedding batching: greedy packing by estimated tokens (`len/3`), 100K budget per request (real limit 120K), each text capped at 96K chars | Fixed count per batch; exact counting via Voyage tokenizer | Fixed count can exceed the 120K token limit on large functions; exact counting adds a tokenizer dependency/download. Estimation over-counts for code and the budget leaves 20% headroom, so the cost is only a few extra requests |
| 20 | Embedding dimension: 1024 (`EMBEDDING_DIM`) | 512; 2048 | Voyage's default/benchmark point; size is fine at personal scale (~4 KB/chunk). Locked in once the Chroma collection exists |
| 21 | Imports: package-style from the project root (`from server.config import ...`) | Bare-module imports run from `server/` | Works identically for uvicorn, the CLI, and tests, all of which run from the root |
| 22 | Chroma: one collection per repo (named after the repo) | One shared collection with a `repo` metadata filter | `filepath:qualified_name` ids would collide across repos in a shared collection; per-repo scopes every search by default and makes dropping a repo trivial. Cross-repo search (not in Phase 1) = query several collections and merge |
| 23 | File selection: `git ls-files --cached --others --exclude-standard` in git repos, plus default ignore lists; `os.walk` + defaults for non-git folders; `.ragxignore` deferred until needed | Defaults only; `.ragxignore` via `pathspec` now | Respects the repo's own `.gitignore` (generated code, local config that may hold secrets, which would otherwise be sent to Voyage) with no new dependency |
| 24 | File-summary content: path, leading docstring/comments, top-level imports, signature line (first line) of each function/class. Chunk id `filepath:<file>`, `level: "file"` | Names only | Signatures carry params/types for free (first line of each existing chunk). `<` can't appear in a real qualified name, so no id collision |
| 25 | Files with no function chunks: file-level chunk is the real content (capped at 6K chars), flagged `kind: "code"`; summaries are `kind: "summary"`; empty files get no chunk | Summary-only for every file | Otherwise constants/config/script files are unsearchable. The `kind` flag keeps Section 11.3 honest: the prompt builder labels by `kind`, never presenting a summary as code |
| 26 | Registry: `repos` table only (name, path, timestamps, counts); `files` table deferred | Full Section 8 now | Chunk hashes in Chroma already skip the costly re-embeds; `summary_text` would duplicate the file chunk in Chroma (against #7). Add the `files` table when LLM summaries need it |
| 27 | Repo name already registered for a different path → refuse with an error naming the existing path | Auto-rename (`api-2`); identify repos by path | Sharing a collection between two folders makes the diff delete the other repo's chunks. Rare, so an explicit error beats a silent rename |
| 28 | Concurrent `/index` for the same repo → return the running job | Queue the second job; allow both | Two jobs diffing one collection can double-embed or delete each other's writes; edits made mid-run are picked up next index |
| 29 | `/index/status` reports embedding batch progress (`batches_done`/`batches_total`) via an `on_batch` callback through indexer → embeddings | Status + final stats only; file counts (original 14.4) | Embedding is the slow phase; file counts would show 100% almost immediately |
| 30 | `/ask` model: Claude Sonnet 5.5 (`claude-sonnet-5-5`), adaptive thinking at effort `medium`, server-side refusal fallback (`fallbacks: "default"`); `/complete` model updated to Claude Haiku 5.5 | Claude Opus 5.5 | ~2-3¢ per question vs ~4-6¢; strong enough for code Q&A. Fallback keeps false-positive safety declines (auth/crypto code) from returning nothing. One-line swap in `config.py` |
| 31 | `/ask` streams NDJSON: `sources` event first, then `text` events, then `done` (with `stop_reason`) or `error` | Whole answer as one JSON (original 14.1); SSE | Answers take 5-20s; streaming shows progress. NDJSON is the simplest to produce and parse from a Python CLI; SSE mainly helps browsers. Errors after streaming starts are in-band, since the 200 is already sent |
| 32 | `/ask` sources = every retrieved chunk; Claude cites inline as `path:start-end` | Structured "sources actually used" from Claude | Simple; inline citations already show what was used. Revisit with Section 6 Option C, which needs "files that contributed" |
| 33 | **Supersedes the model part of #30.** `/ask` runs on a local model via Ollama: `qwen2.5-coder:7b` (largest coding model that fits 16 GB RAM), `/api/chat` over plain `requests`, `num_ctx` 16384 set explicitly | Claude Sonnet 5.5 (paid; key returned 401); free cloud tiers (Gemini, Groq, OpenRouter) | Free, private (code never leaves the machine), no rate limits, works offline. Costs answer quality vs Sonnet and ~20s per answer (~27 tok/s on an M5). `num_ctx` must be explicit: Ollama silently drops the start of an over-long prompt. No provider abstraction yet (rule of three) |
| 34 | **Supersedes #19 (batching) and the model in Section 4.** Embeddings run locally: `qwen3-embedding:0.6b` via Ollama `/api/embed` (1024 dims, normalized). Queries get Qwen3's `Instruct: ...\nQuery: ...` prefix, documents none. `num_ctx` 8192 with `truncate: true`; fixed batches of 32 (progress only). Voyage code and the `voyageai` package removed | `qwen3-embedding:4b`; config switch keeping Voyage | Removes the last paid/rate-limited dependency: the pipeline is fully local and offline. Measured on cb-rag: full re-embed of 51 chunks in 2.4s; file-level routing improved, exact-function ranking slightly weaker than voyage-code-3 (best match 3rd instead of 1st, still within top_k). 4b is a one-line upgrade if quality disappoints |
| 35 | Each Chroma collection records `embedding_model` in its metadata. Indexing with a different model drops and rebuilds the collection; searching it raises `EmbeddingModelMismatch` ("re-index") | No check | Vectors from different models aren't comparable, and mixing them fails silently with meaningless scores |

---

## 19. Changelog

- **2026-08-05** — Initial document created, capturing all decisions through
  Section 10 (query-time retrieval strategy). Section 1 (prompt
  construction) marked in-progress as next step.
- **2026-08-05** — Added Section 11 (Prompt Construction from Retrieved
  Context). Resolved open item #1. Added open item on file-summary
  name-drop vs full function-chunk overlap (Section 11.4).
- **2026-08-06** — Added Section 12 (Inline Completion retrieval & trigger
  design — resolves open item #2's design, though build is deferred) and
  Section 13 (Build Phasing: Phase 1 = `/ask`, Phase 2 = `/complete` + UI,
  gated on Phase 1 success). Open items reorganized by phase.
- **2026-08-07** — Added Section 14 (Daemon API Contract: `/ask`, `/query`,
  `/index`, `/index/status`, `/repos`, `/health`, with optional `top_k`
  override). Resolved open item #3.
- **2026-08-07** — Added Section 15 (Repo-Level Summary Generation) and
  Section 16 (First-Time Repo Registration — Bootstrap Flow). **Phase 1
  design is now complete** — every blocking open item is resolved; only a
  non-blocking "watch and see" item remains (Section 11.4). Ready to move
  into implementation.
- **2026-11-13** — Implementation decisions: package-style imports (#21),
  estimated-token embedding batching (#19), 1024-dim embeddings (#20).
  `server/embeddings.py` written and verified against the live API.
- **2026-11-13** — Indexer decisions: one Chroma collection per repo (#22),
  git-driven file selection (#23, updates Section 16.2). `server/indexer.py`
  (function layer only) written and verified.
- **2026-11-13** — File layer: heuristic summaries (#24) and full-text
  fallback for function-less files (#25). Built in the chunker; every chunk
  now carries `level` and `kind` metadata.
- **2026-11-14** — Registry: `repos` table only (#26), name-clash refusal
  (#27); Section 8 updated. `server/registry.py` written, wired into the
  indexer (Chroma writes first, registry after), and verified.
- **2026-11-14** — Daemon slice 1: `/health`, `/repos`, `/index`,
  `/index/status`, `/query`; Section 10.1 classifier in
  `server/retrieval.py`. Duplicate-job guard (#28), batch progress (#29),
  Section 14.4 updated. Verified with curl.
- **2026-11-14** — `/ask`: Section 11 prompt construction in
  `server/ask.py`, Claude Sonnet 5.5 with refusal fallback (#30), NDJSON
  streaming (#31), all retrieved chunks as sources (#32). Section 14.1
  updated. `.env` loading moved into `config.py`.
- **2026-11-14** — `/ask` switched to a local model via Ollama (#33,
  supersedes the model choice in #30) after the Anthropic key returned 401.
  Verified end to end: streamed, correctly cited answer in ~21s.
- **2026-11-14** — Embeddings moved from Voyage to local
  `qwen3-embedding:0.6b` via Ollama (#34, supersedes #19); collections
  record their embedding model (#35). `voyageai` removed. cb-rag
  re-indexed and search quality compared against the earlier Voyage runs.
