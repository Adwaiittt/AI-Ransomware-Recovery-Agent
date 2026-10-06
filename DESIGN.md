# Design Notes

This document explains the *why* behind non-obvious decisions, phase by phase.

## Phase 1 — Storage + snapshots

### Content-addressed object layout
```
objects/<sha[:2]>/<sha256>      immutable blobs
manifests/<snapshot_id>.json    one manifest per snapshot
```
- **Dedup for free.** A snapshot uploads a file only if its hash isn't stored yet,
  so unchanged files and duplicate files cost nothing after the first upload. This
  works the same way as git's object store and restic/borg backups.
- **Clean blobs can't be overwritten.** When ransomware encrypts `report.docx`, its
  hash changes, so the encrypted version is written to a *new* key. The clean blob
  that earlier snapshots point to stays as it was. Restore (Phase 3) depends on this.
- **`sha[:2]` fan-out** keeps any single prefix small, which helps S3 listing
  and request-rate partitioning.
- *Hardening idea:* turn on S3 Object Lock / bucket versioning so that even
  stolen credentials can't delete blobs. This is listed as future work.

### Dedup check: DB first, then bucket
For each new hash we first check SQLite, which takes one batched `IN` query and no
network calls. Only hashes the DB doesn't know get a `HEAD` request to S3. The
bucket check keeps things correct if the DB is lost while the bucket survives;
the DB check keeps unchanged files from costing one HEAD request each.

### Write order: blobs -> manifest -> DB commit
The DB is the index the API serves from. Committing it last means the DB never
points to an object that hasn't been uploaded. If the process crashes midway,
some blobs may be left without a reference. Because they are content-addressed,
they are harmless and a later snapshot will reuse them.

### Hash and entropy in one streaming pass
Each file is read once, in 1 MiB chunks. Every chunk updates SHA-256 and a
256-bin byte histogram (`numpy.bincount`). Memory use stays the same no matter
how big the file is.
**Shannon entropy** `H = -Σ p·log2 p` ranges over 0-8 bits/byte. Text is about
4-5, while compressed, encrypted, or random data is about 7.9 or higher. That
difference is the main ransomware signal, but zip and jpg files are also high
entropy. This is why Phase 2 uses *entropy deltas* and extension changes rather
than raw entropy alone.

### Diff classification
`diff_snapshots` reports `added / removed / modified / renamed / extension_changed`:
- **renamed** = a removed path and an added path with the *same* hash.
- **extension_changed** = `report.docx` → `report.docx.locked` (appended suffix)
  or `photo.jpg` → `photo.enc` (replaced suffix), whether or not the content
  changed. This is the classic ransomware pattern, so it gets its own category
  instead of appearing as add+remove noise. The detector and the agent both
  use it directly.

### Snapshot IDs
`snap-20261005T120000Z-1a2b3c`: sorts by time, is easy to read in logs and in
agent citations, and the random suffix prevents collisions within one second.

### Snapshot state
Every snapshot starts as `clean`. The detector (Phase 2) changes post-incident
snapshots to `suspect`, and a confirmed attack makes them `infected`. Only
`clean` snapshots can be "last clean" restore candidates.

### Security choices
- `POST /backups` doesn't accept a path. The source is the server-configured
  `WATCH_DIR`, so the API can't be used to read arbitrary directories.
- Symlinks are skipped while scanning, so a link can't pull files from outside
  the watched tree into a backup (or, later, a restore).
- In cloud mode with no keys configured, boto3 uses the default credential chain
  (IAM role or profile) instead of requiring static keys.

### Timezones
SQLite has no timezone-aware type. A `UTCDateTime` TypeDecorator stores UTC
and always returns timezone-aware datetimes. Without it, comparing incident and
snapshot times in Phase 3 could mix naive and aware values.

### Sync handlers
The backup endpoints are plain `def`. boto3 and file hashing block, and FastAPI
runs sync handlers in a threadpool, which keeps the event loop free.
A process-level lock stops two snapshots from running at the same time.
Known limitation: a snapshot runs inside the request. A background job queue
could be added later.

## Phase 2 — Feature engineering + ML detection

### One feature function for training, scanning, and live monitoring
`app/detection/features.py::compute_window_features` is the only code that
turns events into features. The dataset generator simulates *event streams*
and calls it. The watchdog monitor and the snapshot-diff scan also build events
and call it. If training and serving computed features separately, they could
drift apart without anyone noticing (train/serve skew); sharing one function
rules that out.

### Why simulate events instead of sampling feature values
If you sample features directly, you have to invent how they relate to each
other. When you simulate events, the relationships come out naturally: a
rename burst raises `renames_per_sec`, `pct_extension_changed` and
`unknown_ext_count` together, just as it does on a real disk.

### Features (10 s window)
| feature | why |
|---|---|
| files_written_per_sec | bursty mass rewrite |
| renames_per_sec, pct_extension_changed | `.docx -> .docx.locked` |
| mean_entropy_delta | text (~4.5) -> ciphertext (~8). Needs a known "before" value |
| frac_high_entropy | share of written files above 7.5 bits/byte |
| **frac_high_entropy_unexpected** | same, but leaves out formats that are compressed by design (zip/jpg/docx/...). This is the main way the model avoids false positives |
| unknown_ext_count | new names with suffixes nobody normally uses |
| delete_create_ratio | "write encrypted copy, delete original" |
| unique_dirs | how widely the activity spreads across folders |
| events_per_sec | raw event rate |

Writes are de-duplicated per path within a window, because one save fires
several `modified` events. Entropy is measured at the file's **final name**,
so a write followed by a rename is judged as `.locked`, not `.txt`.

### The dataset is built to be hard on purpose
The first version reached F1 = 1.000, which only showed the simulation was
too easy. The current version adds:
- **Hard negatives:** browser cache (extensionless, high entropy),
  ML checkpoints (`.pt/.npy` rewritten at high entropy), zip-then-delete,
  cloud-sync `.partial` renames, log rotation (`.log -> .log.1`), legitimate
  `.gpg` backups, and photo editing.
- **Hard positives:** slow (1-4 files/window), in-place (no rename), partial
  encryption (entropy only reaches ~6.5-7.6), and random per-file extensions.
- **Telemetry degradation:** entropy noise, 25% of rewrites with an unknown
  "before" value (cache miss), attacks that start partway through a window,
  and small encrypted files with entropy below 7.9 (log2(n) caps it).

### Evaluation protocol
- The train/test split is stratified by **scenario**, so every hard case
  appears in both splits.
- The decision threshold comes from **5-fold out-of-fold predictions on the
  training set**: the highest F1 with FPR ≤ 1%. The cut sits midway between
  neighbouring scores. The test set is used once, for reporting only.
- **IsolationForest** (fit on benign data only) is the "no labelled attacks"
  baseline. Its threshold is the 99th percentile of benign scores.
- **Unseen-family test:** the selected model is retrained with one attack
  family removed, and recall is measured on that family. This approximates
  how the model handles "ransomware we've never seen".
- `ml/train.py` is deterministic: two runs produce identical metrics.json.

### Caveats to state in an interview
- The data is synthetic. Real-world precision will be lower, and the base rate
  of attacks in production is far lower than 23%. Even a 0.15% FPR per window
  means roughly 13 false alerts per day at one window every 10 s. That is why
  alerts are merged into incidents (cooldown) and why restore needs a human.
- Encrypting a `.docx` or `.jpg` *in place* barely changes its entropy (it was
  already ~7.9) and leaves no extension signal. That is the main blind spot,
  covered only by rate and spread features.

### Explaining an alert
`importance × max(0, z-score vs. benign mean)` for each feature, top 3. It is a
cheap stand-in for SHAP with no extra dependency, and good enough to tell an
analyst (and, in Phase 4, the agent) *why* the model fired.

### Live monitor
- watchdog thread → locked buffer → drained once per window in the main loop.
- Entropy is sampled from the **first 1 MiB** of each written file, once per
  path per window. Many real families use intermittent encryption that hits
  the start of the file, so the head is the right sample.
- The `entropy_before` cache is seeded from the latest snapshot, then kept up
  to date from events (moves carry their entry along, deletes drop it).
- **Incident merging:** an alert within `INCIDENT_COOLDOWN_SECONDS` of an open
  incident extends that incident instead of opening a new one. A real attack
  spans many windows.
- **Snapshot tainting:** when an incident opens, clean snapshots taken at or
  after its start become `suspect`, and new snapshots stay `suspect` while any
  incident is open. Snapshots are still *taken*, since they are evidence, but
  they are never offered as restore points.
- **Heartbeat row:** the watcher runs in its own process or container, so it
  writes a heartbeat that `/detection/status` reads to decide whether it is
  alive.

### On-demand scan
`POST /detection/scan` diffs a snapshot against the live directory, rebuilds
events from the diff, and uses **mtimes** as timestamps. It then scores them
in the same 10 s windows. 145 files encrypted within one second still look
like a burst, while a day of normal edits spreads across many quiet windows.

### Simulator safety (enforced *and* tested)
- The resolved target must be inside `<project>/sandbox`. The filesystem root
  and the home directory are refused, and symlinks are resolved before the
  check.
- The simulator only touches files listed in its own `.sim_manifest.json`
  **whose hash still matches**. In the live run, 5 files the user had edited
  were skipped.
- It overwrites files with `os.urandom` and does no real cryptography. A test
  parses the module's AST to prove there are no network, process, or crypto
  imports.

### Bug found by the live run
On Windows, `localhost` resolves to IPv6 first. Against an IPv4-only server,
each S3 connection stalled for about 2 s, and a 150-file snapshot took around
10 minutes. Fixes:
- the endpoint is now `127.0.0.1`
- boto has explicit connect and read timeouts, so a down store fails fast
- uploads run in an 8-thread pool

## Phase 3 — Restore

### "Last clean snapshot"
The newest snapshot in state `clean` created **strictly before** the start of
the reference incident. The reference incident is either passed explicitly or
is the *earliest open* incident, because anything after the first sign of
compromise is untrusted. The rule is a pure function, `select_last_clean`, so it
can be unit-tested without a DB. Covered cases:
- a snapshot taken at exactly the incident start is not "before" it
- suspect and infected snapshots are never chosen, even if older
- if no snapshot qualifies, the result is `None` (no guessing)

### Safety model
| risk | mitigation |
|---|---|
| overwriting live data by mistake | default target is a separate `RESTORE_DIR/<snapshot_id>`; in-place only when the target is exactly `WATCH_DIR` |
| running for real by accident | `dry_run` defaults to **true** |
| restoring an infected point | non-clean snapshots need `force: true` |
| path traversal from a tampered DB/manifest (`../../x`) | `safe_join` resolves every destination and requires it to stay inside the target |
| half-written or corrupted file | download to a temp file → verify SHA-256 → `os.replace` (atomic) → re-hash. A bad blob is reported and never placed |
| destroying evidence | files not in the snapshot (`*.locked`, ransom notes) are never deleted; with `quarantine_extras` they are *moved* to `RESTORE_DIR/quarantine/job-N/` |
| "what happened?" afterwards | every restore, dry runs included, is a `restore_jobs` audit row |

Original mtimes are re-applied, so a restored tree diffs as *unchanged*
against the clean snapshot. The demo uses exactly that as its final check.

## Phase 4 — Claude RAG agent

### Corpus
Each piece of metadata becomes one text chunk with a timestamp:
- snapshot summaries
- diffs between consecutive snapshots, with concrete evidence (paths, entropy before → after)
- incidents (score, top features, affected files)
- restore jobs

Chunks are keyed (`incident:3`, `diff:a->b`), so re-indexing is idempotent.

### Incremental indexing, no index file
Chunk text is rebuilt from the DB, which is cheap. Only chunks whose
`text_hash` changed are re-embedded, which is the expensive step. Embeddings
are stored in SQLite, and the FAISS `IndexIDMap2(IndexFlatIP)` is a disposable
in-memory cache rebuilt after each sync. There is no second source of truth to
corrupt or keep in sync.

### Freshness across processes
The watcher records incidents from another container. Before each query the
API compares a cheap DB fingerprint (row counts, latest timestamps, state
tallies) with the one from its last sync, and re-indexes if it changed.
Snapshot, scan, restore and resolve endpoints also schedule a background
re-index after responding.

### Date-filtered retrieval
`parse_time_range` turns phrases into half-open UTC ranges, computing day
boundaries in `AGENT_TIMEZONE`:
- "today", "yesterday", "last night"
- "Tuesday" (most recent, including today) and "last Tuesday" (strictly before today)
- "this/last week", "this/last month"
- "past N hours", ISO dates

Filtering runs **inside FAISS** through an `IDSelectorBatch` built from an
indexed SQL range query. A narrow window therefore returns the top-k *within
the window*, not the global top-k minus whatever falls outside it. If the
window is empty, the agent is told so and gets the closest matches from any
time.

### Embeddings
- **all-MiniLM-L6-v2** is the production embedder. In a real metadata DB it
  ranks the incident first for "was there an attack?" (0.21 vs 0.18) and for
  "ransomware incident locked files" (0.70 vs 0.40).
- **HashingEmbedder** is a feature-hashing bag of words for tests, CI and
  offline use. It is deterministic and needs no download, but it matches
  keywords only, so its tests check mechanics rather than semantic ranking.
  It splits tokens on non-alphanumerics. A test caught that keeping
  `doc.txt.locked` as one token meant "locked" never matched.

### Agent loop
- **Manual tool loop, not the beta tool runner.** It gives a hard turn cap,
  a per-request DB session inside the tools, an audit list of tool calls in
  the response, and trivial mocking.
- **Four read-only tools:** `search_metadata`, `get_snapshot_diff`,
  `list_incidents`, `get_restore_candidates`. **There is no restore tool.**
  "Only recommend" is enforced by what the model *can* do, not by asking
  it nicely.
- **Tool inputs are validated with Pydantic**, since model output is untrusted
  input. Errors go back as `is_error` tool results, and Claude self-corrects.
- **History is append-only.** The full `response.content`, thinking blocks
  included, is appended unchanged, and all tool results for a turn go back in
  one user message.
- **Prompt-injection defence:** file names and labels come from the monitored
  disk, which ransomware controls. Retrieved text is wrapped in
  `<retrieved_context>`, and the system prompt says to treat it as data.
- **Model settings:**
  - model from `ANTHROPIC_MODEL` (default `claude-opus-5-5`)
  - `output_config.effort` configurable (default `medium`)
  - server-side refusal fallback (`fallbacks: "default"`), switchable off with `ANTHROPIC_FALLBACKS=false`
  - stable system prompt with `cache_control`
  - `tool_choice: auto`; forced tool choice is rejected by this model. When the
    turn cap is hit, one final call with `tool_choice: none` forces an answer.

### Trust but verify the answer
The model must end its answer with `RECOMMENDED_SNAPSHOT: <id|NONE>`. The
server then:
- extracts every cited snapshot and incident id and checks each against the DB
- reports unknown ids as `unverified_ids` (hallucination flag)
- **rejects** a recommended snapshot that doesn't exist or isn't `clean`, and
  returns a warning instead

Credential problems map to 503 and upstream errors to 502. The live demo found
that anthropic 1.x signals missing credentials with a plain `TypeError`; that
case is now handled and has a regression test.

## Phase 5 — Packaging & operations

### Image
- One image runs both api and watcher; only the command differs.
- Multi-stage build on `python:3.12-slim`, running as non-root uid 10001.
- **CPU-only torch** (the default Linux wheel ships CUDA, about 2.5 GB more).
- The dependency layer is installed from the `pyproject.toml` dependency list
  alone, so code edits don't invalidate it.
- The detector is **trained during the build**. Training is seeded, so every
  image has the same model.
- MiniLM is pre-downloaded and `HF_HUB_OFFLINE=1` is set, so runtime needs no
  network.

### Compose
- Shared state lives in **named volumes**, not host bind mounts:
  - inotify events cross containers on the same volume, but not reliably from
    a Windows or macOS host folder
  - the non-root user can always write to a named volume
- `MONITOR_POLLING=true` switches watchdog to polling for bind-mount setups.
- Healthchecks:
  - minio: `mc ready`
  - api: `/health` (DB + S3)
  - watcher: `python -m app.detection.monitor --check` (heartbeat age ≤ 3 windows)
- The watcher starts only after the api is healthy, because the api creates
  the schema.

### Observability
- Every log line is JSON from the API and the worker.
- A pure-ASGI middleware assigns a request id: it reuses an incoming
  `X-Request-ID` if it matches a safe charset, otherwise it generates one.
  - The id is echoed in the response and attached to every log line in that
    request, including threadpool endpoints and background tasks.
  - Pure ASGI, not `BaseHTTPMiddleware`, because the context variable must
    reach those code paths.
- uvicorn's own access log is off, so the middleware's one log line per
  request is the only access log.

### SQLite shared by two containers
SQLite runs in WAL mode on a shared volume (same kernel), which is fine for one
writer per table at laptop scale. `DATABASE_URL` accepts Postgres for anything
bigger; the code uses only portable SQLAlchemy.
