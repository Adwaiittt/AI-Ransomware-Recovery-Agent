# AI Ransomware Recovery Agent

A backup and recovery service. It:

- snapshots a directory to S3-compatible storage
- **detects ransomware-like activity** with a scikit-learn model
- **restores the last clean snapshot** and verifies every file by SHA-256
- answers questions like *"What changed on Tuesday, and is it safe to restore?"* through a **Claude-powered RAG agent** over the backup metadata

FastAPI · boto3 (MinIO / AWS S3) · watchdog · scikit-learn · SQLAlchemy · Anthropic SDK · sentence-transformers + FAISS · Docker Compose

> 📽️ _Demo GIF placeholder: `docs/demo.gif` (record `make demo`)_

## Architecture

```mermaid
flowchart LR
    subgraph host["Watched directory"]
        FS[(files)]
    end
    subgraph watcher["watcher container"]
        WD[watchdog events] --> FE[10 s window features] --> ML[RandomForest detector]
    end
    subgraph api["api container (FastAPI)"]
        BK[POST /backups<br/>snapshotter] 
        SC[POST /detection/scan]
        RS[POST /restore<br/>plan · verify · quarantine]
        AG[POST /agent/ask]
        IDX[RAG index<br/>MiniLM + FAISS]
    end
    S3[(MinIO / S3<br/>objects/sha256<br/>manifests/)]
    DB[(SQLite / Postgres<br/>snapshots · files · incidents<br/>restore_jobs · chunks)]
    CL[[Claude API]]

    FS --> WD
    ML -- incident + mark snapshots suspect --> DB
    FS --> BK -- dedup upload --> S3
    BK --> DB
    SC --> ML
    RS -- download + SHA-256 --> S3
    RS --> FS
    DB --> IDX --> AG
    AG <-- read-only tools --> CL
```

**Recovery flow:** clean snapshot → attack → watcher alerts (incident; later snapshots become `suspect`) → agent explains and *recommends* a snapshot → a human calls `POST /restore` (dry run first) → SHA-256 verified → incident resolved.

## Quick start (Docker)

```bash
cp .env.example .env     # set AWS_SECRET_ACCESS_KEY (>= 8 chars); optionally ANTHROPIC_API_KEY
make up                  # build image (trains the model inside), start minio + api + watcher
make demo                # seed → snapshot → attack → detect → ask agent → restore → verify
```

- API docs: http://localhost:8000/docs
- MinIO console: http://localhost:9001

`make simulate` runs only the safe attack simulator, and `make logs` follows the JSON logs.

### Without make / without Docker (local dev)

```bash
python -m venv .venv && .venv/Scripts/activate        # source .venv/bin/activate on Linux/macOS
pip install -e ".[dev]"
python -m ml.generate_dataset && python -m ml.train   # = make train
docker compose up -d minio                            # or any S3 endpoint (see .env.example)
uvicorn app.main:app                                  # terminal 1
python -m app.detection.monitor                       # terminal 2
python scripts/demo.py --watch-dir sandbox/watched    # terminal 3 (= make demo)
pytest && ruff check .                                # = make test / make lint
```

## Demo output (real run, no API key set)

```text
=== 1. Seed sample files ...         seeded 120 files; false positives from seeding: 0
=== 2. Clean baseline snapshot       snap-20261006T052956Z-f0fc76  state=clean  files=121
=== 3. Simulated ransomware attack   simulated attack (fast) on 120 files
=== 4. Detection                     live monitor: ALERT
  incident #1 score=0.9873 files=241
  top features: mean_entropy_delta=2.94, frac_high_entropy_unexpected=0.99, unknown_ext_count=120
  post-attack snapshot snap-20261006T053013Z-c79858 state=suspect
=== 5. Ask the recovery agent        503 (no key) -> retrieval-only view; top hit: [incident] Incident 1 ...
=== 6. Restore last clean snapshot   dry run: create=120 unchanged=1 extra=121
  job #2 succeeded: restored=120 verified=121/121 failed=0 quarantined=121
=== 7. Verify                        diff clean -> post-restore: all 0, unchanged=121
VERIFIED: tree matches the clean snapshot byte-for-byte
```

With `ANTHROPIC_API_KEY` set, step 5 returns a Markdown answer with four sections: what changed, evidence, risk, recommendation. It also returns the cited snapshot and incident ids, checked against the DB, the validated recommended snapshot, and the list of tool calls the agent made.

## API

| Method | Path | Description |
|---|---|---|
| GET  | `/health` | DB + object store (503 if degraded); model loaded flag |
| POST | `/backups` | Snapshot `WATCH_DIR` (SHA-256 dedup); `suspect` while an incident is open |
| GET  | `/backups`, `/backups/{id}` | List / detail with per-file sha256, entropy, size, mtime |
| GET  | `/backups/{id}/diff/{other}` | added / removed / modified / renamed / **extension_changed** |
| POST | `/detection/scan` | Score last snapshot vs live dir (or two snapshots) |
| GET  | `/detection/incidents` | Incidents: score, top features, affected files |
| POST | `/detection/incidents/{id}/resolve` | Close incident → new snapshots trusted again |
| GET  | `/detection/status` | Model info, watcher heartbeat, open incidents |
| GET  | `/restore/candidates` | Snapshots vs incident + computed **last clean snapshot** |
| POST | `/restore` | `{snapshot_id, target_path?, dry_run=true, force, quarantine_extras}` |
| GET  | `/restore/jobs` | Restore audit log |
| POST | `/agent/ask` | `{question}` → answer + validated citations + recommendation |
| GET  | `/agent/search` | Retrieval only (date parsing + vector search), no LLM call |
| POST | `/agent/reindex` | Force RAG index sync |

Every response carries `X-Request-ID`, and the same id appears on every JSON log line for that request.

## Model metrics

From `ml/artifacts/metrics.json` (`python -m ml.train`, seed 42, reproducible run to run):
- 10,400 synthetic 10-second windows, 23% of them attacks
- 25% held out as the test set
- threshold picked from out-of-fold training predictions, capped at 1% false-positive rate

| Model | Precision | Recall | F1 | ROC-AUC | FPR |
|---|---|---|---|---|---|
| IsolationForest (unsupervised baseline) | 0.967 | 0.635 | 0.767 | 0.965 | 0.65% |
| **RandomForest (selected)** | **0.995** | **0.985** | **0.990** | **0.998** | **0.15%** |
| GradientBoosting | 0.980 | 0.983 | 0.982 | 0.998 | 0.60% |

**Recall on attack families left out of training:**
- partial encryption: 0.988
- in-place stealth: 0.988
- random extension: 1.000

**Weakest cases:**
- most false positives come from ML-checkpoint writes (3.7% of those windows)
- encrypt-copy-then-delete attacks have the lowest recall (95%)

## Safety

- **Simulator** (`simulator/fake_ransomware.py`):
  - runs only inside `./sandbox`; root and home directories are refused, with symlinks resolved first
  - touches only files it seeded itself, and only while their hash is unchanged
  - writes random bytes, never real encryption
  - has no network, process or crypto imports, which a test checks
- **Agent:** has read-only tools only and can never restore. Its recommendation is checked server-side, and a non-clean snapshot is rejected.
- **Restore:**
  - dry run by default; restores to a separate directory unless told otherwise
  - path-traversal guard on every file path
  - atomic file placement with SHA-256 checks
  - extra files are quarantined, never deleted
  - every restore is recorded in an audit log

## Limitations

- **The detector is trained on synthetic data.** It includes hard cases on purpose, but real-world precision will be lower. Encrypting a file that is already compressed (`.docx`, `.jpg`) *in place* barely changes its entropy, and only rate and spread features cover it.
- **Real deployment base rates:** a 0.15% false-positive rate per 10-second window still means roughly 13 false windows per day. Alerts are merged into incidents and restores are human-gated, but a real deployment needs tuning on real telemetry.
- **Snapshots run inside the request.** There is no job queue and no retention or GC policy for blobs.
- **SQLite is shared by two containers.** Fine on one host; use `DATABASE_URL=postgresql://…` beyond that.
- **The agent is only as good as the indexed metadata.** It does not read file contents (by design: privacy and prompt injection).
- **Docker image build is unverified.** It was not run on the author's machine (WSL unavailable); CI builds it.

## Future work

- S3 Object Lock / versioning so blobs survive stolen credentials
- Real telemetry (Sysmon / eBPF / ETW) and per-host baselines; SHAP explanations
- Background job queue for snapshots and restores; blob GC with retention policies
- Postgres + pgvector instead of SQLite + FAISS; multi-tenant auth
- Agent evals: a graded question set built from recorded incidents

See [DESIGN.md](DESIGN.md) for the reasoning behind each design choice.
