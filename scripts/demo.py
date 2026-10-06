"""End-to-end demo: seed -> snapshot -> attack -> detect -> ask agent -> restore -> verify.

Runs against a live API (default http://localhost:8000). Inside the compose
stack:  `make demo`  (= docker compose exec api python scripts/demo.py)
Locally: start the API + monitor, then  `python scripts/demo.py`.

Uses only the standard library (urllib) so it runs in the slim runtime image.
Exits non-zero if the final verification fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any

SIM = [sys.executable, "-m", "simulator.fake_ransomware"]


def step(n: int, title: str) -> None:
    print(f"\n=== {n}. {title} " + "=" * max(0, 60 - len(title)), flush=True)


class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def call(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "X-Request-ID": "demo"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return resp.status, self._parse(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, self._parse(exc.read())

    @staticmethod
    def _parse(raw: bytes) -> Any:
        try:
            return json.loads(raw or b"null")
        except ValueError:  # e.g. a plain-text 500 page
            return {"detail": raw.decode(errors="replace")[:300]}

    def ok(self, method: str, path: str, body: Any = None) -> Any:
        status, data = self.call(method, path, body)
        if status >= 400:
            raise SystemExit(f"{method} {path} -> {status}: {data}")
        return data


def wait_healthy(api: Api, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if api.call("GET", "/health")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(2)
    raise SystemExit(f"API at {api.base} not healthy after {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--api", default=os.environ.get("DEMO_API", "http://localhost:8000"))
    parser.add_argument("--watch-dir", default="sandbox/watched")
    parser.add_argument("--files", type=int, default=120)
    parser.add_argument("--window", type=float, default=10.0, help="monitor window seconds")
    args = parser.parse_args()
    api = Api(args.api)

    step(0, "Preflight")
    wait_healthy(api)
    # The watcher writes its first heartbeat after one window; give it up to three.
    deadline = time.time() + 3 * args.window + 2
    status = api.ok("GET", "/detection/status")
    while not status["monitor"]["alive"] and time.time() < deadline:
        time.sleep(2)
        status = api.ok("GET", "/detection/status")
    print(f"model loaded: {status['model_loaded']} | monitor alive: {status['monitor']['alive']}")
    for inc in api.ok("GET", "/detection/incidents?status=open"):  # leftovers from earlier runs
        api.ok("POST", f"/detection/incidents/{inc['id']}/resolve")
        print(f"resolved leftover incident #{inc['id']}")

    step(1, "Seed sample files (simulator, sandbox only)")
    subprocess.run([*SIM, "clean", args.watch_dir], check=False, capture_output=True)
    subprocess.run([*SIM, "seed", args.watch_dir, "--files", str(args.files)], check=True)
    if status["monitor"]["alive"]:
        print(f"letting the monitor score the seeding burst ({2 * args.window:.0f}s)...")
        time.sleep(2 * args.window + 2)
        seeded_alerts = api.ok("GET", "/detection/incidents?status=open")
        print(f"false positives from seeding: {len(seeded_alerts)}")
        for inc in seeded_alerts:
            api.ok("POST", f"/detection/incidents/{inc['id']}/resolve")

    step(2, "Clean baseline snapshot")
    clean = api.ok("POST", "/backups", {"label": "demo-baseline"})
    print(f"{clean['id']}  state={clean['state']}  files={clean['file_count']}")

    step(3, "Simulated ransomware attack")
    attack_start = datetime.now(UTC)
    subprocess.run([*SIM, "attack", args.watch_dir, "--mode", "fast"], check=True)

    step(4, "Detection")
    incident = None
    if status["monitor"]["alive"]:
        deadline = time.time() + 4 * args.window + 5
        while time.time() < deadline and incident is None:
            time.sleep(2)
            for inc in api.ok("GET", "/detection/incidents?status=open"):
                if datetime.fromisoformat(inc["ended_at"]) >= attack_start:
                    incident = inc
        print("live monitor:", "ALERT" if incident else "no alert yet")
    if incident is None:
        scan = api.ok("POST", "/detection/scan", {"base_snapshot_id": clean["id"]})
        print(f"on-demand scan verdict={scan['verdict']} max_score={scan['max_score']}")
        incident = scan["incident"]
    if incident is None:
        print("DETECTION FAILED: no incident recorded")
        return 1
    top = ", ".join(f"{f['feature']}={f['value']}" for f in incident["top_features"])
    print(
        f"incident #{incident['id']} score={incident['score']} "
        f"files={incident['affected_file_count']}"
    )
    print(f"top features: {top}")
    suspect = api.ok("POST", "/backups", {"label": "post-attack"})
    print(f"post-attack snapshot {suspect['id']} state={suspect['state']}")

    step(5, "Ask the recovery agent")
    code, ans = api.call(
        "POST", "/agent/ask", {"question": "What changed today, and is it safe to restore?"}
    )
    if code == 200:
        print(ans["answer"])
        print(f"\nrecommended (validated): {ans['recommended_snapshot_id']}")
        print(f"tool calls: {[c['name'] for c in ans['tool_calls']]}")
        if ans["unverified_ids"]:
            print(f"unverified ids flagged: {ans['unverified_ids']}")
    else:
        print(f"agent unavailable ({code}): {ans.get('detail') if ans else ans}")
        print("(set ANTHROPIC_API_KEY to enable). Retrieval-only view:")
        hits = api.ok("GET", "/agent/search?q=ransomware%20incident%20today&k=3")["hits"]
        for h in hits:
            print(f"  [{h['kind']}] {h['text'][:140]}...")

    step(6, "Restore last clean snapshot (dry run, then in place)")
    cand = api.ok("GET", "/restore/candidates")
    target_id = cand["last_clean_snapshot_id"]
    print(f"last clean snapshot before incident: {target_id}")
    watch_abs = os.path.abspath(args.watch_dir)
    dry = api.ok("POST", "/restore", {"snapshot_id": target_id, "target_path": watch_abs})
    print(f"dry run: {dry['summary']}")
    real = api.ok(
        "POST",
        "/restore",
        {
            "snapshot_id": target_id,
            "target_path": watch_abs,
            "dry_run": False,
            "quarantine_extras": True,
        },
    )
    job = real["job"]
    print(
        f"job #{job['id']} {job['status']}: restored={job['files_restored']} "
        f"verified={job['files_verified']}/{job['files_total']} failed={job['files_failed']} "
        f"quarantined={len(real['quarantined'])}"
    )

    step(7, "Verify")
    api.ok("POST", f"/detection/incidents/{incident['id']}/resolve")
    after = api.ok("POST", "/backups", {"label": "post-restore"})
    diff = api.ok("GET", f"/backups/{target_id}/diff/{after['id']}")["summary"]
    changed = {k: v for k, v in diff.items() if k not in ("unchanged", "mean_entropy_delta") and v}
    verified = job["files_verified"] == job["files_total"]
    ok = job["status"] == "succeeded" and verified and not changed
    print(f"diff clean -> post-restore: {diff}")
    print("VERIFIED: tree matches the clean snapshot byte-for-byte" if ok else "VERIFY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
