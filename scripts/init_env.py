"""Create .env from .env.example, filling the local object-store credentials.

Works on Windows, macOS and Linux (no `make` needed):

    python scripts/init_env.py

Never overwrites an existing .env. The generated secret is random per machine
and is never printed.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / ".env.example"
TARGET = ROOT / ".env"


def main() -> int:
    if TARGET.exists():
        print(".env already exists - left unchanged.")
        return 0
    lines = []
    for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
        if line == "AWS_ACCESS_KEY_ID=":
            line = "AWS_ACCESS_KEY_ID=local-admin"
        elif line == "AWS_SECRET_ACCESS_KEY=":
            line = f"AWS_SECRET_ACCESS_KEY={secrets.token_urlsafe(32)}"
        lines.append(line)
    TARGET.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Created .env with a random object-store secret.")
    print("Optional: add ANTHROPIC_API_KEY to enable the agent; ENABLE_LAB=true for the Lab tab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
