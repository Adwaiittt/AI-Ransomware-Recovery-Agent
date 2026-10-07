"""Application configuration loaded from environment variables / .env.

All settings live here so nothing else reads os.environ directly; this keeps
secrets out of code and makes tests able to override config in one place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """Typed runtime settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Storage
    storage_mode: Literal["local", "cloud"] = "local"
    s3_endpoint_url: str | None = "http://127.0.0.1:9000"
    s3_bucket: str = "ransomware-backups"
    s3_region: str = "us-east-1"
    aws_access_key_id: SecretStr | None = None
    aws_secret_access_key: SecretStr | None = None

    # Backup
    watch_dir: Path = Path("./sandbox/watched")
    # NoDecode: read the env value as a raw comma-separated string, not JSON.
    exclude_patterns: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["*.tmp", "*.swp", ".DS_Store", "~$*"]
    )

    # Restore: default target root (restores land in RESTORE_DIR/<snapshot_id>)
    restore_dir: Path = Path("./restore_out")

    # Detection
    model_path: Path = Path("ml/artifacts/model.joblib")
    detection_window_seconds: float = 10.0  # must match ml/generate_dataset.py WINDOW
    # Alerts within this gap of an open incident extend it instead of opening a new one.
    incident_cooldown_seconds: float = 60.0
    # The monitor samples the head of each file; ransomware that encrypts only
    # the first chunk (intermittent encryption) still shows up there.
    entropy_sample_bytes: int = 1024 * 1024

    # Agent (Claude + RAG). The API key is NOT a setting: the Anthropic SDK reads
    # ANTHROPIC_API_KEY (or an `ant auth login` profile) itself, so it never
    # passes through our config objects or logs.
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Server-side refusal fallback ("fallbacks": "default"). Disable for models
    # or platforms that don't support it.
    anthropic_fallbacks: bool = True
    agent_max_turns: int = 8
    agent_top_k: int = 8
    # "all-MiniLM-L6-v2" (sentence-transformers) or "hash" (offline, no download)
    embedding_model: str = "all-MiniLM-L6-v2"
    agent_timezone: str = "UTC"  # day boundaries for "yesterday", "Tuesday", ...

    # Database
    database_url: str = "sqlite:///./data/metadata.db"

    # Dashboard "Lab" tab: lets the UI drive the SAFE simulator. Off by default;
    # docker compose enables it for the local demo.
    enable_lab: bool = False

    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "json"
    # watchdog's inotify backend misses events on some bind mounts (e.g. Windows
    # host folders mounted into Linux containers); polling always works.
    monitor_polling: bool = False

    @field_validator("exclude_patterns", mode="before")
    @classmethod
    def _split_patterns(cls, value: object) -> object:
        if isinstance(value, str):
            return [p.strip() for p in value.split(",") if p.strip()]
        return value

    @property
    def effective_endpoint_url(self) -> str | None:
        """Endpoint passed to boto3: MinIO URL locally, None (real AWS) in cloud mode."""
        return self.s3_endpoint_url if self.storage_mode == "local" else None


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
