"""Thin wrapper over boto3 that works with both MinIO and AWS S3.

Key layout (content-addressed):

    objects/<sha[:2]>/<sha256>      immutable file blobs
    manifests/<snapshot_id>.json    per-snapshot manifest

Why content addressing: an object's key *is* its hash, so (1) identical content
is stored once across all snapshots (dedup for free), and (2) a blob can never
be silently replaced — ransomware rewriting a file produces a *new* hash/key and
the clean blob from earlier snapshots is untouched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, BinaryIO

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from app.config import Settings

logger = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """Raised when the object store is unreachable or returns an unexpected error."""


def blob_key(sha256: str) -> str:
    """Return the object key for a content hash (2-char fan-out prefix)."""
    return f"objects/{sha256[:2]}/{sha256}"


def manifest_key(snapshot_id: str) -> str:
    """Return the object key for a snapshot manifest."""
    return f"manifests/{snapshot_id}.json"


class S3Storage:
    """Object storage operations used by snapshot/restore code."""

    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self.bucket = bucket

    @classmethod
    def from_settings(cls, settings: Settings) -> S3Storage:
        """Build a client from settings: MinIO when an endpoint is set, else AWS."""
        kwargs: dict[str, Any] = {
            "region_name": settings.s3_region,
            # Path-style addressing is required by MinIO and harmless on AWS.
            "config": Config(
                s3={"addressing_style": "path"},
                retries={"max_attempts": 3},
                # Fail fast when the store is down instead of hanging an API request
                # for botocore's 60 s default connect timeout.
                connect_timeout=5,
                read_timeout=60,
                max_pool_connections=16,  # >= snapshot upload workers
            ),
        }
        if settings.effective_endpoint_url:
            kwargs["endpoint_url"] = settings.effective_endpoint_url
        # Only pass explicit keys if configured; otherwise fall back to the default
        # AWS credential chain (env, profile, IAM role) — the right thing in cloud.
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = settings.aws_access_key_id.get_secret_value()
            kwargs["aws_secret_access_key"] = settings.aws_secret_access_key.get_secret_value()
        return cls(boto3.client("s3", **kwargs), settings.s3_bucket)

    # -- bucket -----------------------------------------------------------------
    def ensure_bucket(self) -> None:
        """Create the bucket if it does not already exist."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in {"404", "NoSuchBucket", "NotFound"}:
                raise StorageError(f"Cannot access bucket {self.bucket!r}: {exc}") from exc
        except Exception as exc:  # connection refused, DNS, etc.
            raise StorageError(f"Object store unreachable: {exc}") from exc

        create_kwargs: dict[str, Any] = {"Bucket": self.bucket}
        region = self._client.meta.region_name
        # us-east-1 must NOT send a LocationConstraint (S3 quirk).
        if region and region != "us-east-1":
            create_kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
        self._client.create_bucket(**create_kwargs)
        logger.info("Created bucket %s", self.bucket)

    def ping(self) -> bool:
        """Return True if the bucket is reachable (used by health checks)."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False

    # -- objects ----------------------------------------------------------------
    def object_exists(self, key: str) -> bool:
        """Return True if ``key`` exists in the bucket."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise StorageError(str(exc)) from exc

    def upload_file(self, local_path: Path, key: str, sha256: str) -> None:
        """Upload a local file; the hash is stored as object metadata for later verification."""
        self._client.upload_file(
            str(local_path), self.bucket, key, ExtraArgs={"Metadata": {"sha256": sha256}}
        )

    def upload_fileobj(self, fileobj: BinaryIO, key: str) -> None:
        """Upload from an open binary stream."""
        self._client.upload_fileobj(fileobj, self.bucket, key)

    def download_file(self, key: str, local_path: Path) -> None:
        """Download ``key`` to ``local_path`` (parent dirs are created)."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(local_path))

    def put_json(self, key: str, payload: dict[str, Any]) -> None:
        """Serialize ``payload`` as JSON and store it at ``key``."""
        body = json.dumps(payload, indent=2, default=str).encode("utf-8")
        self._client.put_object(
            Bucket=self.bucket, Key=key, Body=body, ContentType="application/json"
        )

    def get_json(self, key: str) -> dict[str, Any]:
        """Fetch and parse a JSON object."""
        obj = self._client.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read())

    def list_keys(self, prefix: str = "") -> list[str]:
        """List all keys under ``prefix`` (handles pagination)."""
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return keys
