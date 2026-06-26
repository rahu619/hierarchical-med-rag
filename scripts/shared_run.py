"""Shared run-id resolution and lightweight logging helpers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4


def resolve_run_id(explicit_run_id: str = "") -> str:
    if explicit_run_id:
        return explicit_run_id

    env_run_id = os.getenv("RUN_ID", "").strip()
    if env_run_id:
        return env_run_id

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:8]
    return f"run-{timestamp}-{suffix}"


def log_event(run_id: str, level: str, message: str) -> None:
    print(f"[{level}] run_id={run_id} {message}")
