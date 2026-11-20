"""Environment settings and configuration loader for external LLM endpoints."""
from __future__ import annotations

import os
from pathlib import Path


def _load_env() -> None:
    """Read KEY=VALUE lines from .env (if there is one) without overriding real environment variables."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def settings() -> tuple[str, str, str]:
    """Retrieves endpoint base URL, API key, and model name from environment."""
    _load_env()
    return (
        os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1",
        os.environ.get("LLM_API_KEY", ""),
        os.environ.get("LLM_MODEL", ""),
    )
