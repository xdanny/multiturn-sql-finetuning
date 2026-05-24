"""
Result manifest helpers for reproducible benchmark claims.

A manifest is the public contract behind a reported number: which input was
scored, which output was produced, which model and mode were used, and what the
aggregate metrics were.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str | None:
    """Return the SHA-256 digest for a file, or None when the file is absent."""

    if not path.exists():
        return None

    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit(cwd: Path | None = None) -> str | None:
    """Return the current git commit SHA when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def build_result_manifest(
    *,
    run_id: str,
    benchmark: str,
    input_path: Path | None,
    output_path: Path,
    model_name: str,
    endpoint: str,
    evaluation_mode: str,
    oracle_allowed: bool,
    prompt_variant: str | None,
    database_root: Path | None,
    command: Sequence[str],
    row_count: int,
    metrics: Mapping[str, Any],
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Build a stable JSON-serializable result manifest."""

    return {
        "schema_version": 1,
        "run_id": run_id,
        "benchmark": benchmark,
        "model_name": model_name,
        "endpoint": endpoint,
        "evaluation_mode": evaluation_mode,
        "oracle_allowed": oracle_allowed,
        "prompt_variant": prompt_variant,
        "database_root": str(database_root) if database_root else None,
        "input_path": str(input_path) if input_path else None,
        "input_sha256": sha256_file(input_path) if input_path else None,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "row_count": row_count,
        "metrics": dict(metrics),
        "command": list(command),
        "git_commit": git_commit if git_commit is not None else current_git_commit(Path.cwd()),
    }


def write_result_manifest(manifest: Mapping[str, Any], output: Path) -> None:
    """Write a manifest as sorted, pretty JSON."""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n")
