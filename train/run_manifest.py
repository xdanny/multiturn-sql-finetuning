"""Training-run manifest helpers for finetuning provenance."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.result_manifest import current_git_commit, sha256_file


def _stage_from_target(
    *,
    expected_training_target: str | None,
    inferred_training_target: str | None,
) -> str | None:
    return expected_training_target or inferred_training_target


def build_training_run_manifest(
    *,
    run_id: str,
    status: str,
    config_path: Path,
    output_dir: Path,
    final_dir: Path,
    train_data_path: Path,
    eval_data_path: Path | None,
    train_row_count: int,
    eval_row_count: int,
    benchmark: str | None,
    training_target: str | None,
    evaluation_mode: str | None,
    expected_training_target: str | None,
    expected_evaluation_mode: str | None,
    expected_benchmark: str | None,
    command: Sequence[str],
    git_commit: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "stage": _stage_from_target(
            expected_training_target=expected_training_target,
            inferred_training_target=training_target,
        ),
        "benchmark": expected_benchmark or benchmark,
        "training_target": expected_training_target or training_target,
        "evaluation_mode": expected_evaluation_mode or evaluation_mode,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "train_data_path": str(train_data_path),
        "train_data_sha256": sha256_file(train_data_path),
        "eval_data_path": str(eval_data_path) if eval_data_path else None,
        "eval_data_sha256": sha256_file(eval_data_path) if eval_data_path else None,
        "train_row_count": train_row_count,
        "eval_row_count": eval_row_count,
        "expected_training_target": expected_training_target,
        "expected_evaluation_mode": expected_evaluation_mode,
        "expected_benchmark": expected_benchmark,
        "output_dir": str(output_dir),
        "final_dir": str(final_dir),
        "command": list(command),
        "git_commit": git_commit if git_commit is not None else current_git_commit(Path.cwd()),
    }


def write_training_run_manifest(manifest: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(manifest), indent=2, sort_keys=True) + "\n")
