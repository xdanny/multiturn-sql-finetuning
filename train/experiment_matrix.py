"""Validate the hypothesis-level fine-tuning experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULT_MATRIX_CONFIG = Path("configs/hypothesis_experiment_matrix.yaml")

LIST_FIELDS = {
    "train_artifacts",
    "validation_artifacts",
    "locked_benchmark_artifacts",
    "leakage_checks",
}
REQUIRED_TEXT_FIELDS = {
    "hypothesis_id",
    "method",
    "question",
    "control_hypothesis_id",
    "primary_metric",
    "promotion_gate",
    "current_status",
    "next_gpu_run",
}


def _path_status(repo_root: Path, paths: tuple[str, ...]) -> dict[str, bool]:
    return {
        path: "<" in path or ">" in path or (repo_root / path).exists()
        for path in paths
    }


def _normalize_hypothesis(row: dict[str, Any]) -> dict[str, Any]:
    hypothesis = dict(row)
    hypothesis_id = hypothesis.get("hypothesis_id") or "<unknown hypothesis>"
    missing_fields = [
        field
        for field in sorted(REQUIRED_TEXT_FIELDS)
        if not str(hypothesis.get(field) or "").strip()
    ]
    if missing_fields:
        raise ValueError(f"{hypothesis_id}: missing {', '.join(missing_fields)}")
    for field in LIST_FIELDS:
        values = hypothesis.get(field)
        if values is None:
            values = []
        if not isinstance(values, list):
            raise ValueError(f"{hypothesis_id}: {field} must be a list")
        hypothesis[field] = tuple(str(value) for value in values)
    if not hypothesis["leakage_checks"]:
        raise ValueError(f"{hypothesis_id}: missing leakage_checks")
    if (
        hypothesis_id != "hosted_transfer"
        and not hypothesis["validation_artifacts"]
    ):
        raise ValueError(f"{hypothesis_id}: missing validation_artifacts")
    return hypothesis


def load_experiment_matrix(
    path: Path = DEFAULT_MATRIX_CONFIG,
    *,
    repo_root: Path = Path("."),
) -> tuple[dict[str, Any], ...]:
    """Return validated hypotheses with artifact readiness metadata."""

    payload = yaml.safe_load(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("hypothesis matrix config must use schema_version=1")
    if not str(payload.get("matrix_id") or "").strip():
        raise ValueError("hypothesis matrix config must define matrix_id")
    for field in ("claim_boundary", "holdout_policy"):
        if not str(payload.get(field) or "").strip():
            raise ValueError(f"hypothesis matrix config must define {field}")
    rows = payload.get("hypotheses")
    if not isinstance(rows, list) or not rows:
        raise ValueError("hypothesis matrix config must include hypotheses")

    normalized = []
    seen_ids = {"base_model", "best_local_proxy_method"}
    for raw_row in rows:
        row = _normalize_hypothesis(raw_row)
        hypothesis_id = row["hypothesis_id"]
        if hypothesis_id in seen_ids:
            raise ValueError(f"{hypothesis_id}: duplicate hypothesis_id")
        control_id = row["control_hypothesis_id"]
        if control_id not in seen_ids:
            raise ValueError(
                f"{hypothesis_id}: control_hypothesis_id must reference an earlier "
                f"hypothesis or external control, got {control_id}"
            )
        row["train_artifact_status"] = _path_status(repo_root, row["train_artifacts"])
        row["validation_artifact_status"] = _path_status(
            repo_root, row["validation_artifacts"]
        )
        row["locked_benchmark_artifact_status"] = _path_status(
            repo_root, row["locked_benchmark_artifacts"]
        )
        row["ready_for_validation"] = bool(row["validation_artifacts"]) and all(
            row["train_artifact_status"].values()
        ) and all(row["validation_artifact_status"].values())
        row["ready_for_locked_benchmark"] = (
            bool(row["locked_benchmark_artifacts"])
            and row["ready_for_validation"]
            and all(row["locked_benchmark_artifact_status"].values())
        )
        normalized.append(row)
        seen_ids.add(hypothesis_id)
    return tuple(normalized)


def experiment_matrix_summary(
    *,
    path: Path = DEFAULT_MATRIX_CONFIG,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Return a JSON-serializable matrix summary."""

    payload = yaml.safe_load(path.read_text())
    hypotheses = load_experiment_matrix(path, repo_root=repo_root)
    return {
        "schema_version": 1,
        "matrix_id": payload["matrix_id"],
        "claim_boundary": payload["claim_boundary"],
        "holdout_policy": payload["holdout_policy"],
        "hypothesis_count": len(hypotheses),
        "ready_for_validation_count": sum(
            1 for row in hypotheses if row["ready_for_validation"]
        ),
        "ready_for_locked_benchmark_count": sum(
            1 for row in hypotheses if row["ready_for_locked_benchmark"]
        ),
        "hypotheses": [
            {
                "hypothesis_id": row["hypothesis_id"],
                "method": row["method"],
                "control_hypothesis_id": row["control_hypothesis_id"],
                "primary_metric": row["primary_metric"],
                "current_status": row["current_status"],
                "ready_for_validation": row["ready_for_validation"],
                "ready_for_locked_benchmark": row["ready_for_locked_benchmark"],
                "missing_train_artifacts": [
                    path
                    for path, exists in row["train_artifact_status"].items()
                    if not exists
                ],
                "missing_validation_artifacts": [
                    path
                    for path, exists in row["validation_artifact_status"].items()
                    if not exists
                ],
                "missing_locked_benchmark_artifacts": [
                    path
                    for path, exists in row[
                        "locked_benchmark_artifact_status"
                    ].items()
                    if not exists
                ],
                "promotion_gate": row["promotion_gate"],
                "next_gpu_run": row["next_gpu_run"],
            }
            for row in hypotheses
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_MATRIX_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = experiment_matrix_summary(path=args.config)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote experiment matrix summary to {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
