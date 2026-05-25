"""
Compare generated-history rollout against matching teacher-forced evaluation.

This does not run either evaluation. It verifies two result manifests are
comparable and writes an augmented rollout manifest carrying the comparison
metrics required by the claim ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_GENERATED_SQL_ROLLOUT = "model_generated_sql_rollout"
GOLD_SQL_TEACHER_FORCED = "gold_sql_teacher_forced"


def _metric(manifest: dict[str, Any], name: str) -> float:
    metrics = manifest.get("metrics") or {}
    value = metrics.get(name)
    if value is None:
        raise ValueError(f"manifest {manifest.get('run_id')} is missing metric {name}")
    return float(value)


def _validate_non_oracle(manifest: dict[str, Any], *, label: str) -> None:
    if manifest.get("oracle_allowed") or manifest.get("evaluation_mode") == "oracle_planner_diagnostic":
        raise ValueError(f"{label} manifest must be non-oracle")


def _validate_rollout_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="rollout")
    if manifest.get("benchmark") != "prepared_rollout":
        raise ValueError("rollout manifest must use benchmark=prepared_rollout")
    history_policy = (manifest.get("metrics") or {}).get("history_policy")
    if history_policy != MODEL_GENERATED_SQL_ROLLOUT:
        raise ValueError(
            "rollout manifest must use history_policy=model_generated_sql_rollout"
        )


def _validate_teacher_forced_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="teacher-forced")
    if manifest.get("benchmark") != "prepared":
        raise ValueError("teacher-forced manifest must use benchmark=prepared")
    history_policy = (manifest.get("metrics") or {}).get("history_policy")
    if history_policy != GOLD_SQL_TEACHER_FORCED:
        raise ValueError(
            "teacher-forced manifest must use history_policy=gold_sql_teacher_forced"
        )


def compare_rollout_manifests(
    *,
    rollout_manifest: dict[str, Any],
    teacher_forced_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Return an augmented rollout manifest with teacher-forced comparison metrics."""

    _validate_rollout_manifest(rollout_manifest)
    _validate_teacher_forced_manifest(teacher_forced_manifest)
    if rollout_manifest.get("model_name") != teacher_forced_manifest.get("model_name"):
        raise ValueError("rollout and teacher-forced manifests must use the same model")
    if rollout_manifest.get("input_sha256") != teacher_forced_manifest.get("input_sha256"):
        raise ValueError("rollout and teacher-forced manifests must use the same input")

    rollout_value = _metric(rollout_manifest, "value_execution_accuracy")
    rollout_strict = _metric(rollout_manifest, "strict_execution_accuracy")
    teacher_value = _metric(teacher_forced_manifest, "value_execution_accuracy")
    teacher_strict = _metric(teacher_forced_manifest, "strict_execution_accuracy")

    compared = dict(rollout_manifest)
    compared_metrics = dict(rollout_manifest.get("metrics") or {})
    compared_metrics.update(
        {
            "teacher_forced_comparison_run_id": teacher_forced_manifest.get("run_id"),
            "teacher_forced_model_name": teacher_forced_manifest.get("model_name"),
            "teacher_forced_input_sha256": teacher_forced_manifest.get("input_sha256"),
            "teacher_forced_value_execution_accuracy": teacher_value,
            "teacher_forced_strict_execution_accuracy": teacher_strict,
            "rollout_value_delta_vs_teacher_forced": rollout_value - teacher_value,
            "rollout_strict_delta_vs_teacher_forced": rollout_strict - teacher_strict,
        }
    )
    compared["metrics"] = compared_metrics
    compared["command"] = list(rollout_manifest.get("command") or []) + [
        "# compared-with",
        str(teacher_forced_manifest.get("run_id")),
    ]
    return compared


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def compare_rollout_manifest_files(
    *,
    rollout_manifest_path: Path,
    teacher_forced_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compare two manifest files and write an augmented rollout manifest."""

    compared = compare_rollout_manifests(
        rollout_manifest=_load_manifest(rollout_manifest_path),
        teacher_forced_manifest=_load_manifest(teacher_forced_manifest_path),
    )
    _write_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-manifest", type=Path, required=True)
    parser.add_argument("--teacher-forced-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    compared = compare_rollout_manifest_files(
        rollout_manifest_path=args.rollout_manifest,
        teacher_forced_manifest_path=args.teacher_forced_manifest,
        output_path=args.output,
    )
    metrics = compared["metrics"]
    print(f"Wrote rollout comparison manifest to {args.output}")
    print(
        "Value delta vs teacher-forced: "
        f"{metrics['rollout_value_delta_vs_teacher_forced']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
