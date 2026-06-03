"""
Compare generated-history rollout against matching teacher-forced evaluation.

This does not run either evaluation. It verifies two result manifests are
comparable and writes an augmented rollout manifest carrying the comparison
metrics required for a generated-history diagnostic.
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
    if manifest.get("oracle_allowed"):
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


def _validate_matching_row_count(
    rollout_manifest: dict[str, Any],
    teacher_forced_manifest: dict[str, Any],
) -> int:
    rollout_count = rollout_manifest.get("row_count")
    teacher_count = teacher_forced_manifest.get("row_count")
    if rollout_count is None or teacher_count is None:
        raise ValueError("rollout and teacher-forced manifests must include row_count")
    if int(rollout_count) != int(teacher_count):
        raise ValueError("rollout and teacher-forced manifests must have matching row_count")
    return int(rollout_count)


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    missing = [
        field
        for field in ("dialog_id", "turn_index", "database_id", "reference_sql")
        if field not in row
    ]
    if missing:
        raise ValueError("result row is missing identity field(s): " + ", ".join(missing))
    return (
        str(row.get("dialog_id")),
        str(row.get("turn_index")),
        str(row.get("database_id")),
        str(row.get("reference_sql")),
    )


def _validate_manifest_row_count(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    expected = int(manifest.get("row_count") or 0)
    if expected != len(rows):
        raise ValueError(f"{label} manifest row_count does not match output rows")


def _reject_duplicate_identities(
    identities: list[tuple[str, str, str, str]],
    *,
    label: str,
) -> None:
    seen = set()
    for identity in identities:
        if identity in seen:
            raise ValueError(f"duplicate {label} row identity")
        seen.add(identity)


def _validate_rows(
    *,
    rollout_manifest: dict[str, Any],
    teacher_forced_manifest: dict[str, Any],
    rollout_rows: list[dict[str, Any]],
    teacher_forced_rows: list[dict[str, Any]],
) -> None:
    if not rollout_rows or not teacher_forced_rows:
        raise ValueError("comparison requires non-empty output rows")
    _validate_manifest_row_count(rollout_manifest, rollout_rows, label="rollout")
    _validate_manifest_row_count(
        teacher_forced_manifest,
        teacher_forced_rows,
        label="teacher-forced",
    )
    rollout_policies = {row.get("history_policy") for row in rollout_rows}
    if rollout_policies != {MODEL_GENERATED_SQL_ROLLOUT}:
        raise ValueError("rollout output rows must use model_generated_sql_rollout history")
    teacher_policies = {row.get("history_policy") for row in teacher_forced_rows}
    if teacher_policies != {GOLD_SQL_TEACHER_FORCED}:
        raise ValueError("teacher-forced output rows must use gold_sql_teacher_forced history")

    rollout_identities = [_row_identity(row) for row in rollout_rows]
    teacher_identities = [_row_identity(row) for row in teacher_forced_rows]
    _reject_duplicate_identities(rollout_identities, label="rollout")
    _reject_duplicate_identities(teacher_identities, label="teacher-forced")
    if rollout_identities != teacher_identities:
        raise ValueError("rollout and teacher-forced row identity mismatch")


def compare_rollout_manifests(
    *,
    rollout_manifest: dict[str, Any],
    teacher_forced_manifest: dict[str, Any],
    rollout_rows: list[dict[str, Any]] | None = None,
    teacher_forced_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return an augmented rollout manifest with teacher-forced comparison metrics."""

    _validate_rollout_manifest(rollout_manifest)
    _validate_teacher_forced_manifest(teacher_forced_manifest)
    if rollout_manifest.get("model_name") != teacher_forced_manifest.get("model_name"):
        raise ValueError("rollout and teacher-forced manifests must use the same model")
    if rollout_manifest.get("input_sha256") != teacher_forced_manifest.get("input_sha256"):
        raise ValueError("rollout and teacher-forced manifests must use the same input")
    comparable_row_count = _validate_matching_row_count(
        rollout_manifest,
        teacher_forced_manifest,
    )
    if rollout_rows is None or teacher_forced_rows is None:
        raise ValueError("comparison requires both rollout and teacher-forced output rows")
    _validate_rows(
        rollout_manifest=rollout_manifest,
        teacher_forced_manifest=teacher_forced_manifest,
        rollout_rows=rollout_rows,
        teacher_forced_rows=teacher_forced_rows,
    )
    comparable_row_count = len(rollout_rows)

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
            "teacher_forced_comparable_row_count": comparable_row_count,
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_path(repo_root: Path, path_value: str | None) -> Path:
    if not path_value:
        raise ValueError("manifest is missing output_path")
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def compare_rollout_manifest_files(
    *,
    rollout_manifest_path: Path,
    teacher_forced_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Compare two manifest files and write an augmented rollout manifest."""

    rollout_manifest = _load_manifest(rollout_manifest_path)
    teacher_forced_manifest = _load_manifest(teacher_forced_manifest_path)
    repo_root = repo_root.resolve()
    compared = compare_rollout_manifests(
        rollout_manifest=rollout_manifest,
        teacher_forced_manifest=teacher_forced_manifest,
        rollout_rows=_load_jsonl(_resolve_path(repo_root, rollout_manifest.get("output_path"))),
        teacher_forced_rows=_load_jsonl(
            _resolve_path(repo_root, teacher_forced_manifest.get("output_path"))
        ),
    )
    _write_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-manifest", type=Path, required=True)
    parser.add_argument("--teacher-forced-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_rollout_manifest_files(
        rollout_manifest_path=args.rollout_manifest,
        teacher_forced_manifest_path=args.teacher_forced_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
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
