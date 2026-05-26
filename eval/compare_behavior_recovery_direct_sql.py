"""Compare behavior-recovery SQL against a direct-SQL control on the same rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BEHAVIOR_RECOVERY = "behavior_recovery"
BEHAVIOR_RECOVERY_DIRECT_SQL = "behavior_recovery_direct_sql"
NON_ORACLE_GENERATION = "non_oracle_generation"
ORACLE_MARKERS = (
    "Oracle SQL planning hints",
    "SQL planning hints:",
)


def _metric(manifest: dict[str, Any], name: str) -> float:
    metrics = manifest.get("metrics") or {}
    value = metrics.get(name)
    if value is None:
        raise ValueError(f"manifest {manifest.get('run_id')} is missing metric {name}")
    return float(value)


def _validate_non_oracle(manifest: dict[str, Any], *, label: str) -> None:
    if manifest.get("oracle_allowed") or manifest.get("evaluation_mode") == "oracle_planner_diagnostic":
        raise ValueError(f"{label} manifest must be non-oracle")


def _validate_recovery_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="behavior-recovery")
    if manifest.get("benchmark") != BEHAVIOR_RECOVERY:
        raise ValueError("behavior-recovery manifest must use benchmark=behavior_recovery")
    if manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError("behavior-recovery manifest must use evaluation_mode=non_oracle_generation")
    if _metric(manifest, "recovery_evaluated_rows") <= 0:
        raise ValueError("behavior-recovery comparison requires recovery_evaluated_rows")
    _metric(manifest, "recovery_success_rate")


def _validate_direct_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="direct SQL")
    if manifest.get("benchmark") != BEHAVIOR_RECOVERY_DIRECT_SQL:
        raise ValueError(
            "direct SQL manifest must use benchmark=behavior_recovery_direct_sql"
        )
    if manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError("direct SQL manifest must use evaluation_mode=non_oracle_generation")
    if _metric(manifest, "recovery_evaluated_rows") <= 0:
        raise ValueError("direct SQL comparison requires recovery_evaluated_rows")
    _metric(manifest, "recovery_success_rate")


def _row_uses_oracle(row: dict[str, Any]) -> bool:
    if (
        row.get("uses_oracle_planning_hints")
        or row.get("semantic_context_pruned_by_oracle_labels")
        or row.get("semantic_model_oracle_derived")
    ):
        return True
    return any(
        marker in str(message.get("content", ""))
        for message in row.get("messages", [])
        for marker in ORACLE_MARKERS
    )


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    missing = [field for field in ("id", "reference_sql") if field not in row]
    if missing:
        raise ValueError("result row is missing identity field(s): " + ", ".join(missing))
    database_key = row.get("database_id") or row.get("database_path")
    if not database_key:
        raise ValueError("result row is missing identity field(s): database_id or database_path")
    return (str(row.get("id")), str(database_key), str(row.get("reference_sql")))


def _validate_manifest_row_count(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    expected = int(manifest.get("row_count") or 0)
    if expected != len(rows):
        raise ValueError(f"{label} manifest row_count does not match output rows")


def _validate_rows(
    *,
    behavior_recovery_manifest: dict[str, Any],
    direct_sql_manifest: dict[str, Any],
    behavior_recovery_rows: list[dict[str, Any]],
    direct_sql_rows: list[dict[str, Any]],
) -> None:
    if not behavior_recovery_rows or not direct_sql_rows:
        raise ValueError("comparison requires non-empty output rows")
    _validate_manifest_row_count(
        behavior_recovery_manifest, behavior_recovery_rows, label="behavior-recovery"
    )
    _validate_manifest_row_count(direct_sql_manifest, direct_sql_rows, label="direct SQL")
    if {row.get("evaluation_mode") for row in behavior_recovery_rows} != {NON_ORACLE_GENERATION}:
        raise ValueError(
            "behavior-recovery output rows must all use non_oracle_generation mode"
        )
    if {row.get("evaluation_mode") for row in direct_sql_rows} != {NON_ORACLE_GENERATION}:
        raise ValueError("direct SQL output rows must all use non_oracle_generation mode")
    if any(_row_uses_oracle(row) for row in behavior_recovery_rows + direct_sql_rows):
        raise ValueError("oracle-derived rows found in behavior-recovery comparison")
    if any(
        row.get("value_execution_score") is None or row.get("strict_execution_score") is None
        for row in behavior_recovery_rows + direct_sql_rows
    ):
        raise ValueError("comparison rows must include execution scores")

    behavior_identities = [_row_identity(row) for row in behavior_recovery_rows]
    direct_identities = [_row_identity(row) for row in direct_sql_rows]
    if behavior_identities != direct_identities:
        raise ValueError("behavior-recovery and direct SQL row identity mismatch")


def compare_behavior_recovery_direct_sql_manifests(
    *,
    behavior_recovery_manifest: dict[str, Any],
    direct_sql_manifest: dict[str, Any],
    behavior_recovery_rows: list[dict[str, Any]],
    direct_sql_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_recovery_manifest(behavior_recovery_manifest)
    _validate_direct_manifest(direct_sql_manifest)
    _validate_rows(
        behavior_recovery_manifest=behavior_recovery_manifest,
        direct_sql_manifest=direct_sql_manifest,
        behavior_recovery_rows=behavior_recovery_rows,
        direct_sql_rows=direct_sql_rows,
    )

    recovery_value = _metric(behavior_recovery_manifest, "value_execution_accuracy")
    recovery_strict = _metric(behavior_recovery_manifest, "strict_execution_accuracy")
    recovery_success = _metric(behavior_recovery_manifest, "recovery_success_rate")
    direct_value = _metric(direct_sql_manifest, "value_execution_accuracy")
    direct_strict = _metric(direct_sql_manifest, "strict_execution_accuracy")
    direct_success = _metric(direct_sql_manifest, "recovery_success_rate")

    compared = dict(behavior_recovery_manifest)
    compared_metrics = dict(behavior_recovery_manifest.get("metrics") or {})
    compared_metrics.update(
        {
            "direct_sql_comparison_run_id": direct_sql_manifest.get("run_id"),
            "direct_sql_model_name": direct_sql_manifest.get("model_name"),
            "direct_sql_input_sha256": direct_sql_manifest.get("input_sha256"),
            "direct_sql_output_sha256": direct_sql_manifest.get("output_sha256"),
            "direct_sql_value_execution_accuracy": direct_value,
            "direct_sql_strict_execution_accuracy": direct_strict,
            "direct_sql_recovery_success_rate": direct_success,
            "behavior_recovery_value_delta_vs_direct_sql": recovery_value - direct_value,
            "behavior_recovery_strict_delta_vs_direct_sql": recovery_strict - direct_strict,
            "behavior_recovery_success_delta_vs_direct_sql": recovery_success - direct_success,
            "behavior_recovery_comparable_row_count": len(behavior_recovery_rows),
        }
    )
    compared["metrics"] = compared_metrics
    compared["command"] = list(behavior_recovery_manifest.get("command") or []) + [
        "# compared-with-direct-sql",
        str(direct_sql_manifest.get("run_id")),
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


def compare_behavior_recovery_direct_sql_manifest_files(
    *,
    behavior_recovery_manifest_path: Path,
    direct_sql_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    behavior_recovery_manifest = _load_manifest(behavior_recovery_manifest_path)
    direct_sql_manifest = _load_manifest(direct_sql_manifest_path)
    repo_root = repo_root.resolve()
    compared = compare_behavior_recovery_direct_sql_manifests(
        behavior_recovery_manifest=behavior_recovery_manifest,
        direct_sql_manifest=direct_sql_manifest,
        behavior_recovery_rows=_load_jsonl(
            _resolve_path(repo_root, behavior_recovery_manifest.get("output_path"))
        ),
        direct_sql_rows=_load_jsonl(_resolve_path(repo_root, direct_sql_manifest.get("output_path"))),
    )
    _write_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-recovery-manifest", type=Path, required=True)
    parser.add_argument("--direct-sql-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_behavior_recovery_direct_sql_manifest_files(
        behavior_recovery_manifest_path=args.behavior_recovery_manifest,
        direct_sql_manifest_path=args.direct_sql_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    metrics = compared["metrics"]
    print(f"Wrote behavior-recovery comparison manifest to {args.output}")
    print(
        "Recovery success delta vs direct SQL: "
        f"{metrics['behavior_recovery_success_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
