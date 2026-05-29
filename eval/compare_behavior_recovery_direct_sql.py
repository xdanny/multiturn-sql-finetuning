"""Compare behavior-recovery rollout against a direct-SQL rollout control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_rollout_history import MODEL_GENERATED_SQL_ROLLOUT


def _metric(manifest: dict[str, Any], name: str) -> float:
    metrics = manifest.get("metrics") or {}
    value = metrics.get(name)
    if value is None:
        raise ValueError(f"manifest {manifest.get('run_id')} is missing metric {name}")
    return float(value)


def _validate_rollout_manifest(manifest: dict[str, Any], *, label: str) -> None:
    if manifest.get("oracle_allowed"):
        raise ValueError(f"{label} manifest must be non-oracle")
    if manifest.get("benchmark") != "prepared_rollout":
        raise ValueError(f"{label} manifest must use benchmark=prepared_rollout")
    if (manifest.get("metrics") or {}).get("history_policy") != MODEL_GENERATED_SQL_ROLLOUT:
        raise ValueError(
            f"{label} manifest must use history_policy={MODEL_GENERATED_SQL_ROLLOUT}"
        )
    if int(manifest.get("row_count") or 0) <= 0:
        raise ValueError(f"{label} manifest must include output rows")
    _metric(manifest, "value_execution_accuracy")
    _metric(manifest, "strict_execution_accuracy")


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    missing = [
        field
        for field in ("dialog_id", "turn_index", "database_id", "reference_sql")
        if field not in row
    ]
    if missing:
        raise ValueError("result row is missing identity field(s): " + ", ".join(missing))
    return (
        str(row["dialog_id"]),
        str(row["turn_index"]),
        str(row["database_id"]),
        str(row["reference_sql"]),
    )


def _validate_rows(
    *,
    recovery_manifest: dict[str, Any],
    direct_sql_manifest: dict[str, Any],
    recovery_rows: list[dict[str, Any]],
    direct_sql_rows: list[dict[str, Any]],
) -> None:
    if int(recovery_manifest.get("row_count") or 0) != len(recovery_rows):
        raise ValueError("recovery manifest row_count does not match output rows")
    if int(direct_sql_manifest.get("row_count") or 0) != len(direct_sql_rows):
        raise ValueError("direct-SQL manifest row_count does not match output rows")
    if not recovery_rows or not direct_sql_rows:
        raise ValueError("comparison requires non-empty output rows")
    recovery_policies = {row.get("history_policy") for row in recovery_rows}
    direct_policies = {row.get("history_policy") for row in direct_sql_rows}
    if recovery_policies != {MODEL_GENERATED_SQL_ROLLOUT}:
        raise ValueError("recovery rows must use model-generated rollout history")
    if direct_policies != {MODEL_GENERATED_SQL_ROLLOUT}:
        raise ValueError("direct-SQL rows must use model-generated rollout history")
    recovery_identities = [_row_identity(row) for row in recovery_rows]
    direct_identities = [_row_identity(row) for row in direct_sql_rows]
    if recovery_identities != direct_identities:
        raise ValueError("recovery and direct-SQL row identity mismatch")


def compare_behavior_recovery_direct_sql_manifests(
    *,
    recovery_manifest: dict[str, Any],
    direct_sql_manifest: dict[str, Any],
    recovery_rows: list[dict[str, Any]],
    direct_sql_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a recovery manifest augmented with direct-control metrics."""

    _validate_rollout_manifest(recovery_manifest, label="recovery")
    _validate_rollout_manifest(direct_sql_manifest, label="direct-SQL")
    if recovery_manifest.get("input_sha256") != direct_sql_manifest.get("input_sha256"):
        raise ValueError("recovery and direct-SQL manifests must use the same input")
    _validate_rows(
        recovery_manifest=recovery_manifest,
        direct_sql_manifest=direct_sql_manifest,
        recovery_rows=recovery_rows,
        direct_sql_rows=direct_sql_rows,
    )

    recovery_value = _metric(recovery_manifest, "value_execution_accuracy")
    recovery_strict = _metric(recovery_manifest, "strict_execution_accuracy")
    direct_value = _metric(direct_sql_manifest, "value_execution_accuracy")
    direct_strict = _metric(direct_sql_manifest, "strict_execution_accuracy")
    compared = dict(recovery_manifest)
    metrics = dict(recovery_manifest.get("metrics") or {})
    metrics.update(
        {
            "direct_sql_comparison_run_id": direct_sql_manifest.get("run_id"),
            "direct_sql_model_name": direct_sql_manifest.get("model_name"),
            "direct_sql_input_sha256": direct_sql_manifest.get("input_sha256"),
            "direct_sql_output_sha256": direct_sql_manifest.get("output_sha256"),
            "direct_sql_value_execution_accuracy": direct_value,
            "direct_sql_strict_execution_accuracy": direct_strict,
            "direct_sql_comparable_row_count": len(recovery_rows),
            "behavior_recovery_value_delta_vs_direct_sql": recovery_value - direct_value,
            "behavior_recovery_strict_delta_vs_direct_sql": recovery_strict - direct_strict,
            "rollout_value_accuracy_delta_vs_direct_sql": recovery_value - direct_value,
        }
    )
    compared["metrics"] = metrics
    compared["command"] = list(recovery_manifest.get("command") or []) + [
        "# compared-with-direct-sql",
        str(direct_sql_manifest.get("run_id")),
    ]
    return compared


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_path(repo_root: Path, path_value: str | None) -> Path:
    if not path_value:
        raise ValueError("manifest is missing output_path")
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def compare_behavior_recovery_direct_sql_manifest_files(
    *,
    recovery_manifest_path: Path,
    direct_sql_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Compare manifest files and write an augmented recovery manifest."""

    repo_root = repo_root.resolve()
    recovery_manifest = _load_json(recovery_manifest_path)
    direct_sql_manifest = _load_json(direct_sql_manifest_path)
    compared = compare_behavior_recovery_direct_sql_manifests(
        recovery_manifest=recovery_manifest,
        direct_sql_manifest=direct_sql_manifest,
        recovery_rows=_load_jsonl(_resolve_path(repo_root, recovery_manifest.get("output_path"))),
        direct_sql_rows=_load_jsonl(_resolve_path(repo_root, direct_sql_manifest.get("output_path"))),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(compared, indent=2, sort_keys=True) + "\n")
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    parser.add_argument("--direct-sql-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_behavior_recovery_direct_sql_manifest_files(
        recovery_manifest_path=args.recovery_manifest,
        direct_sql_manifest_path=args.direct_sql_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    metrics = compared["metrics"]
    print(f"Wrote behavior-recovery direct-SQL comparison to {args.output}")
    print(
        "Value delta vs direct SQL: "
        f"{metrics['behavior_recovery_value_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
