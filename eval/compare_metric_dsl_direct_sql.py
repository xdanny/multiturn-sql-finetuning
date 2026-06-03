"""
Compare metric-DSL compiled SQL against a direct-SQL baseline.

This comparison is for method evaluation, not prompt-hint evaluation. The
metric-DSL model and direct-SQL model may differ, but they must be evaluated on
the same metric-heavy task rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METRIC_DSL = "metric_dsl"
METRIC_DSL_DIRECT_SQL = "metric_dsl_direct_sql"
NON_ORACLE_GENERATION = "non_oracle_generation"


def _metric(manifest: dict[str, Any], name: str) -> float:
    metrics = manifest.get("metrics") or {}
    value = metrics.get(name)
    if value is None:
        raise ValueError(f"manifest {manifest.get('run_id')} is missing metric {name}")
    return float(value)


def _validate_non_oracle(manifest: dict[str, Any], *, label: str) -> None:
    if manifest.get("oracle_allowed"):
        raise ValueError(f"{label} manifest must be non-oracle")


def _validate_metric_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="metric-DSL")
    if manifest.get("benchmark") != METRIC_DSL:
        raise ValueError("metric-DSL manifest must use benchmark=metric_dsl")
    if manifest.get("evaluation_mode") != METRIC_DSL:
        raise ValueError("metric-DSL manifest must use evaluation_mode=metric_dsl")
    row_count = int(manifest.get("row_count") or 0)
    if row_count <= 0:
        raise ValueError("metric-DSL comparison requires output rows")
    _metric(manifest, "value_execution_accuracy")
    _metric(manifest, "strict_execution_accuracy")


def _validate_direct_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="direct SQL")
    if manifest.get("benchmark") != METRIC_DSL_DIRECT_SQL:
        raise ValueError("direct SQL manifest must use benchmark=metric_dsl_direct_sql")
    if manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError("direct SQL manifest must use evaluation_mode=non_oracle_generation")


def _row_uses_oracle(row: dict[str, Any]) -> bool:
    return bool(row.get("semantic_model_oracle_derived"))


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
    metric_dsl_manifest: dict[str, Any],
    direct_sql_manifest: dict[str, Any],
    metric_dsl_rows: list[dict[str, Any]],
    direct_sql_rows: list[dict[str, Any]],
) -> None:
    if not metric_dsl_rows or not direct_sql_rows:
        raise ValueError("comparison requires non-empty output rows")
    _validate_manifest_row_count(metric_dsl_manifest, metric_dsl_rows, label="metric-DSL")
    _validate_manifest_row_count(direct_sql_manifest, direct_sql_rows, label="direct SQL")
    metric_modes = {row.get("evaluation_mode") for row in metric_dsl_rows}
    if metric_modes != {METRIC_DSL}:
        raise ValueError("metric-DSL output rows must all use metric_dsl mode")
    direct_modes = {row.get("evaluation_mode") for row in direct_sql_rows}
    if direct_modes != {NON_ORACLE_GENERATION}:
        raise ValueError("direct SQL output rows must all use non_oracle_generation mode")
    if any(_row_uses_oracle(row) for row in metric_dsl_rows + direct_sql_rows):
        raise ValueError("oracle-derived rows found in metric-DSL comparison")
    if not all(row.get("database_path") for row in metric_dsl_rows):
        raise ValueError("metric-DSL comparison requires database-backed metric rows")
    if any(
        row.get("value_execution_score") is None or row.get("strict_execution_score") is None
        for row in direct_sql_rows
    ):
        raise ValueError("direct SQL output rows must include execution scores")

    metric_identities = [_row_identity(row) for row in metric_dsl_rows]
    direct_identities = [_row_identity(row) for row in direct_sql_rows]
    if metric_identities != direct_identities:
        raise ValueError("metric-DSL and direct SQL row identity mismatch")


def compare_metric_dsl_direct_sql_manifests(
    *,
    metric_dsl_manifest: dict[str, Any],
    direct_sql_manifest: dict[str, Any],
    metric_dsl_rows: list[dict[str, Any]],
    direct_sql_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return an augmented metric-DSL manifest with direct-SQL comparison metrics."""

    _validate_metric_manifest(metric_dsl_manifest)
    _validate_direct_manifest(direct_sql_manifest)
    _validate_rows(
        metric_dsl_manifest=metric_dsl_manifest,
        direct_sql_manifest=direct_sql_manifest,
        metric_dsl_rows=metric_dsl_rows,
        direct_sql_rows=direct_sql_rows,
    )

    metric_value = _metric(metric_dsl_manifest, "value_execution_accuracy")
    metric_strict = _metric(metric_dsl_manifest, "strict_execution_accuracy")
    direct_value = _metric(direct_sql_manifest, "value_execution_accuracy")
    direct_strict = _metric(direct_sql_manifest, "strict_execution_accuracy")

    compared = dict(metric_dsl_manifest)
    compared_metrics = dict(metric_dsl_manifest.get("metrics") or {})
    compared_metrics.update(
        {
            "direct_sql_comparison_run_id": direct_sql_manifest.get("run_id"),
            "direct_sql_model_name": direct_sql_manifest.get("model_name"),
            "direct_sql_input_sha256": direct_sql_manifest.get("input_sha256"),
            "direct_sql_output_sha256": direct_sql_manifest.get("output_sha256"),
            "direct_sql_value_execution_accuracy": direct_value,
            "direct_sql_strict_execution_accuracy": direct_strict,
            "metric_dsl_value_delta_vs_direct_sql": metric_value - direct_value,
            "metric_dsl_strict_delta_vs_direct_sql": metric_strict - direct_strict,
            "metric_dsl_measure_preservation": _metric(
                metric_dsl_manifest, "measure_preservation"
            ),
            "metric_dsl_comparable_row_count": len(metric_dsl_rows),
        }
    )
    compared["metrics"] = compared_metrics
    compared["command"] = list(metric_dsl_manifest.get("command") or []) + [
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


def compare_metric_dsl_direct_sql_manifest_files(
    *,
    metric_dsl_manifest_path: Path,
    direct_sql_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Compare manifest files and write an augmented metric-DSL manifest."""

    metric_dsl_manifest = _load_manifest(metric_dsl_manifest_path)
    direct_sql_manifest = _load_manifest(direct_sql_manifest_path)
    repo_root = repo_root.resolve()
    compared = compare_metric_dsl_direct_sql_manifests(
        metric_dsl_manifest=metric_dsl_manifest,
        direct_sql_manifest=direct_sql_manifest,
        metric_dsl_rows=_load_jsonl(_resolve_path(repo_root, metric_dsl_manifest.get("output_path"))),
        direct_sql_rows=_load_jsonl(_resolve_path(repo_root, direct_sql_manifest.get("output_path"))),
    )
    _write_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-dsl-manifest", type=Path, required=True)
    parser.add_argument("--direct-sql-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_metric_dsl_direct_sql_manifest_files(
        metric_dsl_manifest_path=args.metric_dsl_manifest,
        direct_sql_manifest_path=args.direct_sql_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    metrics = compared["metrics"]
    print(f"Wrote metric-DSL comparison manifest to {args.output}")
    print(
        "Value delta vs direct SQL: "
        f"{metrics['metric_dsl_value_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
