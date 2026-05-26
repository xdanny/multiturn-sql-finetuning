"""
Compare predicted-planner SQL against the matching direct-SQL baseline.

This does not run either evaluation. It verifies two prepared result manifests
and their output rows are comparable, then writes an augmented predicted-planner
manifest carrying the comparison metrics required by the claim ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NON_ORACLE_GENERATION = "non_oracle_generation"
PREDICTED_PLANNER = "predicted_planner"
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


def _validate_predicted_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="predicted-planner")
    if manifest.get("benchmark") != "prepared":
        raise ValueError("predicted-planner manifest must use benchmark=prepared")
    if manifest.get("evaluation_mode") != PREDICTED_PLANNER:
        raise ValueError("predicted-planner manifest must use evaluation_mode=predicted_planner")


def _validate_direct_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="direct SQL")
    if manifest.get("benchmark") != "prepared":
        raise ValueError("direct SQL manifest must use benchmark=prepared")
    if manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError("direct SQL manifest must use evaluation_mode=non_oracle_generation")


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


def _row_uses_oracle_plan(row: dict[str, Any]) -> bool:
    if row.get("uses_oracle_planning_hints") or row.get(
        "semantic_context_pruned_by_oracle_labels"
    ):
        return True
    return any(
        marker in str(message.get("content", ""))
        for message in row.get("messages", [])
        for marker in ORACLE_MARKERS
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
    predicted_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    predicted_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> None:
    if not predicted_rows or not direct_rows:
        raise ValueError("comparison requires non-empty output rows")
    _validate_manifest_row_count(predicted_manifest, predicted_rows, label="predicted-planner")
    _validate_manifest_row_count(direct_manifest, direct_rows, label="direct SQL")
    predicted_modes = {row.get("evaluation_mode") for row in predicted_rows}
    if predicted_modes != {PREDICTED_PLANNER}:
        raise ValueError("predicted_planner output rows must all use predicted_planner mode")
    direct_modes = {row.get("evaluation_mode") for row in direct_rows}
    if direct_modes != {NON_ORACLE_GENERATION}:
        raise ValueError("direct SQL output rows must all use non_oracle_generation mode")
    if any(_row_uses_oracle_plan(row) for row in predicted_rows + direct_rows):
        raise ValueError("oracle planning hints found in comparison output rows")

    predicted_identities = [_row_identity(row) for row in predicted_rows]
    direct_identities = [_row_identity(row) for row in direct_rows]
    _reject_duplicate_identities(predicted_identities, label="predicted-planner")
    _reject_duplicate_identities(direct_identities, label="direct SQL")
    if predicted_identities != direct_identities:
        raise ValueError("predicted-planner and direct SQL row identity mismatch")


def compare_predicted_planner_manifests(
    *,
    predicted_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    predicted_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return an augmented predicted-planner manifest with direct-SQL deltas."""

    _validate_predicted_manifest(predicted_manifest)
    _validate_direct_manifest(direct_manifest)
    if predicted_manifest.get("model_name") != direct_manifest.get("model_name"):
        raise ValueError("predicted-planner and direct SQL manifests must use the same model")
    _validate_rows(
        predicted_manifest=predicted_manifest,
        direct_manifest=direct_manifest,
        predicted_rows=predicted_rows,
        direct_rows=direct_rows,
    )

    predicted_value = _metric(predicted_manifest, "value_execution_accuracy")
    predicted_strict = _metric(predicted_manifest, "strict_execution_accuracy")
    direct_value = _metric(direct_manifest, "value_execution_accuracy")
    direct_strict = _metric(direct_manifest, "strict_execution_accuracy")

    compared = dict(predicted_manifest)
    compared_metrics = dict(predicted_manifest.get("metrics") or {})
    compared_metrics.update(
        {
            "direct_sql_comparison_run_id": direct_manifest.get("run_id"),
            "direct_sql_model_name": direct_manifest.get("model_name"),
            "direct_sql_input_sha256": direct_manifest.get("input_sha256"),
            "direct_sql_output_sha256": direct_manifest.get("output_sha256"),
            "direct_sql_value_execution_accuracy": direct_value,
            "direct_sql_strict_execution_accuracy": direct_strict,
            "predicted_planner_value_delta_vs_direct_sql": predicted_value - direct_value,
            "predicted_planner_strict_delta_vs_direct_sql": predicted_strict - direct_strict,
            "direct_sql_comparable_row_count": len(direct_rows),
        }
    )
    compared["metrics"] = compared_metrics
    compared["command"] = list(predicted_manifest.get("command") or []) + [
        "# compared-with-direct-sql",
        str(direct_manifest.get("run_id")),
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


def compare_predicted_planner_manifest_files(
    *,
    predicted_manifest_path: Path,
    direct_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Compare manifest files and write an augmented predicted-planner manifest."""

    predicted_manifest = _load_manifest(predicted_manifest_path)
    direct_manifest = _load_manifest(direct_manifest_path)
    repo_root = repo_root.resolve()
    compared = compare_predicted_planner_manifests(
        predicted_manifest=predicted_manifest,
        direct_manifest=direct_manifest,
        predicted_rows=_load_jsonl(_resolve_path(repo_root, predicted_manifest.get("output_path"))),
        direct_rows=_load_jsonl(_resolve_path(repo_root, direct_manifest.get("output_path"))),
    )
    _write_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_predicted_planner_manifest_files(
        predicted_manifest_path=args.predicted_manifest,
        direct_manifest_path=args.direct_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    metrics = compared["metrics"]
    print(f"Wrote predicted-planner comparison manifest to {args.output}")
    print(
        "Value delta vs direct SQL: "
        f"{metrics['predicted_planner_value_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
