"""Compare semantic-layer SQL against the row-matched direct-SQL control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SEMANTIC_BENCHMARK = "synthetic_semantic_layer"
DIRECT_BENCHMARK = "synthetic_semantic_layer_direct_sql"
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
    if manifest.get("oracle_allowed") or manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError(f"{label} manifest must be non-oracle non_oracle_generation")


def _validate_semantic_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="semantic-layer")
    if manifest.get("benchmark") != SEMANTIC_BENCHMARK:
        raise ValueError("semantic-layer manifest must use benchmark=synthetic_semantic_layer")


def _validate_direct_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="direct SQL")
    if manifest.get("benchmark") != DIRECT_BENCHMARK:
        raise ValueError(
            "direct SQL manifest must use benchmark=synthetic_semantic_layer_direct_sql"
        )


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
    missing = [field for field in ("id", "fixture_id", "reference_sql") if field not in row]
    if missing:
        raise ValueError("result row is missing identity field(s): " + ", ".join(missing))
    return (str(row["id"]), str(row["fixture_id"]), str(row["reference_sql"]))


def _validate_manifest_row_count(manifest: dict[str, Any], rows: list[dict[str, Any]], *, label: str) -> None:
    expected = int(manifest.get("row_count") or 0)
    if expected != len(rows):
        raise ValueError(f"{label} manifest row_count does not match output rows")


def _validate_rows(
    *,
    semantic_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    semantic_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> None:
    if not semantic_rows or not direct_rows:
        raise ValueError("comparison requires non-empty output rows")
    _validate_manifest_row_count(semantic_manifest, semantic_rows, label="semantic-layer")
    _validate_manifest_row_count(direct_manifest, direct_rows, label="direct SQL")
    semantic_modes = {row.get("evaluation_mode") for row in semantic_rows}
    direct_modes = {row.get("evaluation_mode") for row in direct_rows}
    if semantic_modes != {NON_ORACLE_GENERATION} or direct_modes != {NON_ORACLE_GENERATION}:
        raise ValueError("semantic-layer comparison requires non_oracle_generation rows on both sides")
    if any(_row_uses_oracle(row) for row in semantic_rows + direct_rows):
        raise ValueError("oracle-derived rows found in semantic-layer comparison")
    if any(
        row.get("value_execution_score") is None or row.get("strict_execution_score") is None
        for row in semantic_rows + direct_rows
    ):
        raise ValueError("semantic-layer comparison requires execution scores on every row")
    if [_row_identity(row) for row in semantic_rows] != [_row_identity(row) for row in direct_rows]:
        raise ValueError("semantic-layer and direct SQL row identity mismatch")


def compare_semantic_layer_direct_sql_manifests(
    *,
    semantic_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    semantic_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    _validate_semantic_manifest(semantic_manifest)
    _validate_direct_manifest(direct_manifest)
    _validate_rows(
        semantic_manifest=semantic_manifest,
        direct_manifest=direct_manifest,
        semantic_rows=semantic_rows,
        direct_rows=direct_rows,
    )

    semantic_value = _metric(semantic_manifest, "value_execution_accuracy")
    semantic_strict = _metric(semantic_manifest, "strict_execution_accuracy")
    direct_value = _metric(direct_manifest, "value_execution_accuracy")
    direct_strict = _metric(direct_manifest, "strict_execution_accuracy")

    compared = dict(semantic_manifest)
    compared_metrics = dict(semantic_manifest.get("metrics") or {})
    compared_metrics.update(
        {
            "direct_sql_comparison_run_id": direct_manifest.get("run_id"),
            "direct_sql_model_name": direct_manifest.get("model_name"),
            "direct_sql_input_sha256": direct_manifest.get("input_sha256"),
            "direct_sql_output_sha256": direct_manifest.get("output_sha256"),
            "direct_sql_value_execution_accuracy": direct_value,
            "direct_sql_strict_execution_accuracy": direct_strict,
            "semantic_layer_value_delta_vs_direct_sql": semantic_value - direct_value,
            "semantic_layer_strict_delta_vs_direct_sql": semantic_strict - direct_strict,
            "semantic_layer_comparable_row_count": len(semantic_rows),
        }
    )
    compared["metrics"] = compared_metrics
    compared["command"] = list(semantic_manifest.get("command") or []) + [
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


def compare_semantic_layer_direct_sql_manifest_files(
    *,
    semantic_manifest_path: Path,
    direct_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    semantic_manifest = _load_manifest(semantic_manifest_path)
    direct_manifest = _load_manifest(direct_manifest_path)
    repo_root = repo_root.resolve()
    compared = compare_semantic_layer_direct_sql_manifests(
        semantic_manifest=semantic_manifest,
        direct_manifest=direct_manifest,
        semantic_rows=_load_jsonl(_resolve_path(repo_root, semantic_manifest.get("output_path"))),
        direct_rows=_load_jsonl(_resolve_path(repo_root, direct_manifest.get("output_path"))),
    )
    _write_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_semantic_layer_direct_sql_manifest_files(
        semantic_manifest_path=args.semantic_manifest,
        direct_manifest_path=args.direct_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    print(f"Wrote semantic-layer comparison manifest to {args.output}")
    print(
        "Value delta vs direct SQL: "
        f"{compared['metrics']['semantic_layer_value_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
