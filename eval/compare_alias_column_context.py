"""Compare alias/column-context SQL against a direct-SQL baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file, write_result_manifest

NON_ORACLE_GENERATION = "non_oracle_generation"


def _metric(manifest: dict[str, Any], name: str) -> float:
    value = (manifest.get("metrics") or {}).get(name)
    if value is None:
        raise ValueError(f"manifest {manifest.get('run_id')} is missing metric {name}")
    return float(value)


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


def _validate_manifest(manifest: dict[str, Any], *, label: str) -> None:
    if manifest.get("oracle_allowed"):
        raise ValueError(f"{label} manifest must be non-oracle")
    if manifest.get("benchmark") != "prepared":
        raise ValueError(f"{label} manifest must use benchmark=prepared")
    if manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError(f"{label} manifest must use evaluation_mode=non_oracle_generation")


def _validate_rows(
    *,
    alias_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    alias_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> None:
    if not alias_rows or not direct_rows:
        raise ValueError("comparison requires non-empty output rows")
    if int(alias_manifest.get("row_count") or 0) != len(alias_rows):
        raise ValueError("alias/column manifest row_count does not match output rows")
    if int(direct_manifest.get("row_count") or 0) != len(direct_rows):
        raise ValueError("direct SQL manifest row_count does not match output rows")
    if [ _row_identity(row) for row in alias_rows ] != [ _row_identity(row) for row in direct_rows ]:
        raise ValueError("alias/column and direct SQL row identity mismatch")


def compare_alias_column_context_manifests(
    *,
    alias_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    alias_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    alias_context_manifest_sha256: str,
) -> dict[str, Any]:
    """Return an augmented alias/column manifest with direct-SQL deltas."""

    _validate_manifest(alias_manifest, label="alias/column context")
    _validate_manifest(direct_manifest, label="direct SQL")
    if alias_manifest.get("model_name") != direct_manifest.get("model_name"):
        raise ValueError("alias/column and direct SQL manifests must use the same model")
    _validate_rows(
        alias_manifest=alias_manifest,
        direct_manifest=direct_manifest,
        alias_rows=alias_rows,
        direct_rows=direct_rows,
    )

    alias_value = _metric(alias_manifest, "value_execution_accuracy")
    alias_strict = _metric(alias_manifest, "strict_execution_accuracy")
    direct_value = _metric(direct_manifest, "value_execution_accuracy")
    direct_strict = _metric(direct_manifest, "strict_execution_accuracy")

    compared = dict(alias_manifest)
    metrics = dict(alias_manifest.get("metrics") or {})
    metrics.update(
        {
            "direct_sql_comparison_run_id": direct_manifest.get("run_id"),
            "direct_sql_model_name": direct_manifest.get("model_name"),
            "direct_sql_input_sha256": direct_manifest.get("input_sha256"),
            "direct_sql_output_sha256": direct_manifest.get("output_sha256"),
            "direct_sql_value_execution_accuracy": direct_value,
            "direct_sql_strict_execution_accuracy": direct_strict,
            "alias_column_context_value_delta_vs_direct_sql": alias_value - direct_value,
            "alias_column_context_strict_delta_vs_direct_sql": alias_strict - direct_strict,
            "alias_column_context_comparable_row_count": len(alias_rows),
            "alias_column_context_comparer": "eval.compare_alias_column_context",
            "alias_column_context_manifest_sha256": alias_context_manifest_sha256,
        }
    )
    compared["metrics"] = metrics
    compared["command"] = list(alias_manifest.get("command") or []) + [
        "# compared-with-direct-sql",
        str(direct_manifest.get("run_id")),
    ]
    return compared


def compare_alias_column_context_manifest_files(
    *,
    alias_manifest_path: Path,
    direct_manifest_path: Path,
    alias_context_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Compare manifest files and write an augmented alias/column manifest."""

    alias_manifest = _load_json(alias_manifest_path)
    direct_manifest = _load_json(direct_manifest_path)
    alias_context_manifest_sha256 = sha256_file(alias_context_manifest_path)
    if alias_context_manifest_sha256 is None:
        raise ValueError("alias/column context manifest does not exist")
    repo_root = repo_root.resolve()
    compared = compare_alias_column_context_manifests(
        alias_manifest=alias_manifest,
        direct_manifest=direct_manifest,
        alias_rows=_load_jsonl(_resolve_path(repo_root, alias_manifest.get("output_path"))),
        direct_rows=_load_jsonl(_resolve_path(repo_root, direct_manifest.get("output_path"))),
        alias_context_manifest_sha256=alias_context_manifest_sha256,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_result_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--alias-context-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_alias_column_context_manifest_files(
        alias_manifest_path=args.alias_manifest,
        direct_manifest_path=args.direct_manifest,
        alias_context_manifest_path=args.alias_context_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    print(f"Wrote alias/column context comparison manifest to {args.output}")
    print(
        "Value delta vs direct SQL: "
        f"{compared['metrics']['alias_column_context_value_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
