"""Compare semantic-proxy SQL against the row-matched direct-SQL control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

BENCHMARK = "prepared"
NON_ORACLE_GENERATION = "non_oracle_generation"


def _metric(manifest: dict[str, Any], name: str) -> float:
    metrics = manifest.get("metrics") or {}
    value = metrics.get(name)
    if value is None:
        raise ValueError(f"manifest {manifest.get('run_id')} is missing metric {name}")
    return float(value)


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


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dialog_id")),
        str(row.get("turn_index")),
        str(row.get("database_id")),
        str(row.get("reference_sql")),
    )


def compare_semantic_proxy_direct_sql_manifests(
    *,
    semantic_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    semantic_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if semantic_manifest.get("benchmark") != BENCHMARK or direct_manifest.get("benchmark") != BENCHMARK:
        raise ValueError("semantic proxy comparison requires prepared benchmark manifests")
    if semantic_manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError("semantic proxy manifest must use non_oracle_generation")
    if direct_manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError("direct manifest must use non_oracle_generation")
    if [_row_identity(row) for row in semantic_rows] != [_row_identity(row) for row in direct_rows]:
        raise ValueError("semantic proxy and direct SQL row identity mismatch")

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
            "semantic_proxy_value_delta_vs_direct_sql": semantic_value - direct_value,
            "semantic_proxy_strict_delta_vs_direct_sql": semantic_strict - direct_strict,
            "semantic_proxy_comparable_row_count": len(semantic_rows),
        }
    )
    compared["metrics"] = compared_metrics
    return compared


def compare_semantic_proxy_direct_sql_manifest_files(
    *,
    semantic_manifest_path: Path,
    direct_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    semantic_manifest = _load_manifest(semantic_manifest_path)
    direct_manifest = _load_manifest(direct_manifest_path)
    repo_root = repo_root.resolve()
    compared = compare_semantic_proxy_direct_sql_manifests(
        semantic_manifest=semantic_manifest,
        direct_manifest=direct_manifest,
        semantic_rows=_load_jsonl(_resolve_path(repo_root, semantic_manifest.get("output_path"))),
        direct_rows=_load_jsonl(_resolve_path(repo_root, direct_manifest.get("output_path"))),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(compared, indent=2, sort_keys=True) + "\n")
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    compare_semantic_proxy_direct_sql_manifest_files(
        semantic_manifest_path=args.semantic_manifest,
        direct_manifest_path=args.direct_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
