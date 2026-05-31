"""
Compare semantic value-retrieval SQL against a direct-SQL baseline.

This does not run generation. It verifies two prepared result manifests and
their output rows are comparable, then writes an augmented semantic manifest
with the same-row deltas required for a method claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

NON_ORACLE_GENERATION = "non_oracle_generation"
SEMANTIC_PROMOTION_POLICY = {
    "minimum_comparable_row_count": 24,
    "required_split_role": "clean_local_holdout",
    "minimum_value_delta_vs_direct_sql": 0.0,
    "minimum_strict_delta_vs_direct_sql": 0.0,
    "required_value_index_source": "database_contents",
}
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


def _validate_prepared_manifest(manifest: dict[str, Any], *, label: str) -> None:
    _validate_non_oracle(manifest, label=label)
    if manifest.get("benchmark") != "prepared":
        raise ValueError(f"{label} manifest must use benchmark=prepared")
    if manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError(f"{label} manifest must use evaluation_mode=non_oracle_generation")


def _validate_semantic_manifest(
    manifest: dict[str, Any],
    *,
    value_index_manifest_sha256: str,
) -> None:
    _validate_prepared_manifest(manifest, label="semantic value-retrieval")
    metrics = manifest.get("metrics") or {}
    if metrics.get("value_index_manifest_sha256") != value_index_manifest_sha256:
        raise ValueError("semantic manifest must reference the compared value-index manifest")
    if metrics.get("value_index_index_source") != "database_contents":
        raise ValueError("semantic manifest must use a database-derived value index")


def _validate_direct_manifest(manifest: dict[str, Any]) -> None:
    _validate_prepared_manifest(manifest, label="direct SQL")


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
    semantic_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    semantic_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> None:
    if not semantic_rows or not direct_rows:
        raise ValueError("comparison requires non-empty output rows")
    _validate_manifest_row_count(semantic_manifest, semantic_rows, label="semantic")
    _validate_manifest_row_count(direct_manifest, direct_rows, label="direct SQL")
    semantic_modes = {row.get("evaluation_mode") for row in semantic_rows}
    if semantic_modes != {NON_ORACLE_GENERATION}:
        raise ValueError("semantic output rows must all use non_oracle_generation mode")
    direct_modes = {row.get("evaluation_mode") for row in direct_rows}
    if direct_modes != {NON_ORACLE_GENERATION}:
        raise ValueError("direct SQL output rows must all use non_oracle_generation mode")
    if any(_row_uses_oracle(row) for row in semantic_rows + direct_rows):
        raise ValueError("oracle-derived rows found in semantic value-retrieval comparison")
    if any(
        row.get("value_execution_score") is None or row.get("strict_execution_score") is None
        for row in semantic_rows + direct_rows
    ):
        raise ValueError("comparison output rows must include execution scores")

    semantic_identities = [_row_identity(row) for row in semantic_rows]
    direct_identities = [_row_identity(row) for row in direct_rows]
    _reject_duplicate_identities(semantic_identities, label="semantic")
    _reject_duplicate_identities(direct_identities, label="direct SQL")
    if semantic_identities != direct_identities:
        raise ValueError("semantic value-retrieval and direct SQL row identity mismatch")


def compare_semantic_value_retrieval_manifests(
    *,
    semantic_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    semantic_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    value_index_manifest_sha256: str,
) -> dict[str, Any]:
    """Return an augmented semantic manifest with direct-SQL comparison metrics."""

    _validate_semantic_manifest(
        semantic_manifest,
        value_index_manifest_sha256=value_index_manifest_sha256,
    )
    _validate_direct_manifest(direct_manifest)
    if semantic_manifest.get("model_name") != direct_manifest.get("model_name"):
        raise ValueError("semantic and direct SQL manifests must use the same model")
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
    value_delta = semantic_value - direct_value
    strict_delta = semantic_strict - direct_strict
    comparable_row_count = len(semantic_rows)

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
            "semantic_value_retrieval_value_delta_vs_direct_sql": value_delta,
            "semantic_value_retrieval_strict_delta_vs_direct_sql": strict_delta,
            "semantic_value_retrieval_comparable_row_count": comparable_row_count,
            "semantic_value_retrieval_comparer": "eval.compare_semantic_value_retrieval",
            "value_index_manifest_sha256": value_index_manifest_sha256,
        }
    )
    promotion_blockers = semantic_value_retrieval_promotion_blockers(compared_metrics)
    compared_metrics.update(
        {
            "semantic_value_retrieval_promotion_policy": SEMANTIC_PROMOTION_POLICY,
            "semantic_value_retrieval_promotion_blockers": promotion_blockers,
            "semantic_value_retrieval_promotion_ready": not promotion_blockers,
        }
    )
    compared["metrics"] = compared_metrics
    compared["command"] = list(semantic_manifest.get("command") or []) + [
        "# compared-with-direct-sql",
        str(direct_manifest.get("run_id")),
    ]
    return compared


def semantic_value_retrieval_promotion_blockers(metrics: dict[str, Any]) -> list[str]:
    """Return blockers before semantic value retrieval can support a method claim."""

    policy = SEMANTIC_PROMOTION_POLICY
    blockers = []
    comparable_row_count = int(
        metrics.get("semantic_value_retrieval_comparable_row_count") or 0
    )
    if comparable_row_count < policy["minimum_comparable_row_count"]:
        blockers.append(
            f"comparable row count below minimum {policy['minimum_comparable_row_count']}"
        )
    split_roles = {
        str(role): int(count)
        for role, count in (metrics.get("split_roles") or {}).items()
    }
    required_split_role = str(policy["required_split_role"])
    if required_split_role not in split_roles:
        blockers.append(f"missing required split role {required_split_role}")
    value_delta = float(
        metrics.get("semantic_value_retrieval_value_delta_vs_direct_sql") or 0.0
    )
    if value_delta <= float(policy["minimum_value_delta_vs_direct_sql"]):
        blockers.append("value delta vs direct SQL must be positive")
    strict_delta = float(
        metrics.get("semantic_value_retrieval_strict_delta_vs_direct_sql") or 0.0
    )
    if strict_delta < float(policy["minimum_strict_delta_vs_direct_sql"]):
        blockers.append("strict delta vs direct SQL must not regress")
    if metrics.get("value_index_index_source") != policy["required_value_index_source"]:
        blockers.append("value index must be database-derived")
    return blockers


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


def compare_semantic_value_retrieval_manifest_files(
    *,
    semantic_manifest_path: Path,
    direct_manifest_path: Path,
    value_index_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Compare manifest files and write an augmented semantic manifest."""

    semantic_manifest = _load_manifest(semantic_manifest_path)
    direct_manifest = _load_manifest(direct_manifest_path)
    value_index_manifest_sha256 = sha256_file(value_index_manifest_path)
    if value_index_manifest_sha256 is None:
        raise ValueError("value-index manifest does not exist")
    repo_root = repo_root.resolve()
    compared = compare_semantic_value_retrieval_manifests(
        semantic_manifest=semantic_manifest,
        direct_manifest=direct_manifest,
        semantic_rows=_load_jsonl(_resolve_path(repo_root, semantic_manifest.get("output_path"))),
        direct_rows=_load_jsonl(_resolve_path(repo_root, direct_manifest.get("output_path"))),
        value_index_manifest_sha256=value_index_manifest_sha256,
    )
    _write_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--value-index-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_semantic_value_retrieval_manifest_files(
        semantic_manifest_path=args.semantic_manifest,
        direct_manifest_path=args.direct_manifest,
        value_index_manifest_path=args.value_index_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    metrics = compared["metrics"]
    print(f"Wrote semantic value-retrieval comparison manifest to {args.output}")
    print(
        "Value delta vs direct SQL: "
        f"{metrics['semantic_value_retrieval_value_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
