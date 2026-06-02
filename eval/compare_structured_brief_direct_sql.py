"""
Compare structured-brief SQL against a direct-SQL baseline.

This compares two already-evaluated prepared-result manifests on the same row
identities, then writes an augmented structured-brief manifest carrying the
direct-SQL delta metrics required for a Checkpoint 5 method claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NON_ORACLE_GENERATION = "non_oracle_generation"
STRUCTURED_BRIEF_PROMOTION_POLICY = {
    "minimum_comparable_row_count": 24,
    "required_split_role": "clean_local_holdout",
    "minimum_value_delta_vs_direct_sql": 0.0,
    "minimum_strict_delta_vs_direct_sql": 0.0,
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
    structured_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    structured_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> None:
    if not structured_rows or not direct_rows:
        raise ValueError("comparison requires non-empty output rows")
    _validate_manifest_row_count(structured_manifest, structured_rows, label="structured brief")
    _validate_manifest_row_count(direct_manifest, direct_rows, label="direct SQL")
    structured_modes = {row.get("evaluation_mode") for row in structured_rows}
    if structured_modes != {NON_ORACLE_GENERATION}:
        raise ValueError("structured-brief output rows must all use non_oracle_generation mode")
    direct_modes = {row.get("evaluation_mode") for row in direct_rows}
    if direct_modes != {NON_ORACLE_GENERATION}:
        raise ValueError("direct SQL output rows must all use non_oracle_generation mode")
    if any(_row_uses_oracle(row) for row in structured_rows + direct_rows):
        raise ValueError("oracle-derived rows found in structured-brief comparison")
    if any(
        row.get("value_execution_score") is None or row.get("strict_execution_score") is None
        for row in structured_rows + direct_rows
    ):
        raise ValueError("comparison output rows must include execution scores")

    structured_identities = [_row_identity(row) for row in structured_rows]
    direct_identities = [_row_identity(row) for row in direct_rows]
    _reject_duplicate_identities(structured_identities, label="structured-brief")
    _reject_duplicate_identities(direct_identities, label="direct SQL")
    if structured_identities != direct_identities:
        raise ValueError("structured-brief and direct SQL row identity mismatch")


def structured_brief_promotion_blockers(metrics: dict[str, Any]) -> list[str]:
    """Return blockers before structured briefs can support a method claim."""

    policy = STRUCTURED_BRIEF_PROMOTION_POLICY
    blockers = []
    comparable_row_count = int(metrics.get("structured_brief_comparable_row_count") or 0)
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
    value_delta = float(metrics.get("structured_brief_value_delta_vs_direct_sql") or 0.0)
    if value_delta <= float(policy["minimum_value_delta_vs_direct_sql"]):
        blockers.append("value delta vs direct SQL must be positive")
    strict_delta = float(metrics.get("structured_brief_strict_delta_vs_direct_sql") or 0.0)
    if strict_delta < float(policy["minimum_strict_delta_vs_direct_sql"]):
        blockers.append("strict delta vs direct SQL must not regress")
    return blockers


def compare_structured_brief_manifests(
    *,
    structured_manifest: dict[str, Any],
    direct_manifest: dict[str, Any],
    structured_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return an augmented structured-brief manifest with direct-SQL deltas."""

    _validate_prepared_manifest(structured_manifest, label="structured brief")
    _validate_prepared_manifest(direct_manifest, label="direct SQL")
    if structured_manifest.get("model_name") != direct_manifest.get("model_name"):
        raise ValueError("structured brief and direct SQL manifests must use the same model")
    _validate_rows(
        structured_manifest=structured_manifest,
        direct_manifest=direct_manifest,
        structured_rows=structured_rows,
        direct_rows=direct_rows,
    )

    structured_value = _metric(structured_manifest, "value_execution_accuracy")
    structured_strict = _metric(structured_manifest, "strict_execution_accuracy")
    direct_value = _metric(direct_manifest, "value_execution_accuracy")
    direct_strict = _metric(direct_manifest, "strict_execution_accuracy")
    value_delta = structured_value - direct_value
    strict_delta = structured_strict - direct_strict
    comparable_row_count = len(structured_rows)

    compared = dict(structured_manifest)
    compared_metrics = dict(structured_manifest.get("metrics") or {})
    compared_metrics.update(
        {
            "direct_sql_comparison_run_id": direct_manifest.get("run_id"),
            "direct_sql_model_name": direct_manifest.get("model_name"),
            "direct_sql_input_sha256": direct_manifest.get("input_sha256"),
            "direct_sql_output_sha256": direct_manifest.get("output_sha256"),
            "direct_sql_value_execution_accuracy": direct_value,
            "direct_sql_strict_execution_accuracy": direct_strict,
            "structured_brief_value_delta_vs_direct_sql": value_delta,
            "structured_brief_strict_delta_vs_direct_sql": strict_delta,
            "structured_brief_comparable_row_count": comparable_row_count,
            "structured_brief_comparer": "eval.compare_structured_brief_direct_sql",
        }
    )
    promotion_blockers = structured_brief_promotion_blockers(compared_metrics)
    compared_metrics.update(
        {
            "structured_brief_promotion_policy": STRUCTURED_BRIEF_PROMOTION_POLICY,
            "structured_brief_promotion_blockers": promotion_blockers,
            "structured_brief_promotion_ready": not promotion_blockers,
        }
    )
    compared["metrics"] = compared_metrics
    compared["command"] = list(structured_manifest.get("command") or []) + [
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


def compare_structured_brief_manifest_files(
    *,
    structured_manifest_path: Path,
    direct_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Compare manifest files and write an augmented structured-brief manifest."""

    structured_manifest = _load_manifest(structured_manifest_path)
    direct_manifest = _load_manifest(direct_manifest_path)
    repo_root = repo_root.resolve()
    compared = compare_structured_brief_manifests(
        structured_manifest=structured_manifest,
        direct_manifest=direct_manifest,
        structured_rows=_load_jsonl(
            _resolve_path(repo_root, structured_manifest.get("output_path"))
        ),
        direct_rows=_load_jsonl(_resolve_path(repo_root, direct_manifest.get("output_path"))),
    )
    _write_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structured-manifest", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_structured_brief_manifest_files(
        structured_manifest_path=args.structured_manifest,
        direct_manifest_path=args.direct_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    metrics = compared["metrics"]
    print(f"Wrote structured-brief comparison manifest to {args.output}")
    print(
        "Value delta vs direct SQL: "
        f"{metrics['structured_brief_value_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
