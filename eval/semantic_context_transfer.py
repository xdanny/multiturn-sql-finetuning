"""Validate and compare semantic-context generated-history rollout runs.

This module does not call models. It verifies that normal-schema and
semantic/value-context rollout inputs or result manifests are row-matched,
non-oracle, and suitable for Checkpoint 10 evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval.compare_rollout_history import MODEL_GENERATED_SQL_ROLLOUT
from eval.result_manifest import sha256_file
from eval.run_eval import record_uses_oracle_plan

ARTIFACT_TYPE_PREFLIGHT = "semantic_context_transfer_preflight"
ARTIFACT_TYPE_COMPARISON = "semantic_context_transfer_comparison"
NON_ORACLE_GENERATION = "non_oracle_generation"
VALUE_INDEX_ARTIFACT_TYPE = "non_oracle_value_index_v1"
DATABASE_INDEX_SOURCE = "database_contents"
MAX_ALLOWED_STRICT_REGRESSION = 0.01
MAX_ALLOWED_SYNTAX_REGRESSION = 0.01
VALID_SOURCE_HISTORY_POLICIES = {
    "gold_sql_teacher_forced",
    "seeded_generated_failure_then_rollout",
    MODEL_GENERATED_SQL_ROLLOUT,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metric(manifest: dict[str, Any], name: str) -> float:
    value = (manifest.get("metrics") or {}).get(name)
    if value is None:
        raise ValueError(f"manifest {manifest.get('run_id')} is missing metric {name}")
    return float(value)


def _record_dialog_id(record: dict[str, Any], *, index: int) -> str:
    return str(record.get("dialog_id") or record.get("id") or f"prepared-{index}")


def _record_turn_identities(record: dict[str, Any], *, index: int) -> list[tuple[str, str, str, str]]:
    identities = []
    turn_index = 0
    for message in record.get("messages") or []:
        if message.get("role") != "assistant":
            continue
        identities.append(
            (
                _record_dialog_id(record, index=index),
                str(turn_index),
                str(record.get("database_id")),
                str(message.get("content") or ""),
            )
        )
        turn_index += 1
    if not identities:
        raise ValueError("prepared rollout record has no assistant reference SQL")
    return identities


def _input_identities(records: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    identities = []
    for index, record in enumerate(records):
        identities.extend(_record_turn_identities(record, index=index))
    return identities


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


def _reject_duplicates(identities: list[tuple[str, str, str, str]], *, label: str) -> None:
    counts = Counter(identities)
    if any(count > 1 for count in counts.values()):
        raise ValueError(f"duplicate {label} row identity")


def _record_uses_oracle(record: dict[str, Any]) -> bool:
    return bool(record.get("semantic_model_oracle_derived")) or record_uses_oracle_plan(record)


def _row_uses_oracle(row: dict[str, Any]) -> bool:
    return bool(
        row.get("uses_oracle_planning_hints")
        or row.get("semantic_context_pruned_by_oracle_labels")
        or row.get("semantic_model_oracle_derived")
    )


def _validate_value_index_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("artifact_type") != VALUE_INDEX_ARTIFACT_TYPE:
        raise ValueError(
            f"value-index manifest must use artifact_type={VALUE_INDEX_ARTIFACT_TYPE}"
        )
    if manifest.get("index_source") != DATABASE_INDEX_SOURCE:
        raise ValueError("value-index manifest must be database-derived")


def _validate_preflight_inputs(
    *,
    normal_records: list[dict[str, Any]],
    semantic_records: list[dict[str, Any]],
) -> dict[str, Any]:
    if not normal_records or not semantic_records:
        raise ValueError("preflight requires non-empty normal and semantic inputs")
    if len(normal_records) != len(semantic_records):
        raise ValueError("normal and semantic inputs must have matching dialog counts")
    if any(_record_uses_oracle(record) for record in normal_records + semantic_records):
        raise ValueError("semantic context transfer inputs must be non-oracle")
    normal_semantic_markers = [
        record for record in normal_records if record.get("semantic_value_retrieval")
    ]
    if normal_semantic_markers:
        raise ValueError("normal input must not already include semantic value-retrieval context")
    semantic_markers = [record.get("semantic_value_retrieval") for record in semantic_records]
    if not all(
        isinstance(marker, dict) and marker.get("index_source") == DATABASE_INDEX_SOURCE
        for marker in semantic_markers
    ):
        raise ValueError("semantic input must carry database-derived semantic value context")

    normal_identities = _input_identities(normal_records)
    semantic_identities = _input_identities(semantic_records)
    _reject_duplicates(normal_identities, label="normal input")
    _reject_duplicates(semantic_identities, label="semantic input")
    if normal_identities != semantic_identities:
        raise ValueError("normal and semantic input row identity mismatch")

    evaluation_modes = Counter(
        str(record.get("evaluation_mode") or "unknown")
        for record in normal_records + semantic_records
    )
    if set(evaluation_modes) != {NON_ORACLE_GENERATION}:
        raise ValueError("normal and semantic inputs must use non_oracle_generation mode")
    source_history_policies = Counter(
        str(record.get("history_policy") or "unknown")
        for record in normal_records + semantic_records
    )
    invalid_source_history_policies = (
        set(source_history_policies) - VALID_SOURCE_HISTORY_POLICIES
    )
    if invalid_source_history_policies:
        invalid = ", ".join(sorted(invalid_source_history_policies))
        raise ValueError(
            "normal and semantic inputs must use rollout-compatible source "
            f"history policies, got: {invalid}"
        )
    dialog_ids = [_record_dialog_id(record, index=index) for index, record in enumerate(normal_records)]
    return {
        "row_count": len(normal_identities),
        "dialog_count": len(normal_records),
        "database_count": len({str(record.get("database_id")) for record in normal_records}),
        "dialog_ids_sha256": _sha256_json(dialog_ids),
        "row_identities_sha256": _sha256_json(normal_identities),
        "source_history_policies": dict(sorted(source_history_policies.items())),
        "rollout_target_history_policy": MODEL_GENERATED_SQL_ROLLOUT,
    }


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()

    return hashlib.sha256(encoded).hexdigest()


def build_semantic_context_transfer_preflight(
    *,
    normal_input_path: Path,
    semantic_input_path: Path,
    value_index_manifest_path: Path,
) -> dict[str, Any]:
    """Return a preflight artifact for row-matched normal and semantic inputs."""

    normal_sha = sha256_file(normal_input_path)
    semantic_sha = sha256_file(semantic_input_path)
    value_index_sha = sha256_file(value_index_manifest_path)
    if normal_sha is None or semantic_sha is None:
        raise ValueError("normal and semantic input files must exist")
    if value_index_sha is None:
        raise ValueError("value-index manifest does not exist")
    if normal_sha == semantic_sha:
        raise ValueError("normal and semantic inputs must differ")

    value_index_manifest = _load_json(value_index_manifest_path)
    _validate_value_index_manifest(value_index_manifest)
    summary = _validate_preflight_inputs(
        normal_records=_load_jsonl(normal_input_path),
        semantic_records=_load_jsonl(semantic_input_path),
    )
    return {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE_PREFLIGHT,
        "status": "ready_for_semantic_context_rollout_pair",
        "claim_boundary": "preflight only; no SQL execution claim",
        "normal_input_path": str(normal_input_path),
        "normal_input_sha256": normal_sha,
        "semantic_input_path": str(semantic_input_path),
        "semantic_input_sha256": semantic_sha,
        "value_index_manifest_path": str(value_index_manifest_path),
        "value_index_manifest_sha256": value_index_sha,
        "value_index_index_source": value_index_manifest.get("index_source"),
        "oracle_policy": NON_ORACLE_GENERATION,
        "history_policy": MODEL_GENERATED_SQL_ROLLOUT,
        "leakage_boundary": (
            "semantic/value context may use schema-derived context and database-derived "
            "value matches, but not reference SQL, expected rows, future turns, gold DSL, "
            "repair labels, or answer-derived table or column hints"
        ),
        **summary,
    }


def write_semantic_context_transfer_preflight(
    *,
    normal_input_path: Path,
    semantic_input_path: Path,
    value_index_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    preflight = build_semantic_context_transfer_preflight(
        normal_input_path=normal_input_path,
        semantic_input_path=semantic_input_path,
        value_index_manifest_path=value_index_manifest_path,
    )
    _write_json(output_path, preflight)
    return preflight


def _validate_rollout_manifest(manifest: dict[str, Any], *, label: str) -> None:
    if manifest.get("oracle_allowed") or manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError(f"{label} manifest must be non-oracle")
    if manifest.get("benchmark") != "prepared_rollout":
        raise ValueError(f"{label} manifest must use benchmark=prepared_rollout")
    if (manifest.get("metrics") or {}).get("history_policy") != MODEL_GENERATED_SQL_ROLLOUT:
        raise ValueError(f"{label} manifest must use model-generated rollout history")
    for metric in (
        "value_execution_accuracy",
        "strict_execution_accuracy",
        "syntax_accuracy",
    ):
        _metric(manifest, metric)


def _validate_preflight_manifest(
    preflight_manifest: dict[str, Any],
    *,
    normal_manifest: dict[str, Any],
    semantic_manifest: dict[str, Any],
) -> None:
    if preflight_manifest.get("artifact_type") != ARTIFACT_TYPE_PREFLIGHT:
        raise ValueError("preflight manifest must be a semantic context transfer preflight")
    if preflight_manifest.get("normal_input_sha256") != normal_manifest.get("input_sha256"):
        raise ValueError("normal manifest input does not match preflight")
    if preflight_manifest.get("semantic_input_sha256") != semantic_manifest.get("input_sha256"):
        raise ValueError("semantic manifest input does not match preflight")
    if preflight_manifest.get("value_index_index_source") != DATABASE_INDEX_SOURCE:
        raise ValueError("preflight value index must be database-derived")


def _validate_result_rows(
    *,
    normal_manifest: dict[str, Any],
    semantic_manifest: dict[str, Any],
    normal_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
) -> None:
    if not normal_rows or not semantic_rows:
        raise ValueError("comparison requires non-empty output rows")
    if int(normal_manifest.get("row_count") or 0) != len(normal_rows):
        raise ValueError("normal manifest row_count does not match output rows")
    if int(semantic_manifest.get("row_count") or 0) != len(semantic_rows):
        raise ValueError("semantic manifest row_count does not match output rows")
    if {row.get("history_policy") for row in normal_rows} != {MODEL_GENERATED_SQL_ROLLOUT}:
        raise ValueError("normal output rows must use model-generated rollout history")
    if {row.get("history_policy") for row in semantic_rows} != {MODEL_GENERATED_SQL_ROLLOUT}:
        raise ValueError("semantic output rows must use model-generated rollout history")
    if {row.get("evaluation_mode") for row in normal_rows + semantic_rows} != {
        NON_ORACLE_GENERATION
    }:
        raise ValueError("comparison rows must use non_oracle_generation mode")
    if any(_row_uses_oracle(row) for row in normal_rows + semantic_rows):
        raise ValueError("oracle-derived rows found in semantic context transfer comparison")
    if any(
        row.get("value_execution_score") is None
        or row.get("strict_execution_score") is None
        or row.get("syntax_valid") is None
        for row in normal_rows + semantic_rows
    ):
        raise ValueError("comparison rows must include value, strict, and syntax scores")

    normal_identities = [_row_identity(row) for row in normal_rows]
    semantic_identities = [_row_identity(row) for row in semantic_rows]
    _reject_duplicates(normal_identities, label="normal result")
    _reject_duplicates(semantic_identities, label="semantic result")
    if normal_identities != semantic_identities:
        raise ValueError("normal and semantic result row identity mismatch")


def _semantic_context_blockers(
    *,
    value_delta: float,
    strict_delta: float,
    syntax_delta: float,
) -> list[str]:
    blockers = []
    if value_delta <= 0.0:
        blockers.append("value accuracy did not improve")
    if strict_delta < -MAX_ALLOWED_STRICT_REGRESSION:
        blockers.append("strict accuracy regressed beyond policy")
    if syntax_delta < -MAX_ALLOWED_SYNTAX_REGRESSION:
        blockers.append("syntax rate regressed beyond policy")
    return blockers


def compare_semantic_context_transfer_manifests(
    *,
    normal_manifest: dict[str, Any],
    semantic_manifest: dict[str, Any],
    normal_rows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    comparison_role: str,
    preflight_manifest: dict[str, Any] | None = None,
    preflight_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Return an augmented semantic manifest with normal-context deltas."""

    _validate_rollout_manifest(normal_manifest, label="normal-context")
    _validate_rollout_manifest(semantic_manifest, label="semantic-context")
    if normal_manifest.get("model_name") != semantic_manifest.get("model_name"):
        raise ValueError("normal and semantic manifests must use the same model")
    if normal_manifest.get("endpoint") != semantic_manifest.get("endpoint"):
        raise ValueError("normal and semantic manifests must use the same endpoint")
    if normal_manifest.get("input_sha256") == semantic_manifest.get("input_sha256"):
        raise ValueError("normal and semantic manifests must use different inputs")
    if int(normal_manifest.get("row_count") or 0) != int(semantic_manifest.get("row_count") or 0):
        raise ValueError("normal and semantic manifests must have matching row_count")
    if preflight_manifest is not None:
        _validate_preflight_manifest(
            preflight_manifest,
            normal_manifest=normal_manifest,
            semantic_manifest=semantic_manifest,
        )
    _validate_result_rows(
        normal_manifest=normal_manifest,
        semantic_manifest=semantic_manifest,
        normal_rows=normal_rows,
        semantic_rows=semantic_rows,
    )

    normal_value = _metric(normal_manifest, "value_execution_accuracy")
    normal_strict = _metric(normal_manifest, "strict_execution_accuracy")
    normal_syntax = _metric(normal_manifest, "syntax_accuracy")
    semantic_value = _metric(semantic_manifest, "value_execution_accuracy")
    semantic_strict = _metric(semantic_manifest, "strict_execution_accuracy")
    semantic_syntax = _metric(semantic_manifest, "syntax_accuracy")
    value_delta = semantic_value - normal_value
    strict_delta = semantic_strict - normal_strict
    syntax_delta = semantic_syntax - normal_syntax
    blockers = _semantic_context_blockers(
        value_delta=value_delta,
        strict_delta=strict_delta,
        syntax_delta=syntax_delta,
    )

    compared = dict(semantic_manifest)
    compared["artifact_type"] = ARTIFACT_TYPE_COMPARISON
    metrics = dict(semantic_manifest.get("metrics") or {})
    metrics.update(
        {
            "semantic_context_transfer_comparison_role": comparison_role,
            "semantic_context_transfer_comparer": "eval.semantic_context_transfer",
            "normal_context_comparison_run_id": normal_manifest.get("run_id"),
            "normal_context_model_name": normal_manifest.get("model_name"),
            "normal_context_input_sha256": normal_manifest.get("input_sha256"),
            "normal_context_output_sha256": normal_manifest.get("output_sha256"),
            "normal_context_value_execution_accuracy": normal_value,
            "normal_context_strict_execution_accuracy": normal_strict,
            "normal_context_syntax_accuracy": normal_syntax,
            "semantic_context_value_delta_vs_normal": value_delta,
            "semantic_context_strict_delta_vs_normal": strict_delta,
            "semantic_context_syntax_delta_vs_normal": syntax_delta,
            "semantic_context_transfer_comparable_row_count": len(semantic_rows),
            "semantic_context_transfer_policy": {
                "value_delta_must_be_positive": True,
                "max_allowed_strict_regression": MAX_ALLOWED_STRICT_REGRESSION,
                "max_allowed_syntax_regression": MAX_ALLOWED_SYNTAX_REGRESSION,
            },
            "semantic_context_transfer_blockers": blockers,
            "semantic_context_helped": not blockers,
        }
    )
    if preflight_manifest is not None:
        metrics.update(
            {
                "semantic_context_transfer_value_index_manifest_sha256": (
                    preflight_manifest.get("value_index_manifest_sha256")
                ),
            }
        )
        if preflight_manifest_sha256 is not None:
            metrics["semantic_context_transfer_preflight_sha256"] = (
                preflight_manifest_sha256
            )
        else:
            metrics["semantic_context_transfer_preflight_payload_sha256"] = _sha256_json(
                preflight_manifest
            )
    compared["metrics"] = metrics
    compared["command"] = list(semantic_manifest.get("command") or []) + [
        "# compared-with-normal-context",
        str(normal_manifest.get("run_id")),
    ]
    return compared


def _resolve_path(repo_root: Path, path_value: str | None) -> Path:
    if not path_value:
        raise ValueError("manifest is missing output_path")
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def compare_semantic_context_transfer_manifest_files(
    *,
    normal_manifest_path: Path,
    semantic_manifest_path: Path,
    output_path: Path,
    comparison_role: str,
    preflight_manifest_path: Path | None = None,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Compare manifest files and write an augmented semantic manifest."""

    repo_root = repo_root.resolve()
    normal_manifest = _load_json(normal_manifest_path)
    semantic_manifest = _load_json(semantic_manifest_path)
    preflight_manifest = (
        _load_json(preflight_manifest_path) if preflight_manifest_path is not None else None
    )
    preflight_manifest_sha256 = (
        sha256_file(preflight_manifest_path) if preflight_manifest_path is not None else None
    )
    compared = compare_semantic_context_transfer_manifests(
        normal_manifest=normal_manifest,
        semantic_manifest=semantic_manifest,
        normal_rows=_load_jsonl(_resolve_path(repo_root, normal_manifest.get("output_path"))),
        semantic_rows=_load_jsonl(_resolve_path(repo_root, semantic_manifest.get("output_path"))),
        comparison_role=comparison_role,
        preflight_manifest=preflight_manifest,
        preflight_manifest_sha256=preflight_manifest_sha256,
    )
    _write_json(output_path, compared)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--normal-input", type=Path, required=True)
    preflight.add_argument("--semantic-input", type=Path, required=True)
    preflight.add_argument("--value-index-manifest", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--normal-manifest", type=Path, required=True)
    compare.add_argument("--semantic-manifest", type=Path, required=True)
    compare.add_argument("--comparison-role", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--preflight-manifest", type=Path, default=None)
    compare.add_argument("--repo-root", type=Path, default=Path("."))

    args = parser.parse_args()
    if args.command == "preflight":
        artifact = write_semantic_context_transfer_preflight(
            normal_input_path=args.normal_input,
            semantic_input_path=args.semantic_input,
            value_index_manifest_path=args.value_index_manifest,
            output_path=args.output,
        )
        print(f"Wrote semantic context transfer preflight to {args.output}")
        print(f"Row count: {artifact['row_count']}")
        return 0
    compared = compare_semantic_context_transfer_manifest_files(
        normal_manifest_path=args.normal_manifest,
        semantic_manifest_path=args.semantic_manifest,
        output_path=args.output,
        comparison_role=args.comparison_role,
        preflight_manifest_path=args.preflight_manifest,
        repo_root=args.repo_root,
    )
    metrics = compared["metrics"]
    print(f"Wrote semantic context transfer comparison to {args.output}")
    print(
        "Value delta vs normal context: "
        f"{metrics['semantic_context_value_delta_vs_normal']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
