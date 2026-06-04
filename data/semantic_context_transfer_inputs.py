"""Build Checkpoint 10 normal and semantic-context rollout inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from data.semantic_value_retrieval_inputs import (
    DEFAULT_MAX_MATCHES_PER_TURN,
    DEFAULT_MIN_ALIAS_CHARS,
    build_semantic_value_retrieval_inputs,
)
from eval.result_manifest import sha256_file
from eval.rollout_eval import MODEL_GENERATED_SQL_ROLLOUT, load_rollout_prepared_records
from eval.semantic_context_transfer import write_semantic_context_transfer_preflight

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "semantic_context_transfer_rollout_inputs"
SUMMARY_ARTIFACT_TYPE = "semantic_context_transfer_rollout_input_summary"
PREFLIGHT_ARTIFACT_TYPE = "semantic_context_transfer_preflight"
DEFAULT_INPUT = Path("data/processed/direct_sql_full/cosql_dev_clean_holdout_v1.jsonl")
DEFAULT_VALUE_INDEX = Path("docs/data_artifacts/value_index_cosql_dev_100.jsonl")
DEFAULT_VALUE_INDEX_MANIFEST = Path(
    "docs/data_artifacts/value_index_cosql_dev_100.manifest.json"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/semantic_context_transfer")
DEFAULT_NORMAL_OUTPUT = DEFAULT_OUTPUT_DIR / "cp10_limit12_normal.jsonl"
DEFAULT_SEMANTIC_OUTPUT = DEFAULT_OUTPUT_DIR / "cp10_limit12_semantic_value.jsonl"
DEFAULT_SUMMARY = Path(
    "docs/data_artifacts/semantic_context_transfer_cp10_limit12_summary.json"
)
DEFAULT_MANIFEST = Path(
    "docs/data_artifacts/semantic_context_transfer_cp10_limit12.manifest.json"
)
DEFAULT_PREFLIGHT = Path(
    "docs/training_runs/semantic_context_transfer_cp10_limit12_preflight_20260604.json"
)
DEFAULT_LIMIT_DIALOGS = 12
VALID_SOURCE_HISTORY_POLICIES = {
    "gold_sql_teacher_forced",
    "seeded_generated_failure_then_rollout",
    MODEL_GENERATED_SQL_ROLLOUT,
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assistant_turn_count(record: dict[str, Any]) -> int:
    return sum(1 for message in record.get("messages", []) if message.get("role") == "assistant")


def _dialog_id(record: dict[str, Any], *, index: int) -> str:
    return str(record.get("dialog_id") or record.get("id") or f"prepared-{index}")


def _rollout_compatible(record: dict[str, Any]) -> bool:
    if record.get("history_policy") not in VALID_SOURCE_HISTORY_POLICIES:
        return False
    return _assistant_turn_count(record) >= 2


def select_semantic_context_transfer_rows(
    *,
    input_path: Path,
    limit_dialogs: int,
) -> list[dict[str, Any]]:
    """Return the bounded clean-holdout dialog slice for Checkpoint 10."""

    selected = []
    for record in load_rollout_prepared_records(input_path, allow_oracle_plan=False):
        if not _rollout_compatible(record):
            continue
        selected.append(
            {
                **record,
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "checkpoint": 10,
                "comparison_contract": "semantic_context_transfer_generated_history_rollout",
                "rollout_target_history_policy": MODEL_GENERATED_SQL_ROLLOUT,
                "semantic_context_policy": "normal_schema_context",
                "reference_sql_visible_to_model_prompt": False,
                "future_turns_visible_to_model_prompt": False,
                "scorer_labels_visible_to_model_prompt": False,
            }
        )
        if len(selected) >= limit_dialogs:
            break
    if len(selected) < limit_dialogs:
        raise ValueError(
            f"requested {limit_dialogs} rollout-compatible dialogs, found {len(selected)}"
        )
    return selected


def _value_index_database_ids(value_index_rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("database_id")) for row in value_index_rows if row.get("database_id")}


def _validate_value_index_coverage(
    *,
    selected_rows: list[dict[str, Any]],
    value_index_rows: list[dict[str, Any]],
    value_index_manifest: dict[str, Any],
    value_index_path: Path,
) -> None:
    if value_index_manifest.get("artifact_type") != "non_oracle_value_index_v1":
        raise ValueError("value-index manifest must use artifact_type=non_oracle_value_index_v1")
    if value_index_manifest.get("index_source") != "database_contents":
        raise ValueError("semantic context transfer requires a database-derived value index")
    if value_index_manifest.get("output_sha256") != sha256_file(value_index_path):
        raise ValueError("value-index manifest output_sha256 does not match value-index file")
    manifest_output_path = value_index_manifest.get("output_path")
    if (
        manifest_output_path is not None
        and Path(manifest_output_path).resolve() != value_index_path.resolve()
    ):
        raise ValueError("value-index manifest output_path does not match value-index file")
    missing = {
        str(row.get("database_id"))
        for row in selected_rows
        if row.get("database_id")
    } - _value_index_database_ids(value_index_rows)
    if missing:
        raise ValueError(
            "value index is missing selected database(s): " + ", ".join(sorted(missing))
        )


def _summarize_rows(
    *,
    normal_rows: list[dict[str, Any]],
    semantic_summary: dict[str, Any],
    input_path: Path,
    value_index_path: Path,
    value_index_manifest_path: Path,
    limit_dialogs: int,
) -> dict[str, Any]:
    source_history_policies = Counter(
        str(row.get("history_policy") or "unknown") for row in normal_rows
    )
    split_roles = Counter(str(row.get("split_role") or "unknown") for row in normal_rows)
    assistant_turn_count = sum(_assistant_turn_count(row) for row in normal_rows)
    dialog_ids = [_dialog_id(row, index=index) for index, row in enumerate(normal_rows)]
    database_ids = sorted({str(row.get("database_id")) for row in normal_rows})
    split_row_ids = [
        str(row.get("split_row_id"))
        for row in normal_rows
        if row.get("split_row_id") is not None
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "checkpoint": 10,
        "claim_boundary": (
            "Input preparation and preflight only; no model generation, SQL execution "
            "delta, hosted comparison, or semantic-context win is claimed."
        ),
        "comparison_contract": "semantic_context_transfer_generated_history_rollout",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "value_index_path": str(value_index_path),
        "value_index_sha256": sha256_file(value_index_path),
        "value_index_manifest_path": str(value_index_manifest_path),
        "value_index_manifest_sha256": sha256_file(value_index_manifest_path),
        "limit_dialogs": limit_dialogs,
        "dialog_count": len(normal_rows),
        "assistant_turn_count": assistant_turn_count,
        "dialog_ids": dialog_ids,
        "split_row_ids": split_row_ids,
        "database_count": len(database_ids),
        "database_ids": database_ids,
        "split_roles": dict(sorted(split_roles.items())),
        "source_history_policies": dict(sorted(source_history_policies.items())),
        "rollout_target_history_policy": MODEL_GENERATED_SQL_ROLLOUT,
        "normal_prompt_policy": "normal_schema_context",
        "semantic_prompt_policy": "schema_context_plus_database_value_retrieval",
        "semantic_matched_row_count": semantic_summary["matched_row_count"],
        "semantic_matched_turn_count": semantic_summary["matched_turn_count"],
        "semantic_matched_value_count": semantic_summary["matched_value_count"],
        "reference_sql_visible_to_model_prompt": False,
        "future_turns_visible_to_model_prompt": False,
        "scorer_labels_visible_to_model_prompt": False,
    }


def write_semantic_context_transfer_input_artifacts(
    *,
    input_path: Path = DEFAULT_INPUT,
    value_index_path: Path = DEFAULT_VALUE_INDEX,
    value_index_manifest_path: Path = DEFAULT_VALUE_INDEX_MANIFEST,
    normal_output_path: Path = DEFAULT_NORMAL_OUTPUT,
    semantic_output_path: Path = DEFAULT_SEMANTIC_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    manifest_path: Path = DEFAULT_MANIFEST,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    limit_dialogs: int = DEFAULT_LIMIT_DIALOGS,
    max_matches_per_turn: int = DEFAULT_MAX_MATCHES_PER_TURN,
    retrieval_scope: str = "current_turn",
    min_alias_chars: int = DEFAULT_MIN_ALIAS_CHARS,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write Checkpoint 10 input artifacts and preflight evidence."""

    normal_rows = select_semantic_context_transfer_rows(
        input_path=input_path,
        limit_dialogs=limit_dialogs,
    )
    value_index_rows = _load_jsonl(value_index_path)
    value_index_manifest = _load_json(value_index_manifest_path)
    _validate_value_index_coverage(
        selected_rows=normal_rows,
        value_index_rows=value_index_rows,
        value_index_manifest=value_index_manifest,
        value_index_path=value_index_path,
    )
    semantic_rows, semantic_summary = build_semantic_value_retrieval_inputs(
        prepared_rows=normal_rows,
        value_index_rows=value_index_rows,
        max_matches_per_turn=max_matches_per_turn,
        retrieval_scope=retrieval_scope,
        min_alias_chars=min_alias_chars,
    )
    for row in semantic_rows:
        row["semantic_context_policy"] = "schema_context_plus_database_value_retrieval"

    _write_jsonl(normal_output_path, normal_rows)
    _write_jsonl(semantic_output_path, semantic_rows)
    summary = _summarize_rows(
        normal_rows=normal_rows,
        semantic_summary=semantic_summary,
        input_path=input_path,
        value_index_path=value_index_path,
        value_index_manifest_path=value_index_manifest_path,
        limit_dialogs=limit_dialogs,
    )
    summary.update(
        {
            "normal_output_path": str(normal_output_path),
            "normal_output_sha256": sha256_file(normal_output_path),
            "semantic_output_path": str(semantic_output_path),
            "semantic_output_sha256": sha256_file(semantic_output_path),
            "max_matches_per_turn": max_matches_per_turn,
            "retrieval_scope": retrieval_scope,
            "min_alias_chars": min_alias_chars,
        }
    )
    _write_json(summary_path, summary)
    preflight = write_semantic_context_transfer_preflight(
        normal_input_path=normal_output_path,
        semantic_input_path=semantic_output_path,
        value_index_manifest_path=value_index_manifest_path,
        output_path=preflight_path,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "checkpoint": 10,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "value_index_path": str(value_index_path),
        "value_index_sha256": sha256_file(value_index_path),
        "value_index_manifest_path": str(value_index_manifest_path),
        "value_index_manifest_sha256": sha256_file(value_index_manifest_path),
        "normal_output_path": str(normal_output_path),
        "normal_output_sha256": sha256_file(normal_output_path),
        "semantic_output_path": str(semantic_output_path),
        "semantic_output_sha256": sha256_file(semantic_output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "dialog_count": summary["dialog_count"],
        "assistant_turn_count": summary["assistant_turn_count"],
        "database_count": summary["database_count"],
        "semantic_matched_row_count": summary["semantic_matched_row_count"],
        "semantic_matched_turn_count": summary["semantic_matched_turn_count"],
        "semantic_matched_value_count": summary["semantic_matched_value_count"],
        "rollout_target_history_policy": MODEL_GENERATED_SQL_ROLLOUT,
        "source_history_policies": summary["source_history_policies"],
        "preflight_status": preflight["status"],
        "oracle_policy": "non_oracle_generation",
        "command": list(command or []),
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--value-index", type=Path, default=DEFAULT_VALUE_INDEX)
    parser.add_argument(
        "--value-index-manifest",
        type=Path,
        default=DEFAULT_VALUE_INDEX_MANIFEST,
    )
    parser.add_argument("--normal-output", type=Path, default=DEFAULT_NORMAL_OUTPUT)
    parser.add_argument("--semantic-output", type=Path, default=DEFAULT_SEMANTIC_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--limit-dialogs", type=int, default=DEFAULT_LIMIT_DIALOGS)
    parser.add_argument("--max-matches-per-turn", type=int, default=DEFAULT_MAX_MATCHES_PER_TURN)
    parser.add_argument("--retrieval-scope", choices=("current_turn", "history"), default="current_turn")
    parser.add_argument("--min-alias-chars", type=int, default=DEFAULT_MIN_ALIAS_CHARS)
    args = parser.parse_args()

    manifest = write_semantic_context_transfer_input_artifacts(
        input_path=args.input,
        value_index_path=args.value_index,
        value_index_manifest_path=args.value_index_manifest,
        normal_output_path=args.normal_output,
        semantic_output_path=args.semantic_output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        preflight_path=args.preflight_output,
        limit_dialogs=args.limit_dialogs,
        max_matches_per_turn=args.max_matches_per_turn,
        retrieval_scope=args.retrieval_scope,
        min_alias_chars=args.min_alias_chars,
        command=sys.argv,
    )
    print(
        "Wrote Checkpoint 10 semantic-context input manifest to "
        f"{args.manifest_output}"
    )
    print(f"Dialog count: {manifest['dialog_count']}")
    print(f"Preflight status: {manifest['preflight_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
