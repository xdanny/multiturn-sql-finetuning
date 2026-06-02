"""Build paired prediction inputs from clean-holdout Metric DSL gold labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from data.metric_dsl_training_rows import DIRECT_SQL_SYSTEM_PROMPT
from eval.result_manifest import sha256_file

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "metric_dsl_clean_holdout_prediction_input"
SUMMARY_ARTIFACT_TYPE = "metric_dsl_clean_holdout_prediction_input_summary"
MANIFEST_ARTIFACT_TYPE = "metric_dsl_clean_holdout_prediction_input_manifest"
DEFAULT_INPUT = Path("data/processed/metric_dsl/clean_holdout_gold_labels.jsonl")
DEFAULT_METRIC_OUTPUT = Path(
    "data/processed/metric_dsl/clean_holdout_metric_prediction_inputs.jsonl"
)
DEFAULT_DIRECT_OUTPUT = Path(
    "data/processed/metric_dsl/clean_holdout_direct_sql_prediction_inputs.jsonl"
)
DEFAULT_SUMMARY = Path(
    "docs/training_runs/metric_dsl_clean_holdout_prediction_inputs_20260602.json"
)
DEFAULT_MANIFEST = Path(
    "docs/data_artifacts/metric_dsl_clean_holdout_prediction_inputs.manifest.json"
)
DEFAULT_DATABASE_ROOT = Path("data/raw/cosql_dataset/database")
COMPARISON_CONTRACT = "metric_dsl_clean_holdout_prediction_inputs"
NEXT_STEP = (
    "generate same-row Metric DSL and direct-SQL outputs from these inputs, score "
    "them, then run eval.compare_metric_dsl_direct_sql"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _database_path(database_id: str | None, database_root: Path) -> str | None:
    if not database_id:
        return None
    return str(database_root / database_id / f"{database_id}.sqlite")


def _direct_sql_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    updated = [
        {
            "role": "system",
            "content": f"{DIRECT_SQL_SYSTEM_PROMPT} Return only SQL.",
        },
        *[dict(message) for message in messages if message.get("role") != "system"],
    ]
    for message in reversed(updated):
        if message.get("role") == "user":
            content = str(message.get("content") or "").rstrip()
            content = content.replace("Return only the metric DSL.", "Return only SQL.")
            content = content.replace("Return only the metric DSL", "Return only SQL")
            if "Return only SQL" not in content:
                content = f"{content}\n\nReturn only SQL."
            message["content"] = content
            return updated
    return [*updated, {"role": "user", "content": "Return only SQL."}]


def _prompt_text(row: dict[str, Any]) -> str:
    return json.dumps(row.get("messages") or [], sort_keys=True)


def _validate_prediction_input(row: dict[str, Any]) -> None:
    if row.get("reference_sql_visible_to_model") or row.get("scoring_fields_visible_to_model"):
        raise ValueError(f"{row.get('id')} exposes scorer fields to the model prompt")
    prompt = _prompt_text(row)
    for field in ("reference_sql", "gold_dsl"):
        value = row.get(field)
        if value and str(value) in prompt:
            raise ValueError(f"{row.get('id')} leaks {field} into the model prompt")
    if "gold_plan" in row:
        raise ValueError(f"{row.get('id')} carries gold_plan in prediction input")


def _base_prediction_row(row: dict[str, Any], *, database_root: Path) -> dict[str, Any]:
    if not row.get("gold_metric_dsl_available"):
        raise ValueError(f"{row.get('id')} is not a labelled Metric DSL row")
    if row.get("split_role") != "clean_local_holdout":
        raise ValueError(f"{row.get('id')} is not clean_local_holdout")
    base = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "artifact_type",
            "gold_plan",
            "metric_dsl_label_error",
            "readiness_blockers",
            "reference_sql_visible_to_model_prompt",
            "scorer_fields_visible_to_model_prompt",
        }
    }
    base.update(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "comparison_contract": COMPARISON_CONTRACT,
            "oracle_policy": "non_oracle_generation",
            "database_path": _database_path(row.get("database_id"), database_root),
            "reference_sql_visible_to_model": False,
            "scoring_fields_visible_to_model": False,
            "reference_sql_visible_to_model_prompt": False,
            "scorer_fields_visible_to_model_prompt": False,
        }
    )
    return base


def paired_prediction_rows_from_gold_label(
    row: dict[str, Any], *, database_root: Path = DEFAULT_DATABASE_ROOT
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return same-row Metric DSL and direct-SQL prediction inputs."""

    metric_row = _base_prediction_row(row, database_root=database_root)
    metric_row["generation_target"] = "metric_dsl"
    metric_row["evaluation_mode"] = "metric_dsl"

    direct_row = _base_prediction_row(row, database_root=database_root)
    direct_row["generation_target"] = "direct_sql"
    direct_row["evaluation_mode"] = "non_oracle_generation"
    direct_row["messages"] = _direct_sql_messages(row.get("messages") or [])

    _validate_prediction_input(metric_row)
    _validate_prediction_input(direct_row)
    return metric_row, direct_row


def build_metric_dsl_clean_holdout_prediction_inputs(
    input_path: Path,
    *,
    database_root: Path = DEFAULT_DATABASE_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return paired prediction inputs from clean-holdout gold-label rows."""

    metric_rows, direct_rows, _excluded_rows = _build_prediction_inputs_from_rows(
        _load_jsonl(input_path),
        database_root=database_root,
    )
    return metric_rows, direct_rows


def _build_prediction_inputs_from_rows(
    rows: list[dict[str, Any]],
    *,
    database_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows = []
    direct_rows = []
    excluded_rows = []
    for row in rows:
        try:
            metric_row, direct_row = paired_prediction_rows_from_gold_label(
                row,
                database_root=database_root,
            )
        except ValueError as exc:
            reason = str(exc)
            if "leaks reference_sql" not in reason and "leaks gold_dsl" not in reason:
                raise
            excluded_rows.append(
                {
                    "id": row.get("id"),
                    "database_id": row.get("database_id"),
                    "reason": reason,
                }
            )
            continue
        metric_rows.append(metric_row)
        direct_rows.append(direct_row)
    return metric_rows, direct_rows, excluded_rows


def summarize_metric_dsl_clean_holdout_prediction_inputs(
    metric_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
    *,
    input_path: Path,
    database_root: Path,
    input_labelled_count: int | None = None,
    excluded_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    excluded_rows = list(excluded_rows or [])
    split_roles = Counter(str(row.get("split_role") or "unknown") for row in metric_rows)
    split_ids = Counter(str(row.get("split_id") or "unknown") for row in metric_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "checkpoint": 7,
        "run_id": "metric_dsl_clean_holdout_prediction_inputs_20260602",
        "claim_boundary": (
            "Clean-holdout Metric DSL/direct-SQL prediction-input contract only; "
            "no generated outputs, execution scores, value delta, or method win is claimed."
        ),
        "metric_dsl_prediction_input_count": len(metric_rows),
        "direct_sql_prediction_input_count": len(direct_rows),
        "paired_row_count": len(metric_rows),
        "input_labelled_count": (
            input_labelled_count if input_labelled_count is not None else len(metric_rows)
        ),
        "excluded_prompt_leakage_count": len(excluded_rows),
        "excluded_prompt_leakage_rows": excluded_rows,
        "dialog_count": len({row.get("dialog_id") for row in metric_rows}),
        "database_count": len(
            {row.get("database_id") for row in metric_rows if row.get("database_id")}
        ),
        "split_ids": dict(split_ids),
        "split_roles": dict(split_roles),
        "database_path_rows": sum(1 for row in metric_rows if row.get("database_path")),
        "database_root": str(database_root),
        "promotion_status": "not_ready",
        "next_step": NEXT_STEP,
        "oracle_policy": "non_oracle_generation",
        "comparison_contract": COMPARISON_CONTRACT,
        "reference_sql_visible_to_model_prompt": False,
        "scorer_fields_visible_to_model_prompt": False,
        "input_path": str(input_path),
    }


def write_metric_dsl_clean_holdout_prediction_input_artifacts(
    *,
    input_path: Path = DEFAULT_INPUT,
    metric_output_path: Path = DEFAULT_METRIC_OUTPUT,
    direct_output_path: Path = DEFAULT_DIRECT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    manifest_path: Path = DEFAULT_MANIFEST,
    database_root: Path = DEFAULT_DATABASE_ROOT,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write paired clean-holdout prediction inputs, summary, and manifest."""

    input_rows = _load_jsonl(input_path)
    metric_rows, direct_rows, excluded_rows = _build_prediction_inputs_from_rows(
        input_rows,
        database_root=database_root,
    )
    summary = summarize_metric_dsl_clean_holdout_prediction_inputs(
        metric_rows,
        direct_rows,
        input_path=input_path,
        database_root=database_root,
        input_labelled_count=len(input_rows),
        excluded_rows=excluded_rows,
    )
    _write_jsonl(metric_output_path, metric_rows)
    _write_jsonl(direct_output_path, direct_rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "metric_dsl_prediction_input_count": len(metric_rows),
        "direct_sql_prediction_input_count": len(direct_rows),
        "paired_row_count": len(metric_rows),
        "input_labelled_count": len(input_rows),
        "excluded_prompt_leakage_count": len(excluded_rows),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.exists() else None,
        "metric_output_path": str(metric_output_path),
        "metric_output_sha256": sha256_file(metric_output_path),
        "direct_output_path": str(direct_output_path),
        "direct_output_sha256": sha256_file(direct_output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "database_root": str(database_root),
        "oracle_policy": "non_oracle_generation",
        "promotion_status": "not_ready",
        "comparison_contract": COMPARISON_CONTRACT,
        "command": list(command or []),
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--metric-output", type=Path, default=DEFAULT_METRIC_OUTPUT)
    parser.add_argument("--direct-output", type=Path, default=DEFAULT_DIRECT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--database-root", type=Path, default=DEFAULT_DATABASE_ROOT)
    args = parser.parse_args()

    manifest = write_metric_dsl_clean_holdout_prediction_input_artifacts(
        input_path=args.input,
        metric_output_path=args.metric_output,
        direct_output_path=args.direct_output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        database_root=args.database_root,
        command=sys.argv,
    )
    print(
        "Wrote "
        f"{manifest['paired_row_count']} paired clean-holdout Metric DSL prediction inputs"
    )
    return 0 if manifest["paired_row_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
