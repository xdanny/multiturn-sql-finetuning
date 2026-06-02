"""Build Metric DSL clean-holdout candidate rows and readiness summaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from data.metric_dsl_training_rows import METRIC_DSL_SYSTEM_PROMPT
from eval.result_manifest import sha256_file
from eval.run_eval import load_prepared_records

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "metric_dsl_clean_holdout_candidate"
SUMMARY_ARTIFACT_TYPE = "metric_dsl_clean_holdout_candidate_summary"
MANIFEST_ARTIFACT_TYPE = "metric_dsl_clean_holdout_candidate_manifest"
REQUIRED_SPLIT_ROLE = "clean_local_holdout"
DEFAULT_INPUT = Path("data/processed/direct_sql_full/cosql_dev_clean_holdout_v1.jsonl")
DEFAULT_OUTPUT = Path("data/processed/metric_dsl/clean_holdout_candidates.jsonl")
DEFAULT_SUMMARY = Path("docs/training_runs/metric_dsl_clean_holdout_readiness_20260602.json")
DEFAULT_MANIFEST = Path("docs/data_artifacts/metric_dsl_clean_holdout_candidates.manifest.json")
GOLD_DSL_BLOCKER = "structured gold Metric DSL labels missing"
NEXT_STEP = (
    "derive or author scorer-side gold Metric DSL labels, then generate same-row "
    "Metric DSL and direct-SQL predictions for eval.run_metric_dsl_comparison"
)


def _metric_signals(turn: dict[str, Any]) -> list[str]:
    plan = turn.get("gold_plan") or {}
    skeleton = plan.get("query_skeleton") or {}
    projection = plan.get("projection_shape") or {}
    signals = []
    aggregations = projection.get("aggregations") or []
    selected_expressions = [
        str(expression).lower()
        for expression in projection.get("selected_expressions") or []
    ]
    has_aggregate_expression = any(
        token in expression
        for expression in selected_expressions
        for token in ("count(", "sum(", "avg(", "min(", "max(")
    )
    if aggregations or has_aggregate_expression:
        signals.append("aggregation")
    if skeleton.get("group_by") or projection.get("group_by"):
        signals.append("group_by")
    if skeleton.get("having"):
        signals.append("having")
    if skeleton.get("order_by") or projection.get("order_by"):
        signals.append("order_by")
    if skeleton.get("limit") or projection.get("limit"):
        signals.append("limit")
    return signals


def _metric_prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    updated = [
        {
            "role": "system",
            "content": f"{METRIC_DSL_SYSTEM_PROMPT} Return only the metric DSL.",
        },
        *[dict(message) for message in messages if message.get("role") != "system"],
    ]
    for message in reversed(updated):
        if message.get("role") == "user":
            message["content"] = (
                f"{message.get('content', '').rstrip()}\n\nReturn only the metric DSL."
            )
            break
    return updated


def _validate_no_current_label_leakage(row: dict[str, Any]) -> None:
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "user":
        raise ValueError(f"{row.get('id')}: prompt must stop before the current assistant label")
    last_user_index = max(
        index for index, message in enumerate(messages) if message.get("role") == "user"
    )
    if any(message.get("role") == "assistant" for message in messages[last_user_index + 1 :]):
        raise ValueError(f"{row.get('id')}: current assistant label leaked into prompt")
    prompt = json.dumps(messages, sort_keys=True)
    for marker in ("schema_link_labels", "gold_plan", "gold_metric_dsl"):
        if marker in prompt:
            raise ValueError(f"{row.get('id')}: scorer field leaked into prompt")


def metric_dsl_clean_holdout_candidate_from_turn(turn: dict[str, Any]) -> dict[str, Any] | None:
    """Return one candidate row when an expanded clean-holdout turn is metric-shaped."""

    if turn.get("split_role") != REQUIRED_SPLIT_ROLE:
        raise ValueError(
            f"{turn.get('id')}: Metric DSL candidates require split_role={REQUIRED_SPLIT_ROLE}"
        )
    signals = _metric_signals(turn)
    if not signals:
        return None
    row = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "id": str(turn["id"]),
        "dialog_id": str(turn.get("dialog_id") or ""),
        "turn_index": int(turn.get("turn_index") or 0),
        "turn_count": int(turn.get("turn_count") or 0),
        "database_id": turn.get("database_id"),
        "source": turn.get("source"),
        "split_id": turn.get("split_id"),
        "split_role": turn.get("split_role"),
        "split_row_id": turn.get("split_row_id"),
        "history_policy": turn.get("history_policy"),
        "evaluation_mode": "metric_dsl",
        "generation_target": "metric_dsl",
        "messages": _metric_prompt_messages(turn.get("messages", [])),
        "reference_sql": turn.get("reference_sql"),
        "gold_plan": turn.get("gold_plan"),
        "metric_signals": signals,
        "oracle_policy": "non_oracle_generation",
        "semantic_model_source": "prepared_prompt_context_non_oracle",
        "label_source": "clean_holdout_reference_sql_scorer_side",
        "comparison_contract": "metric_dsl_clean_holdout_candidate_slice",
        "reference_sql_visible_to_model_prompt": False,
        "scorer_fields_visible_to_model_prompt": False,
        "gold_metric_dsl_available": False,
        "readiness_blockers": [GOLD_DSL_BLOCKER],
    }
    _validate_no_current_label_leakage(row)
    return row


def build_metric_dsl_clean_holdout_candidates(input_path: Path) -> list[dict[str, Any]]:
    """Return metric-shaped clean-holdout turns from prepared direct-SQL dialogs."""

    candidates = []
    for turn in load_prepared_records(input_path, allow_oracle_plan=False):
        candidate = metric_dsl_clean_holdout_candidate_from_turn(turn)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def summarize_metric_dsl_clean_holdout_candidates(
    rows: list[dict[str, Any]], *, input_path: Path
) -> dict[str, Any]:
    split_ids = Counter(str(row.get("split_id") or "unknown") for row in rows)
    split_roles = Counter(str(row.get("split_role") or "unknown") for row in rows)
    signal_counts = Counter(signal for row in rows for signal in row["metric_signals"])
    blockers = Counter(blocker for row in rows for blocker in row["readiness_blockers"])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "checkpoint": 7,
        "run_id": "metric_dsl_clean_holdout_readiness_20260602",
        "claim_boundary": (
            "Clean-holdout Metric DSL candidate-slice readiness only; no generated "
            "DSL, compiled SQL execution score, value delta, or method win is claimed."
        ),
        "candidate_count": len(rows),
        "dialog_count": len({row.get("dialog_id") for row in rows}),
        "database_count": len({row.get("database_id") for row in rows if row.get("database_id")}),
        "split_ids": dict(split_ids),
        "split_roles": dict(split_roles),
        "metric_signal_counts": dict(sorted(signal_counts.items())),
        "readiness_blockers": dict(sorted(blockers.items())),
        "promotion_status": "not_ready",
        "next_step": NEXT_STEP,
        "oracle_policy": "non_oracle_generation",
        "comparison_contract": "metric_dsl_clean_holdout_candidate_slice",
        "reference_sql_visible_to_model_prompt": False,
        "scorer_fields_visible_to_model_prompt": False,
        "input_path": str(input_path),
    }


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


def write_metric_dsl_clean_holdout_candidate_artifacts(
    *,
    input_path: Path,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    manifest_path: Path = DEFAULT_MANIFEST,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write clean-holdout candidate rows, a readiness summary, and a manifest."""

    rows = build_metric_dsl_clean_holdout_candidates(input_path)
    summary = summarize_metric_dsl_clean_holdout_candidates(rows, input_path=input_path)
    _write_jsonl(output_path, rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "candidate_count": len(rows),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.exists() else None,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "oracle_policy": "non_oracle_generation",
        "promotion_status": "not_ready",
        "command": list(command or []),
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = write_metric_dsl_clean_holdout_candidate_artifacts(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(
        f"Wrote {manifest['candidate_count']} Metric DSL clean-holdout "
        f"candidate rows to {args.output}"
    )
    return 0 if manifest["candidate_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
