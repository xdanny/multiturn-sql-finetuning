"""Package the fixed prepared semantic slice as a Stage 5 behavior-recovery artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

ARTIFACT_TYPE = "behavior_recovery_proxy_dataset"
BENCHMARK = "prepared"
DEFAULT_TRAIN_INPUT = Path("data/processed/train_64_each_semantic.jsonl")
DEFAULT_EVAL_INPUT = Path("data/processed/eval_cosql_dev_100.jsonl")
DEFAULT_TRAIN_OUTPUT = Path("docs/data_artifacts/behavior_recovery_proxy_train.jsonl")
DEFAULT_EVAL_OUTPUT = Path("docs/data_artifacts/behavior_recovery_proxy_eval.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/behavior_recovery_proxy_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/behavior_recovery_proxy.manifest.json")
FORBIDDEN_PROMPT_MARKERS = ("reference_sql", "expected_rows", "gold_metric_dsl", "resolved_value")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _assistant_turn_count(row: dict[str, Any]) -> int:
    if row.get("assistant_turn_count") is not None:
        return int(row["assistant_turn_count"])
    return sum(1 for message in row.get("messages", []) if message.get("role") == "assistant")


def _validate_non_oracle_messages(messages: list[dict[str, Any]]) -> None:
    user_text = "\n".join(
        str(message.get("content", "")) for message in messages if message.get("role") == "user"
    )
    if "Semantic model:" not in user_text:
        raise ValueError("behavior recovery proxy rows must include semantic model context")
    for marker in FORBIDDEN_PROMPT_MARKERS:
        if marker in user_text:
            raise ValueError(f"behavior recovery proxy prompt leaked forbidden marker: {marker}")


def build_behavior_recovery_proxy_train_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packaged: list[dict[str, Any]] = []
    for row in rows:
        try:
            _validate_non_oracle_messages(row["messages"])
        except ValueError:
            continue
        if _assistant_turn_count(row) < 2:
            continue
        packaged.append(
            {
                **row,
                "artifact_type": ARTIFACT_TYPE,
                "benchmark": BENCHMARK,
                "training_target": "behavior_recovery",
                "evaluation_mode": row.get("evaluation_mode", "non_oracle_generation"),
                "oracle_policy": "non_oracle_inputs_only",
            }
        )
    return packaged


def build_behavior_recovery_proxy_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packaged: list[dict[str, Any]] = []
    for row in rows:
        _validate_non_oracle_messages(row["messages"])
        if row.get("uses_oracle_planning_hints") or row.get("semantic_context_pruned_by_oracle_labels"):
            raise ValueError("behavior recovery proxy eval rows must be non-oracle")
        if _assistant_turn_count(row) < 2:
            continue
        packaged.append(
            {
                **row,
                "artifact_type": ARTIFACT_TYPE,
                "benchmark": BENCHMARK,
                "training_target": "behavior_recovery",
                "oracle_policy": "non_oracle_inputs_only",
            }
        )
    return packaged


def summarize_behavior_recovery_proxy_artifacts(
    *, train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "artifact_type": ARTIFACT_TYPE,
        "training_target": "behavior_recovery",
        "benchmark": BENCHMARK,
        "train_row_count": len(train_rows),
        "eval_row_count": len(eval_rows),
        "eval_dialog_count": len({str(row.get("dialog_id") or row.get("database_id")) for row in eval_rows}),
        "eval_multi_turn_count": sum(1 for row in eval_rows if _assistant_turn_count(row) >= 2),
        "eval_database_count": len({str(row.get("database_id")) for row in eval_rows}),
    }


def write_behavior_recovery_proxy_artifacts(
    *,
    train_input_path: Path = DEFAULT_TRAIN_INPUT,
    eval_input_path: Path = DEFAULT_EVAL_INPUT,
    train_output_path: Path = DEFAULT_TRAIN_OUTPUT,
    eval_output_path: Path = DEFAULT_EVAL_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    train_rows = build_behavior_recovery_proxy_train_rows(_load_jsonl(train_input_path))
    eval_rows = build_behavior_recovery_proxy_eval_rows(_load_jsonl(eval_input_path))
    summary = summarize_behavior_recovery_proxy_artifacts(train_rows=train_rows, eval_rows=eval_rows)
    _write_jsonl(train_output_path, train_rows)
    _write_jsonl(eval_output_path, eval_rows)
    _write_json(summary_path, summary)
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "training_target": "behavior_recovery",
        "benchmark": BENCHMARK,
        "train_input_path": str(train_input_path),
        "train_input_sha256": sha256_file(train_input_path),
        "eval_input_path": str(eval_input_path),
        "eval_input_sha256": sha256_file(eval_input_path),
        "train_output_path": str(train_output_path),
        "train_output_sha256": sha256_file(train_output_path),
        "eval_output_path": str(eval_output_path),
        "eval_output_sha256": sha256_file(eval_output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--eval-input", type=Path, default=DEFAULT_EVAL_INPUT)
    parser.add_argument("--train-output", type=Path, default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--eval-output", type=Path, default=DEFAULT_EVAL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    manifest = write_behavior_recovery_proxy_artifacts(
        train_input_path=args.train_input,
        eval_input_path=args.eval_input,
        train_output_path=args.train_output,
        eval_output_path=args.eval_output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(f"Wrote behavior-recovery proxy train rows to {args.train_output}")
    print(f"Wrote behavior-recovery proxy eval rows to {args.eval_output}")
    print(f"Wrote manifest to {args.manifest_output}")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
