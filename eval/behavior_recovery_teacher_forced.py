"""Write teacher-forced behavior-recovery rows for rollout comparison."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from eval.ragas_metrics import score_single_turn
from eval.result_manifest import build_result_manifest, write_result_manifest
from eval.run_eval import summarize_eval_metrics

GOLD_SQL_TEACHER_FORCED = "gold_sql_teacher_forced"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _assistant_indices(messages: list[dict[str, str]]) -> list[int]:
    return [index for index, message in enumerate(messages) if message.get("role") == "assistant"]


def evaluate_teacher_forced_records(
    records: list[dict[str, Any]],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    """Score the final repair label as a teacher-forced diagnostic row."""

    rows = []
    for record in records:
        messages = record["messages"]
        assistant_indices = _assistant_indices(messages)
        if not assistant_indices:
            raise ValueError(f"{record.get('id')} has no assistant reference SQL")
        assistant_index = assistant_indices[-1]
        reference_sql = messages[assistant_index]["content"]
        score = score_single_turn(reference_sql, reference_sql, database_path=None)
        strict_score = (
            score.strict_execution_score
            if score.strict_execution_score is not None
            else score.execution_score
        )
        value_score = (
            score.value_execution_score
            if score.value_execution_score is not None
            else score.execution_score
        )
        dialog_id = str(record.get("dialog_id") or record.get("id"))
        turn_index = len(assistant_indices) - 1
        rows.append(
            {
                "id": f"{dialog_id}:{turn_index}",
                "dialog_id": dialog_id,
                "turn_index": turn_index,
                "turn_count": len(assistant_indices),
                "messages": [dict(message) for message in messages[:assistant_index]],
                "reference_sql": reference_sql,
                "source": record.get("source", "behavior_recovery_rollout_input"),
                "database_id": record.get("database_id"),
                "history_policy": GOLD_SQL_TEACHER_FORCED,
                "original_history_policy": record.get("history_policy"),
                "evaluation_mode": record.get("evaluation_mode") or "non_oracle_generation",
                "uses_oracle_planning_hints": bool(record.get("uses_oracle_planning_hints")),
                "semantic_context_pruned_by_oracle_labels": bool(
                    record.get("semantic_context_pruned_by_oracle_labels")
                ),
                "model_name": model_name,
                "raw_generation": reference_sql,
                "generated_sql": reference_sql,
                "generation_latency_ms": 0.0,
                "execution_score": score.execution_score,
                "strict_execution_score": strict_score,
                "value_execution_score": value_score,
                "order_sensitive": score.order_sensitive,
                "normalized_match": score.normalized_match,
                "syntax_valid": score.syntax_valid,
                "score_error": score.error,
                "database_path": None,
            }
        )
    return rows


def run_behavior_recovery_teacher_forced(
    *,
    input_path: Path,
    output_path: Path,
    manifest_output: Path | None,
    model_name: str,
    command: Sequence[str] | None = None,
) -> int:
    """Write teacher-forced rows and a prepared benchmark manifest."""

    rows = evaluate_teacher_forced_records(_load_jsonl(input_path), model_name=model_name)
    written = _write_jsonl(output_path, rows)
    metrics = summarize_eval_metrics(rows)
    if manifest_output is None:
        manifest_output = output_path.with_suffix(".manifest.json")
    manifest = build_result_manifest(
        run_id=output_path.stem,
        benchmark="prepared",
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        endpoint="offline",
        evaluation_mode="non_oracle_generation",
        oracle_allowed=False,
        prompt_variant=None,
        database_root=None,
        command=list(command or sys.argv),
        row_count=written,
        metrics=metrics,
    )
    write_result_manifest(manifest, manifest_output)
    print(f"Wrote {written} teacher-forced rows to {output_path}")
    print(f"Wrote teacher-forced manifest to {manifest_output}")
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    return run_behavior_recovery_teacher_forced(
        input_path=args.input,
        output_path=args.output,
        manifest_output=args.manifest_output,
        model_name=args.model_name,
        command=sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
