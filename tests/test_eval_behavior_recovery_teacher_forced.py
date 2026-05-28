from __future__ import annotations

import json
from pathlib import Path

from eval.behavior_recovery_teacher_forced import (
    GOLD_SQL_TEACHER_FORCED,
    evaluate_teacher_forced_records,
    run_behavior_recovery_teacher_forced,
)


def _rollout_input() -> dict:
    return {
        "id": "recovery_empty_result",
        "dialog_id": "recovery_empty_result",
        "database_id": "synthetic_revenue_schema_v1",
        "source": "synthetic_behavior_recovery",
        "history_policy": "seeded_generated_failure_then_rollout",
        "evaluation_mode": "non_oracle_generation",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "SELECT bad;"},
            {"role": "user", "content": "repair"},
            {"role": "assistant", "content": "SELECT fixed;"},
        ],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_evaluate_teacher_forced_records_scores_final_repair_turn() -> None:
    [row] = evaluate_teacher_forced_records([_rollout_input()], model_name="local-9b")

    assert row["id"] == "recovery_empty_result:1"
    assert row["dialog_id"] == "recovery_empty_result"
    assert row["turn_index"] == 1
    assert row["history_policy"] == GOLD_SQL_TEACHER_FORCED
    assert row["original_history_policy"] == "seeded_generated_failure_then_rollout"
    assert row["generated_sql"] == "SELECT fixed;"
    assert row["reference_sql"] == "SELECT fixed;"
    assert row["messages"][2] == {"role": "assistant", "content": "SELECT bad;"}
    assert row["value_execution_score"] == 1.0
    assert row["strict_execution_score"] == 1.0


def test_run_behavior_recovery_teacher_forced_writes_prepared_manifest(tmp_path) -> None:
    input_path = tmp_path / "rollout_inputs.jsonl"
    output_path = tmp_path / "teacher.jsonl"
    manifest_path = tmp_path / "teacher.manifest.json"
    _write_jsonl(input_path, [_rollout_input()])

    exit_code = run_behavior_recovery_teacher_forced(
        input_path=input_path,
        output_path=output_path,
        manifest_output=manifest_path,
        model_name="local-9b",
        command=["python", "-m", "eval.behavior_recovery_teacher_forced"],
    )

    assert exit_code == 0
    manifest = json.loads(manifest_path.read_text())
    assert manifest["benchmark"] == "prepared"
    assert manifest["evaluation_mode"] == "non_oracle_generation"
    assert manifest["oracle_allowed"] is False
    assert manifest["row_count"] == 1
    assert manifest["metrics"]["history_policy"] == GOLD_SQL_TEACHER_FORCED
    assert manifest["metrics"]["value_execution_accuracy"] == 1.0
