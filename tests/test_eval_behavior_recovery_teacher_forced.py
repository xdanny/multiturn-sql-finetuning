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


def test_evaluate_teacher_forced_records_scores_all_assistant_turns() -> None:
    rows = evaluate_teacher_forced_records([_rollout_input()], model_name="local-9b")

    assert [row["id"] for row in rows] == [
        "recovery_empty_result:0",
        "recovery_empty_result:1",
    ]
    assert [row["turn_index"] for row in rows] == [0, 1]
    assert {row["history_policy"] for row in rows} == {GOLD_SQL_TEACHER_FORCED}
    assert rows[1]["dialog_id"] == "recovery_empty_result"
    assert rows[1]["original_history_policy"] == "seeded_generated_failure_then_rollout"
    assert rows[1]["generated_sql"] == "SELECT fixed;"
    assert rows[1]["reference_sql"] == "SELECT fixed;"
    assert rows[1]["messages"][2] == {"role": "assistant", "content": "SELECT bad;"}
    assert rows[1]["value_execution_score"] == 1.0
    assert rows[1]["strict_execution_score"] == 1.0


def test_evaluate_teacher_forced_records_skips_seeded_failure_turn() -> None:
    record = _rollout_input()
    record["seeded_failure_turn_index"] = 0

    [row] = evaluate_teacher_forced_records([record], model_name="local-9b")

    assert row["id"] == "recovery_empty_result:1"
    assert row["turn_index"] == 1
    assert row["generated_sql"] == "SELECT fixed;"


def test_evaluate_teacher_forced_records_matches_rollout_dialog_id_fallback() -> None:
    record = _rollout_input()
    record.pop("id")
    record.pop("dialog_id")

    rows = evaluate_teacher_forced_records([record], model_name="local-9b")

    assert [row["dialog_id"] for row in rows] == ["prepared-0", "prepared-0"]
    assert [row["id"] for row in rows] == ["prepared-0:0", "prepared-0:1"]


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
    assert manifest["row_count"] == 2
    assert manifest["metrics"]["history_policy"] == GOLD_SQL_TEACHER_FORCED
    assert manifest["metrics"]["value_execution_accuracy"] == 1.0
