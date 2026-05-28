from __future__ import annotations

import json
from pathlib import Path

from eval.run_behavior_recovery_comparison import run_behavior_recovery_comparison


def _rollout_input() -> dict:
    return {
        "id": "recovery_empty_result",
        "dialog_id": "recovery_empty_result",
        "database_id": "synthetic_revenue_schema_v1",
        "source": "synthetic_behavior_recovery",
        "history_policy": "seeded_generated_failure_then_rollout",
        "evaluation_mode": "non_oracle_generation",
        "seeded_failure_turn_index": 0,
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


def test_run_behavior_recovery_comparison_writes_rollout_teacher_and_comparison(
    tmp_path,
) -> None:
    input_path = tmp_path / "rollout_inputs.jsonl"
    output_dir = tmp_path / "results" / "rollout"
    _write_jsonl(input_path, [_rollout_input()])

    compared = run_behavior_recovery_comparison(
        input_path=input_path,
        output_dir=output_dir,
        run_id="recovery-smoke",
        model_name="local-9b",
        generate_fn=lambda _messages: ("SELECT fixed;", 5.0),
        command=["python", "-m", "eval.run_behavior_recovery_comparison"],
    )

    rollout_output = output_dir / "recovery-smoke.rollout.jsonl"
    teacher_output = output_dir / "recovery-smoke.teacher_forced.jsonl"
    comparison_output = output_dir / "recovery-smoke.comparison.manifest.json"

    assert rollout_output.exists()
    assert teacher_output.exists()
    assert comparison_output.exists()
    assert compared["benchmark"] == "prepared_rollout"
    assert compared["metrics"]["teacher_forced_comparison_run_id"] == (
        "recovery-smoke.teacher_forced"
    )
    assert compared["metrics"]["rollout_value_delta_vs_teacher_forced"] == 0.0
    assert compared["metrics"]["teacher_forced_comparable_row_count"] == 1
