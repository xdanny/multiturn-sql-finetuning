from __future__ import annotations

from data.behavior_recovery_dataset import (
    build_behavior_recovery_training_rows,
    summarize_behavior_recovery_training_rows,
)
from data.synthetic_method_fixtures import build_synthetic_method_fixtures


def test_build_behavior_recovery_training_rows_uses_only_recovery_fixture() -> None:
    fixtures = build_synthetic_method_fixtures()

    rows = build_behavior_recovery_training_rows(fixtures)

    assert [row["fixture_id"] for row in rows] == ["recovery_empty_result"]
    assert rows[0]["benchmark"] == "synthetic_behavior_recovery"
    assert rows[0]["training_target"] == "behavior_recovery"
    assert rows[0]["evaluation_mode"] == "non_oracle_generation"


def test_behavior_recovery_prompt_keeps_recovery_state_without_label_leakage() -> None:
    fixtures = build_synthetic_method_fixtures()

    [row] = build_behavior_recovery_training_rows(fixtures)
    user_prompt = row["messages"][1]["content"]

    assert "That returned no rows. Repair it and show the top customer there." in user_prompt
    assert "previous SQL" in user_prompt
    assert "observed rows" in user_prompt
    assert "replace_display_value_with_storage_value" not in user_prompt
    assert "requires_repair_action" not in user_prompt
    assert row["reference_sql"] not in user_prompt


def test_summarize_behavior_recovery_training_rows_counts_recovery_fixture() -> None:
    fixtures = build_synthetic_method_fixtures()
    rows = build_behavior_recovery_training_rows(fixtures)

    summary = summarize_behavior_recovery_training_rows(rows)

    assert summary == {
        "artifact_type": "behavior_recovery_training_rows",
        "benchmark": "synthetic_behavior_recovery",
        "training_target": "behavior_recovery",
        "row_count": 1,
        "fixture_ids": ["recovery_empty_result"],
        "failure_mode_counts": {
            "entity_resolution": 1,
            "recovery": 1,
            "value_normalization": 1,
        },
    }
