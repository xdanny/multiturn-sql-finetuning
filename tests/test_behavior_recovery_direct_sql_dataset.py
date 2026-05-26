from __future__ import annotations

from data.behavior_recovery_direct_sql_dataset import (
    build_behavior_recovery_direct_sql_training_rows,
    summarize_behavior_recovery_direct_sql_training_rows,
)
from data.synthetic_method_fixtures import build_synthetic_method_fixtures


def test_build_behavior_recovery_direct_sql_training_rows_uses_same_recovery_fixture() -> None:
    fixtures = build_synthetic_method_fixtures()

    rows = build_behavior_recovery_direct_sql_training_rows(fixtures)

    assert [row["fixture_id"] for row in rows] == ["recovery_empty_result"]
    assert rows[0]["benchmark"] == "behavior_recovery_direct_sql"
    assert rows[0]["training_target"] == "direct_sql_control"
    assert rows[0]["evaluation_mode"] == "non_oracle_generation"


def test_behavior_recovery_direct_sql_prompt_keeps_recovery_context_without_labels() -> None:
    fixtures = build_synthetic_method_fixtures()

    [row] = build_behavior_recovery_direct_sql_training_rows(fixtures)
    user_prompt = row["messages"][1]["content"]

    assert "Turn 2 previous SQL:" in user_prompt
    assert "Turn 2 observed rows: []" in user_prompt
    assert "Turn 3 observed rows: []" in user_prompt
    assert "requires_generated_history" not in user_prompt
    assert "observed_failure" not in user_prompt
    assert row["reference_sql"] not in user_prompt


def test_summarize_behavior_recovery_direct_sql_training_rows_counts_recovery_fixture() -> None:
    fixtures = build_synthetic_method_fixtures()
    rows = build_behavior_recovery_direct_sql_training_rows(fixtures)

    summary = summarize_behavior_recovery_direct_sql_training_rows(rows)

    assert summary == {
        "artifact_type": "behavior_recovery_direct_sql_training_rows",
        "benchmark": "behavior_recovery_direct_sql",
        "training_target": "direct_sql_control",
        "row_count": 1,
        "fixture_ids": ["recovery_empty_result"],
        "failure_mode_counts": {
            "entity_resolution": 1,
            "recovery": 1,
            "value_normalization": 1,
        },
    }
