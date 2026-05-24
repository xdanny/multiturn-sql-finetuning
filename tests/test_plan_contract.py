from __future__ import annotations

import pytest

from data.plan_contract import (
    NON_ORACLE_GENERATION,
    ORACLE_PLANNER_DIAGNOSTIC,
    PREDICTED_PLANNER,
    evaluation_mode_from_flags,
    normalize_plan,
    validate_prepared_record_contract,
)


def test_evaluation_mode_from_flags_prioritizes_oracle_over_predicted_plan() -> None:
    assert (
        evaluation_mode_from_flags(
            uses_oracle_planning_hints=True,
            semantic_context_pruned_by_oracle_labels=False,
            has_predicted_plan=True,
        )
        == ORACLE_PLANNER_DIAGNOSTIC
    )
    assert (
        evaluation_mode_from_flags(
            uses_oracle_planning_hints=False,
            semantic_context_pruned_by_oracle_labels=False,
            has_predicted_plan=True,
        )
        == PREDICTED_PLANNER
    )
    assert (
        evaluation_mode_from_flags(
            uses_oracle_planning_hints=False,
            semantic_context_pruned_by_oracle_labels=False,
        )
        == NON_ORACLE_GENERATION
    )


def test_normalize_plan_returns_stable_fields() -> None:
    plan = normalize_plan(
        {
            "relevant_tables": ["Airlines"],
            "relevant_columns": ["Airlines.Name"],
            "query_skeleton": {"select": True, "where": True},
            "projection_shape": {
                "selected_expressions": ["Airlines.Name"],
                "selected_count": 1,
                "preserve_duplicates": False,
            },
        }
    )

    assert plan["relevant_tables"] == ["airlines"]
    assert plan["relevant_columns"] == ["airlines.name"]
    assert set(plan["query_skeleton"]) == {
        "select",
        "join",
        "where",
        "group_by",
        "having",
        "order_by",
        "limit",
        "nested",
        "distinct",
    }
    assert plan["query_skeleton"]["where"] is True
    assert plan["query_skeleton"]["join"] is False
    assert plan["projection_shape"]["selected_expressions"] == ["airlines.name"]
    assert plan["projection_shape"]["preserve_duplicates"] is False


def test_validate_prepared_record_contract_requires_gold_plan_per_assistant_turn() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "SELECT 1;"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "SELECT 2;"},
        ],
        "evaluation_mode": NON_ORACLE_GENERATION,
        "gold_plans": [normalize_plan({})],
    }

    with pytest.raises(ValueError, match="gold_plans length"):
        validate_prepared_record_contract(record)


def test_validate_prepared_record_contract_rejects_predicted_mode_without_predictions() -> None:
    record = {
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "SELECT 1;"},
        ],
        "evaluation_mode": PREDICTED_PLANNER,
        "gold_plans": [normalize_plan({})],
    }

    with pytest.raises(ValueError, match="predicted planner"):
        validate_prepared_record_contract(record)
