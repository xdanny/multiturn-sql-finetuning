from __future__ import annotations

import pytest

from data.plan_contract import (
    NON_ORACLE_GENERATION,
    ORACLE_PLANNER_DIAGNOSTIC,
    PREDICTED_PLANNER,
    evaluation_mode_from_flags,
    normalize_plan,
    predicted_planning_hint_from_plan,
    validate_predicted_plan_for_prompt,
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


def test_normalize_plan_preserves_projection_expression_order() -> None:
    plan = normalize_plan(
        {
            "relevant_tables": ["Users"],
            "projection_shape": {
                "selected_expressions": ["Users.name", "Users.id"],
                "selected_count": 2,
            },
        }
    )

    assert plan["projection_shape"]["selected_expressions"] == ["users.name", "users.id"]


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


def test_validate_predicted_plan_for_prompt_rejects_empty_plan() -> None:
    with pytest.raises(ValueError, match="relevant table or column"):
        validate_predicted_plan_for_prompt({"query_skeleton": {"select": True}})


def test_validate_predicted_plan_for_prompt_rejects_oracle_markers() -> None:
    with pytest.raises(ValueError, match="oracle"):
        validate_predicted_plan_for_prompt(
            {
                "prediction_source": "gold_reference_sql_planner",
                "relevant_tables": ["airlines"],
                "query_skeleton": {"select": True},
                "projection_shape": {"selected_count": 1},
            }
        )


def test_predicted_planning_hint_validates_plan_before_prompt_injection() -> None:
    hint = predicted_planning_hint_from_plan(
        {
            "prediction_source": "json_planner_predictions",
            "relevant_tables": ["Airlines"],
            "relevant_columns": ["Airlines.Name"],
            "query_skeleton": {"select": True},
            "projection_shape": {
                "selected_count": 1,
                "selected_expressions": ["Airlines.Name"],
                "preserve_duplicates": True,
            },
        }
    )

    assert "Predicted SQL plan" in hint
    assert "Relevant tables: airlines" in hint
    assert "Relevant columns: airlines.name" in hint
    assert "output columns must follow exactly this order: airlines.name" in hint

    with pytest.raises(ValueError, match="relevant table or column"):
        predicted_planning_hint_from_plan({})


def test_predicted_planning_hint_preserves_projection_expression_order() -> None:
    hint = predicted_planning_hint_from_plan(
        {
            "relevant_tables": ["Users"],
            "projection_shape": {
                "selected_count": 2,
                "selected_expressions": ["Users.name", "Users.id"],
            },
        }
    )

    assert "output columns must follow exactly this order: users.name; users.id" in hint


def test_predicted_planning_hint_accepts_legacy_minimal_predicted_plan() -> None:
    hint = predicted_planning_hint_from_plan(
        {
            "relevant_tables": ["Orders"],
            "projection_shape": {"selected_count": 1},
        }
    )

    assert "Relevant tables: orders" in hint
    assert "Query skeleton: select" in hint
