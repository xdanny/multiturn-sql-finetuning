from __future__ import annotations

import csv
import json

import pytest

from eval.planner_optimize import (
    DEFAULT_PLANNER_PROMPT_VARIANTS,
    PlannerPromptVariant,
    apply_planner_prompt_variant,
    evaluate_planner_variant,
    load_planner_prompt_variants,
    planner_summary_sort_key,
    score_planner_prediction,
    summarize_planner_variant,
    write_planner_summary,
)


def _record() -> dict:
    return {
        "id": "dialog-a:0",
        "dialog_id": "dialog-a",
        "turn_index": 0,
        "database_id": "store",
        "messages": [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": (
                    "Schema/context:\n"
                    "customers(id int, name text)\n"
                    "orders(id int, customer_id int, amount real)\n\n"
                    "Question:\nShow customer names."
                ),
            },
        ],
        "reference_sql": "SELECT customers.name FROM customers;",
        "gold_plan": {
            "relevant_tables": ["customers"],
            "relevant_columns": ["customers.name"],
            "join_path": [],
            "query_skeleton": {"select": True},
            "projection_shape": {
                "selected_count": 1,
                "selected_expressions": ["customers.name"],
                "preserve_duplicates": True,
            },
        },
    }


def test_apply_planner_prompt_variant_appends_policy_without_mutating() -> None:
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    variant = PlannerPromptVariant("projection", "Predict selected_count exactly.")

    updated = apply_planner_prompt_variant(messages, variant)

    assert "Planner prompt policy (projection)" in updated[0]["content"]
    assert "Predict selected_count exactly." in updated[0]["content"]
    assert messages[0]["content"] == "sys"


def test_default_projection_variant_preserves_answer_column_sequence() -> None:
    projection_variant = next(
        variant for variant in DEFAULT_PLANNER_PROMPT_VARIANTS if variant.name == "projection_shape"
    )

    assert "user's requested answer-column order" in projection_variant.instruction
    assert "without alphabetizing" in projection_variant.instruction
    assert "SQL alias names" in projection_variant.instruction


def test_load_planner_prompt_variants_from_json(tmp_path) -> None:
    path = tmp_path / "planner_variants.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "schema_link",
                    "instruction": "Prefer exact table and column names.",
                    "source": "unit",
                }
            ]
        )
    )

    assert load_planner_prompt_variants(path) == [
        PlannerPromptVariant("schema_link", "Prefer exact table and column names.", "unit")
    ]


def test_evaluate_planner_variant_scores_predictions_without_reference_in_prompt() -> None:
    prompts: list[list[dict[str, str]]] = []

    def fake_generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        prompts.append(messages)
        return (
            json.dumps(
                {
                    "relevant_tables": ["customers"],
                    "relevant_columns": ["customers.name"],
                    "query_skeleton": {"select": True},
                    "projection_shape": {
                        "selected_count": 1,
                        "selected_expressions": ["customers.name"],
                        "preserve_duplicates": True,
                    },
                }
            ),
            7.0,
        )

    rows = evaluate_planner_variant(
        records=[_record()],
        variant=PlannerPromptVariant("schema", "Prefer visible schema objects."),
        generate_fn=fake_generate,
        model_name="planner-9b",
    )

    assert "SELECT customers.name" not in json.dumps(prompts[0])
    assert "Planner prompt policy (schema)" in json.dumps(prompts[0])
    assert rows[0]["prompt_variant"] == "schema"
    assert rows[0]["prompt_variant_source"] == "static"
    assert rows[0]["planner_latency_ms"] == 7.0
    assert rows[0]["predicted_plan"]["relevant_tables"] == ["customers"]
    assert rows[0]["planner_scores"]["macro_planner_score"] == pytest.approx(1.0)


def test_summarize_planner_variant_reports_macro_and_parse_rate() -> None:
    rows = [
        {
            "planner_scores": {
                "table_f1": 1.0,
                "column_f1": 1.0,
                "join_f1": 1.0,
                "skeleton_f1": 1.0,
                "aggregation_f1": 1.0,
                "group_by_f1": 1.0,
                "selected_count_match": 1.0,
                "selected_expression_order_match": 1.0,
                "duplicate_policy_match": 1.0,
                "macro_planner_score": 1.0,
            },
            "predicted_plan": {"parseable": True},
            "planner_latency_ms": 10.0,
        },
        {
            "planner_scores": {
                "table_f1": 0.0,
                "column_f1": 0.0,
                "join_f1": 1.0,
                "skeleton_f1": 0.5,
                "aggregation_f1": 1.0,
                "group_by_f1": 1.0,
                "selected_count_match": 0.0,
                "selected_expression_order_match": 0.0,
                "duplicate_policy_match": 1.0,
                "macro_planner_score": 0.5,
            },
            "predicted_plan": {"parseable": False},
            "planner_latency_ms": 30.0,
        },
    ]

    summary = summarize_planner_variant(rows)

    assert summary["samples"] == 2
    assert summary["parse_rate"] == 0.5
    assert summary["mean_latency_ms"] == 20.0
    assert summary["macro_planner_score"] == pytest.approx(0.75)
    assert summary["table_f1"] == pytest.approx(0.5)


def test_malformed_planner_output_gets_zero_score_and_cannot_win_ranking() -> None:
    gold = {
        "relevant_tables": [],
        "relevant_columns": [],
        "join_path": [],
        "query_skeleton": {},
        "projection_shape": {"selected_count": None, "preserve_duplicates": False},
    }
    malformed_scores = score_planner_prediction(gold, {"parseable": False})

    assert malformed_scores == {
        "table_f1": 0.0,
        "column_f1": 0.0,
        "join_f1": 0.0,
        "skeleton_f1": 0.0,
        "aggregation_f1": 0.0,
        "group_by_f1": 0.0,
        "selected_count_match": 0.0,
        "selected_expression_order_match": 0.0,
        "output_slot_order_match": 0.0,
        "duplicate_policy_match": 0.0,
        "macro_planner_score": 0.0,
    }

    ranked = sorted(
        [
            {
                "prompt_variant": "malformed",
                "parse_rate": 0.0,
                "macro_planner_score": 1.0,
                "column_f1": 1.0,
                "table_f1": 1.0,
            },
            {
                "prompt_variant": "parseable",
                "parse_rate": 1.0,
                "macro_planner_score": 0.5,
                "column_f1": 0.3,
                "table_f1": 0.3,
            },
        ],
        key=planner_summary_sort_key,
        reverse=True,
    )

    assert ranked[0]["prompt_variant"] == "parseable"


def test_write_planner_summary_writes_csv(tmp_path) -> None:
    output = tmp_path / "summary.csv"
    write_planner_summary(
        [
            {
                "prompt_variant": "schema",
                "prompt_variant_source": "static",
                "samples": 2,
                "parse_rate": 1.0,
                "mean_latency_ms": 8.0,
                "macro_planner_score": 0.75,
                "table_f1": 1.0,
                "column_f1": 0.5,
                "join_f1": 1.0,
                "skeleton_f1": 0.5,
                "aggregation_f1": 1.0,
                "group_by_f1": 1.0,
                "selected_count_match": 0.0,
                "selected_expression_order_match": 0.0,
                "duplicate_policy_match": 1.0,
            }
        ],
        output,
    )

    rows = list(csv.DictReader(output.open()))
    assert rows[0]["prompt_variant"] == "schema"
    assert rows[0]["macro_planner_score"] == "0.75"
