from __future__ import annotations

import json

import pytest

from data.plan_contract import PREDICTED_PLANNER
from eval.planner_eval import (
    annotate_prepared_records_with_lexical_plans,
    evaluate_planner_records,
    extract_schema_inventory,
    lexical_planner,
    run_planner_eval,
    score_plans,
    summarize_planner_scores,
)


def test_score_plans_separates_table_column_and_shape_scores() -> None:
    gold = {
        "relevant_tables": ["airlines", "routes"],
        "relevant_columns": ["airlines.name", "routes.destination_airport_id"],
        "join_path": ["airlines.airline_id = routes.airline_id"],
        "query_skeleton": {"select": True, "join": True, "where": True},
        "projection_shape": {
            "selected_count": 1,
            "aggregations": [],
            "group_by": [],
            "preserve_duplicates": False,
        },
    }
    predicted = {
        "relevant_tables": ["airlines"],
        "relevant_columns": ["airlines.name"],
        "join_path": [],
        "query_skeleton": {"select": True, "join": False, "where": True},
        "projection_shape": {
            "selected_count": 1,
            "aggregations": [],
            "group_by": [],
            "preserve_duplicates": True,
        },
    }

    scores = score_plans(gold, predicted)

    assert scores["table_f1"] == pytest.approx(2 / 3)
    assert scores["column_f1"] == pytest.approx(2 / 3)
    assert scores["join_f1"] == 0.0
    assert scores["selected_count_match"] == 1.0
    assert scores["duplicate_policy_match"] == 0.0
    assert 0.0 < scores["macro_planner_score"] < 1.0


def test_extract_schema_inventory_reads_compact_and_create_table_schema() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Schema/context:\n"
                "airlines(airline_id int, name text, country text)\n"
                "CREATE TABLE routes (route_id INT, airline_id INT, destination_airport_id INT);\n\n"
                "Question:\nWhich airlines fly to Boston?"
            ),
        }
    ]

    inventory = extract_schema_inventory(messages)

    assert inventory["airlines"] == ["airline_id", "country", "name"]
    assert inventory["routes"] == ["airline_id", "destination_airport_id", "route_id"]


def test_lexical_planner_uses_question_text_not_schema_tokens() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Schema/context:\n"
                "airlines(airline_id int, name text, country text)\n"
                "routes(route_id int, airline_id int, destination_airport_id int)\n\n"
                "Question:\nOnly show airlines from France."
            ),
        }
    ]

    plan = lexical_planner(messages)

    assert plan["prediction_source"] == "lexical_schema_baseline"
    assert plan["relevant_tables"] == ["airlines"]
    assert "airlines.country" in plan["relevant_columns"]
    assert "routes.route_id" not in plan["relevant_columns"]


def test_evaluate_planner_records_adds_gold_predicted_and_scores() -> None:
    records = [
        {
            "id": "turn-1",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Schema/context:\n"
                        "airlines(airline_id int, name text, country text)\n\n"
                        "Question:\nOnly show airlines from France."
                    ),
                }
            ],
            "schema_link_labels": {
                "relevant_tables": ["airlines"],
                "relevant_columns": ["airlines.country"],
                "join_path": [],
                "query_skeleton": {"select": True, "where": True},
                "projection_shape": {
                    "selected_count": 1,
                    "aggregations": [],
                    "group_by": [],
                    "preserve_duplicates": True,
                },
            },
        }
    ]

    evaluated = evaluate_planner_records(records)

    assert evaluated[0]["gold_plan"]["relevant_tables"] == ["airlines"]
    assert evaluated[0]["predicted_plan"]["relevant_tables"] == ["airlines"]
    assert "macro_planner_score" in evaluated[0]["planner_scores"]
    assert summarize_planner_scores(evaluated)["rows"] == 1


def test_run_planner_eval_rejects_oracle_prompt_by_default(tmp_path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {
                        "role": "user",
                        "content": (
                            "Oracle SQL planning hints (derived from reference SQL; diagnostic only):\n"
                            "Relevant tables: airlines\n\n"
                            "Question:\nList airlines."
                        ),
                    },
                    {"role": "assistant", "content": "SELECT name FROM airlines;"},
                ],
                "schema_link_labels": [
                    {
                        "relevant_tables": ["airlines"],
                        "relevant_columns": ["airlines.name"],
                        "join_path": [],
                        "query_skeleton": {"select": True},
                        "projection_shape": {"selected_count": 1, "preserve_duplicates": True},
                    }
                ],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="oracle planning hints"):
        run_planner_eval(
            input_path=input_path,
            output=tmp_path / "planner.jsonl",
            summary_output=tmp_path / "summary.json",
            limit=None,
            allow_oracle_plan=False,
        )


def test_run_planner_eval_writes_rows_and_summary(tmp_path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output = tmp_path / "planner.jsonl"
    summary = tmp_path / "summary.json"
    input_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {
                        "role": "user",
                        "content": (
                            "Schema/context:\n"
                            "airlines(airline_id int, name text, country text)\n\n"
                            "Question:\nList airline names."
                        ),
                    },
                    {"role": "assistant", "content": "SELECT name FROM airlines;"},
                ],
                "schema_link_labels": [
                    {
                        "relevant_tables": ["airlines"],
                        "relevant_columns": ["airlines.name"],
                        "join_path": [],
                        "query_skeleton": {"select": True},
                        "projection_shape": {"selected_count": 1, "preserve_duplicates": True},
                    }
                ],
            }
        )
        + "\n"
    )

    assert (
        run_planner_eval(
            input_path=input_path,
            output=output,
            summary_output=summary,
            limit=None,
            allow_oracle_plan=False,
        )
        == 0
    )

    row = json.loads(output.read_text().splitlines()[0])
    summary_row = json.loads(summary.read_text())
    assert row["planner_scores"]["macro_planner_score"] >= 0.0
    assert summary_row["rows"] == 1


def test_annotate_prepared_records_with_lexical_plans_writes_predicted_mode(tmp_path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "predicted.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {
                        "role": "user",
                        "content": (
                            "Schema/context:\n"
                            "airlines(airline_id int, name text, country text)\n\n"
                            "Question:\nList airline names."
                        ),
                    },
                    {"role": "assistant", "content": "SELECT name FROM airlines;"},
                ],
                "gold_plans": [
                    {
                        "relevant_tables": ["airlines"],
                        "relevant_columns": ["airlines.name"],
                        "join_path": [],
                        "query_skeleton": {"select": True},
                        "projection_shape": {"selected_count": 1, "preserve_duplicates": True},
                    }
                ],
            }
        )
        + "\n"
    )

    assert annotate_prepared_records_with_lexical_plans(input_path, output_path, limit=None) == 1

    row = json.loads(output_path.read_text())
    assert row["evaluation_mode"] == PREDICTED_PLANNER
    assert row["uses_oracle_planning_hints"] is False
    assert row["semantic_context_pruned_by_oracle_labels"] is False
    assert row["predicted_plans"][0]["prediction_source"] == "lexical_schema_baseline"
    assert row["predicted_plans"][0]["relevant_tables"] == ["airlines"]
