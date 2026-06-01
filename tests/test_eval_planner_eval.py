from __future__ import annotations

import json

import pytest

from data.plan_contract import PREDICTED_PLANNER
from eval.planner_eval import (
    annotate_prepared_records_with_lexical_plans,
    annotate_prepared_records_with_plans,
    evaluate_planner_records,
    extract_schema_inventory,
    lexical_planner,
    parse_json_plan_prediction,
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
            "selected_expressions": ["airlines.name"],
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
            "selected_expressions": ["airlines.name"],
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
    assert scores["selected_expression_order_match"] == 1.0
    assert scores["duplicate_policy_match"] == 0.0
    assert 0.0 < scores["macro_planner_score"] < 1.0


def test_score_plans_tracks_projection_order_separately_from_count() -> None:
    gold = {
        "projection_shape": {
            "selected_count": 2,
            "selected_expressions": ["stadium.name", "stadium.location"],
        },
    }
    predicted = {
        "projection_shape": {
            "selected_count": 2,
            "selected_expressions": ["stadium.location", "stadium.name"],
        },
    }

    scores = score_plans(gold, predicted)

    assert scores["selected_count_match"] == 1.0
    assert scores["selected_expression_order_match"] == 0.0
    assert scores["macro_planner_score"] < 1.0


def test_score_plans_compares_projection_order_without_sql_aliases() -> None:
    gold = {
        "projection_shape": {
            "selected_count": 2,
            "selected_expressions": ["t2.name", "t2.location"],
        },
    }
    predicted = {
        "projection_shape": {
            "selected_count": 2,
            "selected_expressions": ["stadium.name", "stadium.location"],
        },
    }

    scores = score_plans(gold, predicted)

    assert scores["selected_count_match"] == 1.0
    assert scores["selected_expression_order_match"] == 1.0


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


def test_lexical_planner_matches_simple_plural_question_tokens() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Schema/context:\n"
                "teacher(Teacher_ID number, Name text, Age text)\n\n"
                "Question:\nHow many teachers are there?"
            ),
        }
    ]

    plan = lexical_planner(messages)

    assert plan["relevant_tables"] == ["teacher"]
    assert "count(*)" in plan["projection_shape"]["aggregations"]


def test_lexical_planner_splits_camel_case_column_tokens() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Schema/context:\n"
                "countries(CountryId number, CountryName text)\n"
                "car_makers(Id number, FullName text, Country text)\n\n"
                "Question:\nList the country name and maker full name."
            ),
        }
    ]

    plan = lexical_planner(messages)

    assert "countries.countryname" in plan["relevant_columns"]
    assert "car_makers.fullname" in plan["relevant_columns"]
    assert "countries.countryid" not in plan["relevant_columns"]


def test_lexical_planner_generic_name_column_does_not_pull_unmentioned_table() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Schema/context:\n"
                "airlines(airline_id int, name text, country text)\n"
                "routes(route_id int, airline_id int, destination_airport_id int)\n"
                "airports(airport_id int, name text, city text)\n\n"
                "Question:\nList airline names."
            ),
        }
    ]

    plan = lexical_planner(messages)

    assert plan["relevant_tables"] == ["airlines"]
    assert "airlines.name" in plan["relevant_columns"]
    assert "airports.name" not in plan["relevant_columns"]
    assert plan["query_skeleton"]["join"] is False


def test_lexical_planner_selected_count_uses_projection_prior_not_column_hits() -> None:
    messages = [
        {
            "role": "user",
            "content": (
                "Schema/context:\n"
                "orders(id number, customer_id number, amount number)\n"
                "customers(id number, name text, country text)\n\n"
                "Question:\nShow customers from France with total order amount."
            ),
        }
    ]

    plan = lexical_planner(messages)

    assert len(plan["relevant_columns"]) > 1
    assert plan["projection_shape"]["selected_count"] == 1
    assert plan["projection_shape"]["selected_expressions"]


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


def test_evaluate_planner_records_prefers_current_schema_labels_over_stale_gold_plan() -> None:
    records = [
        {
            "id": "turn-1",
            "messages": [{"role": "user", "content": "Question:\nShow name and location."}],
            "gold_plan": {
                "projection_shape": {
                    "selected_expressions": ["location", "name"],
                    "selected_count": 2,
                }
            },
            "schema_link_labels": {
                "projection_shape": {
                    "selected_expressions": ["name", "location"],
                    "selected_count": 2,
                }
            },
            "predicted_plan": {
                "projection_shape": {
                    "selected_expressions": ["name", "location"],
                    "selected_count": 2,
                }
            },
        }
    ]

    evaluated = evaluate_planner_records(records)

    assert evaluated[0]["gold_plan"]["projection_shape"]["selected_expressions"] == [
        "name",
        "location",
    ]
    assert evaluated[0]["planner_scores"]["selected_expression_order_match"] == 1.0


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


def test_annotate_prepared_records_with_plans_falls_back_for_unlabeled_turns(tmp_path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "predicted.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Schema/context:\nitems(id int, name text)\n\nQuestion:\nShow items.",
                    },
                    {"role": "assistant", "content": "SELECT first;"},
                    {"role": "user", "content": "Question:\nShow item names."},
                    {"role": "assistant", "content": "SELECT second;"},
                ],
                "schema_link_labels": [
                    {
                        "projection_shape": {
                            "selected_expressions": ["first_schema_label"],
                            "selected_count": 1,
                        }
                    }
                ],
                "gold_plans": [
                    {
                        "projection_shape": {
                            "selected_expressions": ["first_gold_plan"],
                            "selected_count": 1,
                        }
                    },
                    {
                        "projection_shape": {
                            "selected_expressions": ["second_gold_plan"],
                            "selected_count": 1,
                        }
                    },
                ],
            }
        )
        + "\n"
    )

    assert annotate_prepared_records_with_lexical_plans(input_path, output_path, limit=None) == 1

    row = json.loads(output_path.read_text())
    assert row["gold_plans"][0]["projection_shape"]["selected_expressions"] == [
        "first_schema_label",
    ]
    assert row["gold_plans"][1]["projection_shape"]["selected_expressions"] == [
        "second_gold_plan",
    ]


def test_parse_json_plan_prediction_extracts_normalized_plan() -> None:
    plan = parse_json_plan_prediction(
        """```json
        {
          "relevant_tables": ["Airlines"],
          "relevant_columns": ["Airlines.Name"],
          "query_skeleton": {"select": true},
          "projection_shape": {
            "selected_count": 1,
            "selected_expressions": ["Airlines.Name"],
            "preserve_duplicates": true
          }
        }
        ```""",
        prediction_source="json_planner_predictions",
    )

    assert plan["parseable"] is True
    assert plan["prediction_source"] == "json_planner_predictions"
    assert plan["relevant_tables"] == ["airlines"]
    assert plan["relevant_columns"] == ["airlines.name"]
    assert plan["projection_shape"]["selected_count"] == 1


def test_parse_json_plan_prediction_records_malformed_output() -> None:
    plan = parse_json_plan_prediction(
        "not json at all",
        prediction_source="json_planner_predictions",
    )

    assert plan["parseable"] is False
    assert plan["prediction_source"] == "json_planner_predictions"
    assert "no JSON object" in plan["planner_parse_error"]


def test_parse_json_plan_prediction_preserves_oracle_markers_for_validation(tmp_path) -> None:
    plan = parse_json_plan_prediction(
        json.dumps(
            {
                "relevant_tables": ["airlines"],
                "query_skeleton": {"select": True},
                "projection_shape": {"selected_count": 1},
                "notes": "derived from reference sql",
            }
        ),
        prediction_source="json_planner_predictions",
    )

    assert "derived from reference sql" in plan["notes"]
    input_path = tmp_path / "prepared.jsonl"
    predictions_path = tmp_path / "planner_predictions.jsonl"
    output_path = tmp_path / "predicted.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "dialog_id": "dialog-a",
                "messages": [
                    {"role": "user", "content": "Question:\nList airline names."},
                    {"role": "assistant", "content": "SELECT name FROM airlines;"},
                ],
                "gold_plans": [{"relevant_tables": ["airlines"]}],
            }
        )
        + "\n"
    )
    predictions_path.write_text(
        json.dumps(
            {
                "id": "dialog-a:0",
                "raw_planner_output": json.dumps(
                    {
                        "relevant_tables": ["airlines"],
                        "query_skeleton": {"select": True},
                        "projection_shape": {"selected_count": 1},
                        "notes": "derived from reference sql",
                    }
                ),
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="oracle"):
        annotate_prepared_records_with_plans(
            input_path,
            output_path,
            limit=None,
            planner_source="json_planner_predictions",
            planner_predictions_path=predictions_path,
        )


@pytest.mark.parametrize(
    "prediction_row",
    [
        {
            "id": "dialog-a:0",
            "predicted_plan": {
                "relevant_tables": ["airlines"],
                "query_skeleton": {"select": True},
                "projection_shape": {"selected_count": 1},
                "uses_oracle_planning_hints": True,
            },
        },
        {
            "id": "dialog-a:0",
            "relevant_tables": ["airlines"],
            "query_skeleton": {"select": True},
            "projection_shape": {"selected_count": 1},
            "notes": "gold_reference_sql",
        },
        {
            "id": "dialog-a:0",
            "uses_oracle_planning_hints": True,
            "raw_planner_output": json.dumps(
                {
                    "relevant_tables": ["airlines"],
                    "query_skeleton": {"select": True},
                    "projection_shape": {"selected_count": 1},
                }
            ),
        },
    ],
)
def test_json_plan_prediction_loader_rejects_oracle_markers_before_stripping(
    tmp_path,
    prediction_row,
) -> None:
    input_path = tmp_path / "prepared.jsonl"
    predictions_path = tmp_path / "planner_predictions.jsonl"
    output_path = tmp_path / "predicted.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "dialog_id": "dialog-a",
                "messages": [
                    {"role": "user", "content": "Question:\nList airline names."},
                    {"role": "assistant", "content": "SELECT name FROM airlines;"},
                ],
                "gold_plans": [{"relevant_tables": ["airlines"]}],
            }
        )
        + "\n"
    )
    predictions_path.write_text(json.dumps(prediction_row) + "\n")

    with pytest.raises(ValueError, match="oracle"):
        annotate_prepared_records_with_plans(
            input_path,
            output_path,
            limit=None,
            planner_source="json_planner_predictions",
            planner_predictions_path=predictions_path,
        )


def test_annotate_prepared_records_with_json_plan_predictions(tmp_path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    predictions_path = tmp_path / "planner_predictions.jsonl"
    output_path = tmp_path / "predicted.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "dialog_id": "dialog-a",
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
    predictions_path.write_text(
        json.dumps(
            {
                "id": "dialog-a:0",
                "raw_planner_output": json.dumps(
                    {
                        "relevant_tables": ["airlines"],
                        "relevant_columns": ["airlines.name"],
                        "query_skeleton": {"select": True},
                        "projection_shape": {
                            "selected_count": 1,
                            "selected_expressions": ["airlines.name"],
                            "preserve_duplicates": True,
                        },
                    }
                ),
            }
        )
        + "\n"
    )

    assert (
        annotate_prepared_records_with_plans(
            input_path,
            output_path,
            limit=None,
            planner_source="json_planner_predictions",
            planner_predictions_path=predictions_path,
        )
        == 1
    )

    row = json.loads(output_path.read_text())
    assert row["evaluation_mode"] == PREDICTED_PLANNER
    assert row["predicted_plan_source"] == "json_planner_predictions"
    assert row["predicted_plans"][0]["prediction_source"] == "json_planner_predictions"
    assert row["predicted_plans"][0]["relevant_tables"] == ["airlines"]
