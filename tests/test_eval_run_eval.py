from __future__ import annotations

import json

import pytest

from eval.run_eval import (
    assistant_turn_indices,
    database_path_for_record,
    enforce_sql_only_instruction,
    expand_prepared_record,
    extract_reference_sql,
    generate_sql,
    load_prepared_records,
    write_results,
)


def test_extract_reference_sql_uses_last_assistant_message() -> None:
    assert (
        extract_reference_sql(
            [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "SELECT 1;"},
                {"role": "user", "content": "again"},
                {"role": "assistant", "content": "SELECT 2;"},
            ]
        )
        == "SELECT 2;"
    )


def test_load_prepared_records_hides_reference_assistant_turn(tmp_path) -> None:
    path = tmp_path / "prepared.jsonl"
    path.write_text(
        json.dumps(
            {
                "source": "unit",
                "database_id": "db1",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "SELECT 1;"},
                ],
            }
        )
        + "\n"
    )

    records = load_prepared_records(path)

    assert records[0]["reference_sql"] == "SELECT 1;"
    assert records[0]["database_id"] == "db1"
    assert records[0]["messages"][-1]["role"] == "user"
    assert records[0]["turn_index"] == 0
    assert records[0]["turn_count"] == 1


def test_load_prepared_records_expands_multi_turn_dialogs(tmp_path) -> None:
    path = tmp_path / "prepared.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "dialog-a",
                "source": "unit",
                "database_id": "db1",
                "evaluation_mode": "oracle_planner_diagnostic",
                "planning_label_source": "gold_reference_sql",
                "uses_oracle_planning_hints": True,
                "semantic_context_pruned_by_oracle_labels": True,
                "oracle_diagnostic_warning": "oracle warning",
                "gold_plans": [
                    {"relevant_tables": ["one"], "projection_shape": {"selected_count": 1}},
                    {"relevant_tables": ["two"], "projection_shape": {"selected_count": 1}},
                ],
                "predicted_plans": [
                    {"relevant_tables": ["one"]},
                    {"relevant_tables": ["wrong"]},
                ],
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "SELECT 1;"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "SELECT 2;"},
                ],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="oracle planning hints"):
        load_prepared_records(path)

    records = load_prepared_records(path, allow_oracle_plan=True)

    assert assistant_turn_indices(json.loads(path.read_text())["messages"]) == [2, 4]
    assert [record["reference_sql"] for record in records] == ["SELECT 1;", "SELECT 2;"]
    assert [record["turn_index"] for record in records] == [0, 1]
    assert [record["turn_count"] for record in records] == [2, 2]
    assert records[0]["messages"][-1]["content"] == "q1"
    assert records[1]["messages"][-1]["content"] == "q2"
    assert records[1]["messages"][2]["content"] == "SELECT 1;"
    assert records[0]["evaluation_mode"] == "oracle_planner_diagnostic"
    assert records[0]["planning_label_source"] == "gold_reference_sql"
    assert records[0]["uses_oracle_planning_hints"] is True
    assert records[0]["semantic_context_pruned_by_oracle_labels"] is True
    assert records[0]["oracle_diagnostic_warning"] == "oracle warning"
    assert records[0]["gold_plan"]["relevant_tables"] == ["one"]
    assert records[1]["gold_plan"]["relevant_tables"] == ["two"]
    assert records[0]["predicted_plan"]["relevant_tables"] == ["one"]
    assert records[1]["predicted_plan"]["relevant_tables"] == ["wrong"]


def test_load_prepared_records_rejects_legacy_oracle_hint_marker(tmp_path) -> None:
    path = tmp_path / "prepared.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {
                        "role": "user",
                        "content": "SQL planning hints:\nRelevant tables: singer\n\nQuestion:\nList singers.",
                    },
                    {"role": "assistant", "content": "SELECT name FROM singer;"},
                ],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="oracle planning hints"):
        load_prepared_records(path)


def test_load_prepared_records_limit_applies_to_turns(tmp_path) -> None:
    path = tmp_path / "prepared.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "SELECT 1;"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "SELECT 2;"},
                ],
            }
        )
        + "\n"
    )

    records = load_prepared_records(path, limit=1)

    assert len(records) == 1
    assert records[0]["reference_sql"] == "SELECT 1;"


def test_expand_prepared_record_uses_stable_fallback_dialog_id() -> None:
    records = expand_prepared_record(
        {
            "messages": [
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "SELECT 1;"},
            ]
        },
        index=7,
    )

    assert records[0]["dialog_id"] == "prepared-7"
    assert records[0]["id"] == "prepared-7:0"


def test_write_results_writes_jsonl(tmp_path) -> None:
    output = tmp_path / "results.jsonl"

    assert write_results([{"id": 1, "score": 1.0}], output) == 1
    assert json.loads(output.read_text()) == {"id": 1, "score": 1.0}


def test_enforce_sql_only_instruction_appends_to_system_message() -> None:
    messages = enforce_sql_only_instruction(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
    )

    assert messages[0]["role"] == "system"
    assert "Return only one SQL query" in messages[0]["content"]


def test_database_path_for_record_uses_cosql_layout(tmp_path) -> None:
    db_dir = tmp_path / "car_1"
    db_dir.mkdir()
    db_path = db_dir / "car_1.sqlite"
    db_path.write_text("")

    assert database_path_for_record({"database_id": "car_1"}, tmp_path) == db_path
    assert database_path_for_record({"database_id": "missing"}, tmp_path) is None


def test_generate_sql_disables_qwen_thinking() -> None:
    class Message:
        content = "SELECT 1"

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    class Completions:
        def create(self, **kwargs):
            assert kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
            return Response()

    class Chat:
        completions = Completions()

    class Client:
        chat = Chat()

    text, latency_ms = generate_sql(
        Client(),
        model_name="model",
        messages=[{"role": "user", "content": "q"}],
        temperature=0.0,
        max_tokens=8,
    )

    assert text == "SELECT 1"
    assert latency_ms >= 0
