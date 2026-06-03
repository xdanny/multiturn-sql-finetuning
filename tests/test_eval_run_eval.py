from __future__ import annotations

import json

from eval.run_eval import (
    assistant_turn_indices,
    database_path_for_record,
    enforce_sql_only_instruction,
    expand_prepared_record,
    extract_reference_sql,
    generate_sql,
    load_prepared_records,
    messages_for_generation,
    summarize_eval_metrics,
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
                "history_policy": "gold_sql_teacher_forced",
                "evaluation_mode": "non_oracle_generation",
                "schema_link_label_source": "reference_sql_for_scoring_only",
                "schema_link_labels": [{"relevant_tables": ["one"]}, {"relevant_tables": ["two"]}],
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

    records = load_prepared_records(path)

    assert assistant_turn_indices(json.loads(path.read_text())["messages"]) == [2, 4]
    assert [record["reference_sql"] for record in records] == ["SELECT 1;", "SELECT 2;"]
    assert [record["turn_index"] for record in records] == [0, 1]
    assert [record["turn_count"] for record in records] == [2, 2]
    assert records[0]["messages"][-1]["content"] == "q1"
    assert records[1]["messages"][-1]["content"] == "q2"
    assert records[1]["messages"][2]["content"] == "SELECT 1;"
    assert records[0]["evaluation_mode"] == "non_oracle_generation"
    assert records[0]["history_policy"] == "gold_sql_teacher_forced"
    assert records[0]["schema_link_labels"]["relevant_tables"] == ["one"]
    assert records[1]["schema_link_labels"]["relevant_tables"] == ["two"]


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


def test_messages_for_generation_only_adds_sql_instruction() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "List airline names."},
        ],
    }

    messages = messages_for_generation(record)

    assert "Return only one SQL query" in messages[0]["content"]
    assert messages[-1]["content"] == "List airline names."


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


def test_summarize_eval_metrics_records_manifest_compatible_latency_alias() -> None:
    metrics = summarize_eval_metrics(
        [
            {
                "execution_score": 1.0,
                "strict_execution_score": 1.0,
                "value_execution_score": 1.0,
                "syntax_valid": True,
                "generation_latency_ms": 10.0,
            },
            {
                "execution_score": 0.0,
                "strict_execution_score": 0.0,
                "value_execution_score": 0.0,
                "syntax_valid": True,
                "generation_latency_ms": 20.0,
            },
        ]
    )

    assert metrics["mean_generation_latency_ms"] == 15.0
    assert metrics["mean_latency_ms"] == 15.0


def test_summarize_eval_metrics_records_teacher_forced_history_policy() -> None:
    metrics = summarize_eval_metrics(
        [
            {
                "execution_score": 1.0,
                "strict_execution_score": 1.0,
                "value_execution_score": 1.0,
                "syntax_valid": True,
                "generation_latency_ms": 10.0,
                "source": "unit",
                "evaluation_mode": "non_oracle_generation",
                "dialog_id": "dialog-a",
                "history_policy": "gold_sql_teacher_forced",
            },
            {
                "execution_score": 0.0,
                "strict_execution_score": 0.0,
                "value_execution_score": 0.0,
                "syntax_valid": True,
                "generation_latency_ms": 20.0,
                "source": "unit",
                "evaluation_mode": "non_oracle_generation",
                "dialog_id": "dialog-a",
                "history_policy": "gold_sql_teacher_forced",
            },
        ]
    )

    assert metrics["history_policy"] == "gold_sql_teacher_forced"
    assert metrics["history_policies"] == {"gold_sql_teacher_forced": 2}


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
