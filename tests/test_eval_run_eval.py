from __future__ import annotations

import json

import pytest

import eval.run_eval as run_eval_module
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


def test_load_prepared_records_preserves_split_provenance(tmp_path) -> None:
    path = tmp_path / "prepared.jsonl"
    path.write_text(
        json.dumps(
            {
                "source": "cosql_dev_clean_holdout_v1",
                "database_id": "car_1",
                "evaluation_mode": "non_oracle_generation",
                "planning_label_source": "gold_reference_sql",
                "uses_oracle_planning_hints": False,
                "semantic_context_pruned_by_oracle_labels": False,
                "split_id": "cosql_dev_clean_holdout_v1",
                "split_role": "clean_local_holdout",
                "split_row_id": "cosql_dev:0100:car_1",
                "split_source_path": "data/raw/cosql_dataset/sql_state_tracking/cosql_dev.json",
                "split_source_sha256": "abc123",
                "split_row_ids_sha256": "row-hash",
                "schema_link_labels": [
                    {"relevant_tables": ["cars"], "projection_shape": {"selected_count": 1}},
                    {"relevant_tables": ["cars"], "projection_shape": {"selected_count": 1}},
                ],
                "gold_plans": [
                    {"relevant_tables": ["cars"], "projection_shape": {"selected_count": 1}},
                    {"relevant_tables": ["cars"], "projection_shape": {"selected_count": 1}},
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

    records = load_prepared_records(path)

    assert [record["split_row_id"] for record in records] == [
        "cosql_dev:0100:car_1",
        "cosql_dev:0100:car_1",
    ]
    assert records[0]["split_id"] == "cosql_dev_clean_holdout_v1"
    assert records[0]["split_role"] == "clean_local_holdout"
    assert records[0]["split_source_sha256"] == "abc123"
    assert records[1]["turn_index"] == 1


def test_load_prepared_records_expands_multi_turn_dialogs(tmp_path) -> None:
    path = tmp_path / "prepared.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "dialog-a",
                "source": "unit",
                "database_id": "db1",
                "history_policy": "gold_sql_teacher_forced",
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
                    {"prediction_source": "json_planner_predictions", "relevant_tables": ["one"]},
                    {"prediction_source": "json_planner_predictions", "relevant_tables": ["wrong"]},
                ],
                "predicted_plan_source": "json_planner_predictions",
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
    assert records[0]["history_policy"] == "gold_sql_teacher_forced"
    assert records[0]["planning_label_source"] == "gold_reference_sql"
    assert records[0]["uses_oracle_planning_hints"] is True
    assert records[0]["semantic_context_pruned_by_oracle_labels"] is True
    assert records[0]["oracle_diagnostic_warning"] == "oracle warning"
    assert records[0]["gold_plan"]["relevant_tables"] == ["one"]
    assert records[1]["gold_plan"]["relevant_tables"] == ["two"]
    assert records[0]["predicted_plan"]["relevant_tables"] == ["one"]
    assert records[1]["predicted_plan"]["relevant_tables"] == ["wrong"]
    assert records[0]["predicted_plan_source"] == "json_planner_predictions"
    assert records[1]["predicted_plan_source"] == "json_planner_predictions"


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


def test_load_prepared_records_rejects_predicted_mode_without_predictions(tmp_path) -> None:
    path = tmp_path / "prepared.jsonl"
    path.write_text(
        json.dumps(
            {
                "evaluation_mode": "predicted_planner",
                "gold_plans": [{"relevant_tables": ["singer"]}],
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "SELECT 1;"},
                ],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="predicted planner"):
        load_prepared_records(path)


def test_messages_for_generation_adds_predicted_plan_without_oracle_language() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "List airline names."},
        ],
        "evaluation_mode": "predicted_planner",
        "predicted_plan": {
            "relevant_tables": ["airlines"],
            "relevant_columns": ["airlines.name"],
            "join_path": [],
            "query_skeleton": {"select": True},
            "projection_shape": {
                "selected_count": 1,
                "selected_expressions": ["airlines.name"],
                "preserve_duplicates": True,
            },
        },
    }

    messages = messages_for_generation(record)

    assert "Return only one SQL query" in messages[0]["content"]
    assert "Predicted SQL plan" in messages[-1]["content"]
    assert "Relevant tables: airlines" in messages[-1]["content"]
    assert "derived from reference SQL" not in messages[-1]["content"]


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


def test_messages_for_generation_rejects_empty_predicted_plan() -> None:
    record = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "List airline names."},
        ],
        "evaluation_mode": "predicted_planner",
        "predicted_plan": {"query_skeleton": {"select": True}},
    }

    with pytest.raises(ValueError, match="relevant table or column"):
        messages_for_generation(record)


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


def test_summarize_eval_metrics_records_split_provenance_hashes() -> None:
    metrics = summarize_eval_metrics(
        [
            {
                "execution_score": 1.0,
                "strict_execution_score": 1.0,
                "value_execution_score": 1.0,
                "syntax_valid": True,
                "generation_latency_ms": 10.0,
                "split_id": "cosql_dev_clean_holdout_v1",
                "split_role": "clean_local_holdout",
                "split_row_id": "cosql_dev:0100:car_1",
                "split_source_sha256": "source-hash",
                "turn_index": 0,
            },
            {
                "execution_score": 0.0,
                "strict_execution_score": 0.0,
                "value_execution_score": 0.0,
                "syntax_valid": True,
                "generation_latency_ms": 20.0,
                "split_id": "cosql_dev_clean_holdout_v1",
                "split_role": "clean_local_holdout",
                "split_row_id": "cosql_dev:0100:car_1",
                "split_source_sha256": "source-hash",
                "turn_index": 1,
            },
            {
                "execution_score": 1.0,
                "strict_execution_score": 1.0,
                "value_execution_score": 1.0,
                "syntax_valid": True,
                "generation_latency_ms": 30.0,
                "split_id": "cosql_dev_clean_holdout_v1",
                "split_role": "clean_local_holdout",
                "split_row_id": "cosql_dev:0101:poker_player",
                "split_source_sha256": "source-hash",
                "turn_index": 0,
            },
        ]
    )

    assert metrics["split_ids"] == {"cosql_dev_clean_holdout_v1": 3}
    assert metrics["split_roles"] == {"clean_local_holdout": 3}
    assert metrics["split_row_count"] == 2
    assert metrics["split_eval_turn_count"] == 3
    assert metrics["split_source_sha256s"] == ["source-hash"]
    assert metrics["split_row_ids_sha256"]
    assert metrics["split_eval_turn_ids_sha256"]
    assert metrics["split_row_ids_sha256"] != metrics["split_eval_turn_ids_sha256"]


def test_run_eval_manifest_records_prepared_split_provenance(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "results.jsonl"
    manifest_path = tmp_path / "results.manifest.json"
    input_path.write_text(
        json.dumps(
            {
                "source": "cosql_dev_clean_holdout_v1",
                "database_id": "car_1",
                "evaluation_mode": "non_oracle_generation",
                "planning_label_source": "gold_reference_sql",
                "uses_oracle_planning_hints": False,
                "semantic_context_pruned_by_oracle_labels": False,
                "split_id": "cosql_dev_clean_holdout_v1",
                "split_role": "clean_local_holdout",
                "split_row_id": "cosql_dev:0100:car_1",
                "split_source_path": "data/raw/cosql_dataset/sql_state_tracking/cosql_dev.json",
                "split_source_sha256": "source-hash",
                "split_row_ids_sha256": "row-hash",
                "schema_link_labels": [
                    {"relevant_tables": [], "projection_shape": {"selected_count": 1}},
                ],
                "gold_plans": [
                    {"relevant_tables": [], "projection_shape": {"selected_count": 1}},
                ],
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def fake_generate_sql(*args, **kwargs):
        return "SELECT 1", 12.0

    monkeypatch.setattr(run_eval_module, "generate_sql", fake_generate_sql)

    assert (
        run_eval_module.run_eval(
            benchmark="prepared",
            endpoint="http://localhost:8000/v1",
            model_name="unit-model",
            output=output_path,
            input_path=input_path,
            limit=None,
            database_root=None,
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=16,
            allow_oracle_plan=False,
            manifest_output=manifest_path,
            command=["uv", "run", "python", "-m", "eval.run_eval"],
        )
        == 0
    )

    result = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result["split_row_id"] == "cosql_dev:0100:car_1"
    assert manifest["metrics"]["split_ids"] == {"cosql_dev_clean_holdout_v1": 1}
    assert manifest["metrics"]["split_roles"] == {"clean_local_holdout": 1}
    assert manifest["metrics"]["split_row_count"] == 1
    assert manifest["metrics"]["split_eval_turn_count"] == 1
    assert manifest["metrics"]["split_source_sha256s"] == ["source-hash"]
    assert manifest["metrics"]["split_row_ids_sha256"]
    assert manifest["metrics"]["split_eval_turn_ids_sha256"]


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
