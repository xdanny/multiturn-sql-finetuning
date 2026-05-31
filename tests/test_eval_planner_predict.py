from __future__ import annotations

import json

import pytest

from eval.planner_eval import annotate_prepared_records_with_plans
from eval.planner_predict import (
    generate_planner_json,
    planner_messages_for_record,
    predict_planner_records,
    run_planner_predict,
    write_planner_predictions,
)


def _expanded_record() -> dict:
    return {
        "id": "dialog-a:0",
        "dialog_id": "dialog-a",
        "turn_index": 0,
        "database_id": "store",
        "messages": [
            {"role": "system", "content": "You are a SQL expert."},
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
        "gold_plan": {"relevant_tables": ["customers"]},
    }


def test_planner_messages_exclude_reference_sql_gold_plan_and_oracle_markers() -> None:
    record = _expanded_record()

    messages = planner_messages_for_record(record)
    prompt_text = json.dumps(messages).lower()

    assert "return only one json object" in prompt_text
    assert "relevant_tables" in prompt_text
    assert "select customers.name" not in prompt_text
    assert "gold_plan" not in prompt_text
    assert "gold_reference_sql" not in prompt_text
    assert "oracle sql planning hints" not in prompt_text


def test_predict_planner_records_preserves_fenced_and_malformed_json() -> None:
    records = [
        _expanded_record(),
        {**_expanded_record(), "id": "dialog-a:1", "turn_index": 1},
    ]
    outputs = iter(
        [
            """```json
            {"relevant_tables": ["Customers"], "projection_shape": {"selected_count": 1}}
            ```""",
            "not json",
        ]
    )

    def fake_generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        assert "SELECT customers.name" not in json.dumps(messages)
        return next(outputs), 2.5

    predictions = predict_planner_records(
        records,
        generate_fn=fake_generate,
        model_name="planner-9b",
    )

    assert predictions[0]["id"] == "dialog-a:0"
    assert predictions[0]["model_name"] == "planner-9b"
    assert predictions[0]["predicted_plan"]["relevant_tables"] == ["customers"]
    assert predictions[0]["planner_latency_ms"] == 2.5
    assert predictions[1]["predicted_plan"]["parseable"] is False
    assert "no JSON object" in predictions[1]["predicted_plan"]["planner_parse_error"]
    assert predictions[1]["raw_planner_output"] == "not json"


def test_generate_planner_json_disables_qwen_thinking() -> None:
    class Message:
        content = '{"relevant_tables": ["customers"]}'

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

    text, latency_ms = generate_planner_json(
        Client(),
        model_name="planner-9b",
        messages=[{"role": "user", "content": "q"}],
        temperature=0.0,
        max_tokens=8,
    )

    assert text == '{"relevant_tables": ["customers"]}'
    assert latency_ms >= 0


def test_planner_predictions_feed_existing_json_planner_loader(tmp_path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    predictions_path = tmp_path / "planner_predictions.jsonl"
    output_path = tmp_path / "predicted_prepared.jsonl"
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
                            "customers(id int, name text)\n\n"
                            "Question:\nShow customer names."
                        ),
                    },
                    {"role": "assistant", "content": "SELECT name FROM customers;"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [
                    {
                        "relevant_tables": ["customers"],
                        "relevant_columns": ["customers.name"],
                        "projection_shape": {"selected_count": 1},
                    }
                ],
            }
        )
        + "\n"
    )
    predictions = predict_planner_records(
        [
            {
                **_expanded_record(),
                "id": "dialog-a:0",
                "dialog_id": "dialog-a",
            }
        ],
        generate_fn=lambda messages: (
            json.dumps(
                {
                    "relevant_tables": ["customers"],
                    "relevant_columns": ["customers.name"],
                    "projection_shape": {"selected_count": 1},
                }
            ),
            1.0,
        ),
        model_name="planner-9b",
    )
    write_planner_predictions(predictions, predictions_path)

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
    assert row["evaluation_mode"] == "predicted_planner"
    assert row["predicted_plans"][0]["prediction_source"] == "json_planner_predictions"
    assert row["predicted_plans"][0]["relevant_tables"] == ["customers"]


def test_run_planner_predict_can_use_local_adapter_backend(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    input_path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {
                        "role": "user",
                        "content": (
                            "Schema/context:\n"
                            "customers(id int, name text)\n\n"
                            "Question:\nShow customer names."
                        ),
                    },
                    {"role": "assistant", "content": "SELECT name FROM customers;"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{"relevant_tables": ["customers"]}],
            }
        )
        + "\n"
    )

    def fake_local_adapter_generate_fn(*, model_name, adapter_path, max_tokens, max_memory_gb):
        assert model_name == "unsloth/Qwen3.5-9B"
        assert adapter_path == tmp_path / "adapter"
        assert max_tokens == 128
        assert max_memory_gb == 24

        def generate(messages: list[dict[str, str]]) -> tuple[str, float]:
            assert "return only one json object" in json.dumps(messages).lower()
            return json.dumps({"relevant_tables": ["customers"]}), 3.0

        return generate

    monkeypatch.setattr(
        "eval.local_generation.local_adapter_generate_fn",
        fake_local_adapter_generate_fn,
    )

    assert (
        run_planner_predict(
            input_path=input_path,
            output=output_path,
            model_name="unsloth/Qwen3.5-9B",
            backend="local",
            endpoint="http://localhost:8000/v1",
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=128,
            adapter_path=adapter_path,
            max_memory_gb=24,
            limit=1,
        )
        == 0
    )
    prediction = json.loads(output_path.read_text())
    metadata = json.loads(output_path.with_suffix(".jsonl.meta.json").read_text())
    assert prediction["predicted_plan"]["relevant_tables"] == ["customers"]
    assert metadata["backend"] == "local"
    assert metadata["adapter_path"] == str(adapter_path)


def test_run_planner_predict_rejects_oracle_prompt_rows(tmp_path) -> None:
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
                            "Relevant tables: customers\n\n"
                            "Question:\nShow customer names."
                        ),
                    },
                    {"role": "assistant", "content": "SELECT name FROM customers;"},
                ],
                "evaluation_mode": "oracle_planner_diagnostic",
                "planning_label_source": "gold_reference_sql",
                "uses_oracle_planning_hints": True,
                "gold_plans": [{"relevant_tables": ["customers"]}],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="oracle planning hints"):
        run_planner_predict(
            input_path=input_path,
            output=tmp_path / "predictions.jsonl",
            model_name="planner-9b",
            backend="endpoint",
            endpoint="http://localhost:8000/v1",
            api_key="EMPTY",
            temperature=0.0,
            max_tokens=256,
            adapter_path=None,
            max_memory_gb=None,
            limit=1,
        )
