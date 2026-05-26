from __future__ import annotations

import json
from types import SimpleNamespace

from eval.local_benchmark import (
    SQL_ONLY_INSTRUCTION,
    enforce_sql_only_instruction,
    prompt_from_messages,
    run_local_benchmark,
)


def test_prompt_from_messages_uses_chat_template() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False
            assert add_generation_prompt is True
            assert kwargs["enable_thinking"] is False
            return messages[-1]["content"] + "\nassistant:"

    prompt = prompt_from_messages(Tokenizer(), [{"role": "user", "content": "SQL?"}])

    assert prompt == "SQL?\nassistant:"


def test_prompt_from_messages_fallback() -> None:
    class Tokenizer:
        chat_template = ""

    prompt = prompt_from_messages(Tokenizer(), [{"role": "user", "content": "SQL?"}])

    assert prompt == f"system: {SQL_ONLY_INSTRUCTION}\nuser: SQL?\nassistant:"


def test_enforce_sql_only_instruction_adds_system_message() -> None:
    messages = enforce_sql_only_instruction([{"role": "user", "content": "SQL?"}])

    assert messages[0]["role"] == "system"
    assert SQL_ONLY_INSTRUCTION in messages[0]["content"]


def test_run_local_benchmark_injects_predicted_plan_messages(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "results.jsonl"
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
                "evaluation_mode": "predicted_planner",
                "gold_plans": [{"relevant_tables": ["customers"]}],
                "predicted_plans": [
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
    captured: dict[str, list[dict[str, str]]] = {}

    monkeypatch.setattr(
        "eval.local_benchmark.load_model_and_tokenizer",
        lambda **kwargs: (object(), object()),
    )

    def fake_generate(model, tokenizer, *, messages, max_new_tokens):
        captured["messages"] = messages
        return "SELECT name FROM customers;", 1.0

    monkeypatch.setattr("eval.local_benchmark.generate_local_sql", fake_generate)
    monkeypatch.setattr(
        "eval.local_benchmark.score_single_turn",
        lambda reference_sql, generated_sql, database_path=None: SimpleNamespace(
            execution_score=1.0,
            strict_execution_score=1.0,
            value_execution_score=1.0,
            order_sensitive=False,
            normalized_match=True,
            syntax_valid=True,
            error=None,
        ),
    )

    assert (
        run_local_benchmark(
            model_name="local-9b",
            adapter_path=None,
            benchmark="prepared",
            input_path=input_path,
            output=output_path,
            limit=None,
            max_new_tokens=32,
            max_memory_gb=None,
            database_root=None,
            allow_oracle_plan=False,
        )
        == 0
    )

    prompt_text = json.dumps(captured["messages"])
    assert "Predicted SQL plan" in prompt_text
    assert "Relevant tables: customers" in prompt_text
