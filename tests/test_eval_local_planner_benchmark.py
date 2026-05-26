from __future__ import annotations

import json

from eval.local_planner_benchmark import (
    prompt_from_messages,
    run_local_planner_benchmark,
)


def test_prompt_from_messages_uses_chat_template_for_planner() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False
            assert add_generation_prompt is True
            assert kwargs["enable_thinking"] is False
            return messages[-1]["content"] + "\nassistant:"

    prompt = prompt_from_messages(Tokenizer(), [{"role": "user", "content": "Plan?"}])

    assert prompt == "Plan?\nassistant:"


def test_run_local_planner_benchmark_writes_prediction_rows(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "planner_predictions.jsonl"
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
                "gold_plans": [{"relevant_tables": ["customers"]}],
            }
        )
        + "\n"
    )

    monkeypatch.setattr(
        "eval.local_planner_benchmark.load_model_and_tokenizer",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        "eval.local_planner_benchmark.generate_local_text",
        lambda model, tokenizer, *, messages, max_new_tokens: (
            json.dumps({"relevant_tables": ["customers"], "projection_shape": {"selected_count": 1}}),
            1.0,
        ),
    )

    assert (
        run_local_planner_benchmark(
            model_name="local-9b",
            adapter_path=None,
            input_path=input_path,
            output_path=output_path,
            max_new_tokens=64,
            max_memory_gb=None,
        )
        == 0
    )

    [row] = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert row["id"] == "dialog-a:0"
    assert row["model_name"] == "local-9b"
    assert row["predicted_plan"]["relevant_tables"] == ["customers"]
    assert row["planner_latency_ms"] == 1.0
