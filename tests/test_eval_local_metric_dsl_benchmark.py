from __future__ import annotations

import json

from eval.local_metric_dsl_benchmark import (
    prompt_from_messages,
    run_local_metric_dsl_benchmark,
)


def test_prompt_from_messages_uses_chat_template_without_sql_suffix() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            assert tokenize is False
            assert add_generation_prompt is True
            assert kwargs["enable_thinking"] is False
            return messages[-1]["content"] + "\nassistant:"

    prompt = prompt_from_messages(Tokenizer(), [{"role": "user", "content": "DSL?"}])

    assert prompt == "DSL?\nassistant:"


def test_run_local_metric_dsl_benchmark_writes_metric_generation_field(tmp_path, monkeypatch) -> None:
    input_path = tmp_path / "metric.jsonl"
    output_path = tmp_path / "metric.results.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "fixture_id": "measure_preservation_metric",
                "messages": [
                    {"role": "system", "content": "You write only governed metric DSL."},
                    {"role": "user", "content": "Show governed revenue by country."},
                    {"role": "assistant", "content": "MEASURE(revenue) BY customer_country"},
                ],
                "evaluation_mode": "metric_dsl",
                "reference_sql": "SELECT 1",
                "gold_dsl": "MEASURE(revenue) BY customer_country",
            }
        )
        + "\n"
    )

    monkeypatch.setattr(
        "eval.local_metric_dsl_benchmark.load_model_and_tokenizer",
        lambda **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        "eval.local_metric_dsl_benchmark.generate_local_text",
        lambda model, tokenizer, *, messages, max_new_tokens: (
            "MEASURE(revenue) BY customer_country",
            1.0,
        ),
    )

    assert (
        run_local_metric_dsl_benchmark(
            model_name="local-9b",
            adapter_path=None,
            input_path=input_path,
            output_path=output_path,
            output_field="generated_metric_dsl",
            max_new_tokens=32,
            max_memory_gb=None,
        )
        == 0
    )

    [row] = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert row["generated_metric_dsl"] == "MEASURE(revenue) BY customer_country"
    assert row["model_name"] == "local-9b"
    assert row["generation_latency_ms"] == 1.0
