from __future__ import annotations

import json
from pathlib import Path

from eval.generate_metric_dsl_predictions import (
    _chat_prompt,
    generate_prediction_rows,
    run_generate_metric_dsl_predictions,
)


def _input_rows() -> list[dict]:
    return [
        {
            "id": "metric-1",
            "fixture_id": "metric-1",
            "generation_target": "metric_dsl",
            "messages": [
                {"role": "system", "content": "Return metric DSL."},
                {"role": "user", "content": "Question"},
            ],
            "reference_sql": "SELECT SUM(amount) FROM orders",
            "gold_dsl": "MEASURE(revenue)",
            "reference_sql_visible_to_model": False,
            "scoring_fields_visible_to_model": False,
        }
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_generate_prediction_rows_preserves_scorer_fields_but_not_prompt_leakage() -> None:
    seen_prompts: list[list[dict[str, str]]] = []

    def fake_generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        seen_prompts.append(messages)
        return "MEASURE(revenue)", 12.0

    [row] = generate_prediction_rows(
        _input_rows(),
        model_name="metric-adapter",
        generate_fn=fake_generate,
    )

    assert row["model_name"] == "metric-adapter"
    assert row["raw_generation"] == "MEASURE(revenue)"
    assert row["predicted_dsl"] == "MEASURE(revenue)"
    assert row["generation_latency_ms"] == 12.0
    assert row["reference_sql"] == "SELECT SUM(amount) FROM orders"
    assert row["gold_dsl"] == "MEASURE(revenue)"
    assert row["reference_sql_visible_to_model"] is False
    assert row["scoring_fields_visible_to_model"] is False
    assert "SELECT SUM" not in json.dumps(seen_prompts[0])
    assert "MEASURE(revenue)" not in json.dumps(seen_prompts[0])


def test_generate_prediction_rows_extracts_direct_sql_from_markdown() -> None:
    def fake_generate(_messages: list[dict[str, str]]) -> tuple[str, float]:
        return "```sql\nSELECT 1;\n```", 3.0

    [row] = generate_prediction_rows(
        [{**_input_rows()[0], "generation_target": "direct_sql"}],
        model_name="direct-adapter",
        generate_fn=fake_generate,
    )

    assert row["generated_sql"] == "SELECT 1;"
    assert "predicted_dsl" not in row


def test_generate_prediction_rows_rejects_visible_scorer_fields() -> None:
    bad_row = {**_input_rows()[0], "reference_sql_visible_to_model": True}

    def fake_generate(_messages: list[dict[str, str]]) -> tuple[str, float]:
        return "MEASURE(revenue)", 1.0

    try:
        generate_prediction_rows([bad_row], model_name="metric-adapter", generate_fn=fake_generate)
    except ValueError as exc:
        assert "scorer fields" in str(exc)
    else:
        raise AssertionError("expected scorer-field leakage rejection")


def test_run_generate_metric_dsl_predictions_writes_jsonl(tmp_path) -> None:
    input_path = tmp_path / "inputs.jsonl"
    output_path = tmp_path / "predictions.jsonl"
    _write_jsonl(input_path, _input_rows())

    exit_code = run_generate_metric_dsl_predictions(
        input_path=input_path,
        output_path=output_path,
        model_name="metric-adapter",
        generate_fn=lambda _messages: ("MEASURE(revenue)", 2.0),
    )

    assert exit_code == 0
    [row] = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert row["predicted_dsl"] == "MEASURE(revenue)"
    assert row["model_name"] == "metric-adapter"


def test_chat_prompt_does_not_replace_metric_dsl_system_prompt() -> None:
    class Tokenizer:
        chat_template = ""

    prompt = _chat_prompt(
        Tokenizer(),
        [
            {"role": "system", "content": "Return metric DSL only."},
            {"role": "user", "content": "Question"},
        ],
    )

    assert "Return metric DSL only." in prompt
    assert "generate only the correct SQL query" not in prompt
