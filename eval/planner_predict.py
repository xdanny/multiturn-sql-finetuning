"""
Generate non-oracle planner JSON predictions for prepared SQL turns.

The output is keyed by the expanded turn id (`dialog_id:turn_index`) so it can
feed `eval.planner_eval --planner-source json_planner_predictions`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

from eval.planner_eval import (
    JSON_PLANNER_PREDICTIONS_SOURCE,
    parse_json_plan_prediction,
)
from eval.run_eval import load_prepared_records

PLANNER_JSON_INSTRUCTION = """
You are a non-oracle SQL planner. Return only one JSON object, with no markdown
or prose. Do not write SQL. Do not use reference SQL or answer-key labels.

Use this schema:
{
  "relevant_tables": ["table_name"],
  "relevant_columns": ["table.column"],
  "join_path": ["table_a.key = table_b.key"],
  "query_skeleton": {
    "select": true,
    "join": false,
    "where": false,
    "group_by": false,
    "having": false,
    "order_by": false,
    "limit": false,
    "nested": false,
    "distinct": false
  },
  "projection_shape": {
    "selected_expressions": ["table.column"],
    "selected_count": 1,
    "aggregations": [],
    "group_by": [],
    "order_by": null,
    "limit": null,
    "distinct": false,
    "preserve_duplicates": true
  }
}

Projection rules:
- `projection_shape.selected_expressions` is the final answer column sequence.
- Preserve the user's requested output order exactly; do not alphabetize or sort.
- Use visible schema table/column names, not generated SQL aliases such as T1 or T2.
- Keep `selected_count` equal to the length of `selected_expressions`.
- For "name and id", output name before id; for "average X for each Y", output Y before avg(X)
  unless the user asks for the aggregate first.
""".strip()

GeneratePlannerFn = Callable[[list[dict[str, str]]], tuple[str, float]]


def planner_messages_for_record(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return prompt messages for planner prediction without answer-key fields."""

    messages = [dict(message) for message in record["messages"]]
    for message in messages:
        if message.get("role") == "system":
            message["content"] = f"{message['content']}\n\n{PLANNER_JSON_INSTRUCTION}"
            return messages
    return [{"role": "system", "content": PLANNER_JSON_INSTRUCTION}, *messages]


def generate_planner_json(
    client: OpenAI,
    *,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> tuple[str, float]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    return response.choices[0].message.content or "", latency_ms


def predict_planner_records(
    records: list[dict[str, Any]],
    *,
    generate_fn: GeneratePlannerFn,
    model_name: str,
    prediction_source: str = JSON_PLANNER_PREDICTIONS_SOURCE,
) -> list[dict[str, Any]]:
    """Generate planner prediction rows for already-expanded prepared records."""

    predictions = []
    for record in records:
        messages = planner_messages_for_record(record)
        raw_output, latency_ms = generate_fn(messages)
        predictions.append(
            {
                "id": str(record["id"]),
                "dialog_id": str(record.get("dialog_id") or ""),
                "turn_index": int(record.get("turn_index") or 0),
                "database_id": record.get("database_id"),
                "model_name": model_name,
                "prediction_source": prediction_source,
                "raw_planner_output": raw_output,
                "predicted_plan": parse_json_plan_prediction(
                    raw_output,
                    prediction_source=prediction_source,
                ),
                "planner_latency_ms": latency_ms,
            }
        )
    return predictions


def write_planner_predictions(rows: list[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def run_planner_predict(
    *,
    input_path: Path,
    output: Path,
    model_name: str,
    backend: str,
    endpoint: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    adapter_path: Path | None,
    max_memory_gb: int | None,
    limit: int | None,
    command: Sequence[str] | None = None,
) -> int:
    records = load_prepared_records(input_path, limit=limit, allow_oracle_plan=False)
    if backend == "endpoint":
        client = OpenAI(base_url=endpoint, api_key=api_key)

        def generate_fn(messages: list[dict[str, str]]) -> tuple[str, float]:
            return generate_planner_json(
                client,
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

    elif backend == "local":
        if temperature != 0.0:
            raise ValueError("local planner prediction currently supports only temperature=0.0")
        from eval.local_generation import local_adapter_generate_fn  # noqa: PLC0415

        generate_fn = local_adapter_generate_fn(
            model_name=model_name,
            adapter_path=adapter_path,
            max_tokens=max_tokens,
            max_memory_gb=max_memory_gb,
        )
    else:
        raise ValueError(f"unknown planner prediction backend: {backend}")

    predictions = predict_planner_records(
        records,
        generate_fn=generate_fn,
        model_name=model_name,
    )
    written = write_planner_predictions(predictions, output)
    metadata = {
        "command": list(command or sys.argv),
        "backend": backend,
        "input_path": str(input_path),
        "model_name": model_name,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "prediction_source": JSON_PLANNER_PREDICTIONS_SOURCE,
        "row_count": written,
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {written} planner prediction rows to {output}")
    print(f"Wrote planner prediction metadata to {metadata_path}")
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--backend", choices=["endpoint", "local"], default="endpoint")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-memory-gb", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    return run_planner_predict(
        input_path=args.input,
        output=args.output,
        model_name=args.model_name,
        backend=args.backend,
        endpoint=args.endpoint,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        adapter_path=args.adapter_path,
        max_memory_gb=args.max_memory_gb,
        limit=args.limit,
        command=sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
