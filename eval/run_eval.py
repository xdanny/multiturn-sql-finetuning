"""
Run SQL benchmarks against an OpenAI-compatible chat endpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI

from data.plan_contract import ORACLE_PLANNER_DIAGNOSTIC
from data.prepare import ORACLE_DIAGNOSTIC_WARNING
from eval.ragas_metrics import extract_sql, score_single_turn

BENCHMARKS = ["prepared", "sparc", "bird_mini_dev"]
SQL_ONLY_INSTRUCTION = (
    "Return only one SQL query. Do not explain, reason step by step, use markdown, or include prose."
)
ORACLE_PLAN_MARKERS = (
    "Oracle SQL planning hints",
    "SQL planning hints:",
)


def extract_reference_sql(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message["content"]
    raise ValueError("prepared record has no assistant reference SQL")


def enforce_sql_only_instruction(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    updated = [dict(message) for message in messages]
    for message in updated:
        if message["role"] == "system":
            message["content"] = f"{message['content']}\n\n{SQL_ONLY_INSTRUCTION}"
            return updated
    return [{"role": "system", "content": SQL_ONLY_INSTRUCTION}, *updated]


def assistant_turn_indices(messages: list[dict[str, str]]) -> list[int]:
    return [index for index, message in enumerate(messages) if message.get("role") == "assistant"]


def record_uses_oracle_plan(record: dict[str, Any]) -> bool:
    if record.get("uses_oracle_planning_hints") or record.get(
        "semantic_context_pruned_by_oracle_labels"
    ):
        return True
    return any(
        marker in str(message.get("content", ""))
        for message in record.get("messages", [])
        for marker in ORACLE_PLAN_MARKERS
    )


def expand_prepared_record(record: dict[str, Any], *, index: int) -> list[dict[str, Any]]:
    messages = record["messages"]
    assistant_indices = assistant_turn_indices(messages)
    if not assistant_indices:
        raise ValueError("prepared record has no assistant reference SQL")

    dialog_id = str(record.get("dialog_id") or record.get("id") or f"prepared-{index}")
    source = record.get("source", "prepared")
    database_id = record.get("database_id")
    schema_link_labels = record.get("schema_link_labels") or []
    gold_plans = record.get("gold_plans") or schema_link_labels
    predicted_plans = record.get("predicted_plans") or []
    evaluation_mode = record.get("evaluation_mode") or "unknown"
    uses_oracle_planning_hints = bool(record.get("uses_oracle_planning_hints"))
    semantic_context_pruned_by_oracle_labels = bool(
        record.get("semantic_context_pruned_by_oracle_labels")
    )
    turn_count = len(assistant_indices)
    expanded = []
    for turn_index, assistant_index in enumerate(assistant_indices):
        expanded.append(
            {
                "id": f"{dialog_id}:{turn_index}",
                "dialog_id": dialog_id,
                "turn_index": turn_index,
                "turn_count": turn_count,
                "messages": messages[:assistant_index],
                "reference_sql": messages[assistant_index]["content"],
                "source": source,
                "database_id": database_id,
                "evaluation_mode": evaluation_mode,
                "planning_label_source": record.get("planning_label_source"),
                "uses_oracle_planning_hints": uses_oracle_planning_hints,
                "semantic_context_pruned_by_oracle_labels": semantic_context_pruned_by_oracle_labels,
                "oracle_diagnostic_warning": record.get("oracle_diagnostic_warning"),
                "gold_plan": gold_plans[turn_index] if turn_index < len(gold_plans) else None,
                "predicted_plan": predicted_plans[turn_index]
                if turn_index < len(predicted_plans)
                else None,
                "schema_link_labels": schema_link_labels[turn_index]
                if turn_index < len(schema_link_labels)
                else None,
            }
        )
    return expanded


def load_prepared_records(
    path: Path,
    limit: int | None = None,
    *,
    allow_oracle_plan: bool = False,
) -> list[dict[str, Any]]:
    records = []
    with path.open() as f:
        for line_index, line in enumerate(f):
            if limit is not None and len(records) >= limit:
                break
            record = json.loads(line)
            if record_uses_oracle_plan(record) and not allow_oracle_plan:
                raise ValueError(
                    f"{path} contains gold SQL-derived oracle planning hints. "
                    "Pass --allow-oracle-plan only for diagnostic upper-bound evaluation."
                )
            for expanded in expand_prepared_record(record, index=line_index):
                if limit is not None and len(records) >= limit:
                    break
                records.append(expanded)
    return records


def load_sparc_records(limit: int | None = None) -> list[dict[str, Any]]:
    split = f"validation[:{limit}]" if limit else "validation"
    dataset = load_dataset("jellyChiru/SParC", split=split)
    records = []
    for index, row in enumerate(dataset):
        records.append(
            {
                "id": index,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Database: {row['database_id']}\n\nQuestion:\n{row['question']}",
                    }
                ],
                "reference_sql": row["query"],
                "source": "jellyChiru/SParC",
            }
        )
    return records


def load_bird_records(limit: int | None = None) -> list[dict[str, Any]]:
    split = f"mini_dev_sqlite[:{limit}]" if limit else "mini_dev_sqlite"
    dataset = load_dataset("birdsql/bird_mini_dev", split=split)
    records = []
    for row in dataset:
        records.append(
            {
                "id": row["question_id"],
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Database: {row['db_id']}\n\n"
                            f"Evidence:\n{row.get('evidence') or ''}\n\n"
                            f"Question:\n{row['question']}"
                        ),
                    }
                ],
                "reference_sql": row["SQL"],
                "source": "birdsql/bird_mini_dev",
            }
        )
    return records


def load_benchmark_records(
    benchmark: str,
    *,
    input_path: Path | None,
    limit: int | None,
    allow_oracle_plan: bool = False,
) -> list[dict[str, Any]]:
    if benchmark == "prepared":
        if input_path is None:
            raise ValueError("--input is required for --benchmark prepared")
        return load_prepared_records(input_path, limit=limit, allow_oracle_plan=allow_oracle_plan)
    if benchmark == "sparc":
        return load_sparc_records(limit=limit)
    if benchmark == "bird_mini_dev":
        return load_bird_records(limit=limit)
    raise ValueError(f"unknown benchmark: {benchmark}")


def generate_sql(
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


def write_results(records: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def database_path_for_record(record: dict[str, Any], database_root: Path | None) -> Path | None:
    if database_root is None or not record.get("database_id"):
        return None
    database_id = record["database_id"]
    path = database_root / database_id / f"{database_id}.sqlite"
    return path if path.exists() else None


def run_eval(
    *,
    benchmark: str,
    endpoint: str,
    model_name: str,
    output: Path,
    input_path: Path | None,
    limit: int | None,
    database_root: Path | None,
    api_key: str,
    temperature: float,
    max_tokens: int,
    allow_oracle_plan: bool,
) -> int:
    client = OpenAI(base_url=endpoint, api_key=api_key)
    records = load_benchmark_records(
        benchmark,
        input_path=input_path,
        limit=limit,
        allow_oracle_plan=allow_oracle_plan,
    )
    if any(record.get("evaluation_mode") == ORACLE_PLANNER_DIAGNOSTIC for record in records):
        print(f"WARNING: {ORACLE_DIAGNOSTIC_WARNING}")
    results = []
    for record in records:
        raw_generation, generation_latency_ms = generate_sql(
            client,
            model_name=model_name,
            messages=enforce_sql_only_instruction(record["messages"]),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        generated_sql = extract_sql(raw_generation)
        database_path = database_path_for_record(record, database_root)
        score = score_single_turn(record["reference_sql"], generated_sql, database_path=database_path)
        results.append(
            {
                **record,
                "model_name": model_name,
                "raw_generation": raw_generation,
                "generated_sql": generated_sql,
                "generation_latency_ms": generation_latency_ms,
                "execution_score": score.execution_score,
                "strict_execution_score": score.strict_execution_score,
                "value_execution_score": score.value_execution_score,
                "order_sensitive": score.order_sensitive,
                "normalized_match": score.normalized_match,
                "syntax_valid": score.syntax_valid,
                "score_error": score.error,
                "database_path": str(database_path) if database_path else None,
            }
        )

    written = write_results(results, output)
    mean_score = (
        sum(result["execution_score"] for result in results) / len(results) if results else 0.0
    )
    print(f"Wrote {written} rows to {output}")
    print(f"Mean normalized/execution score: {mean_score:.3f}")
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/eval.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--allow-oracle-plan",
        action="store_true",
        help="Allow prepared inputs containing gold SQL-derived planning hints.",
    )
    args = parser.parse_args()

    return run_eval(
        benchmark=args.benchmark,
        endpoint=args.endpoint,
        model_name=args.model_name,
        output=args.output,
        input_path=args.input,
        limit=args.limit,
        database_root=args.database_root,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        allow_oracle_plan=args.allow_oracle_plan,
    )


if __name__ == "__main__":
    raise SystemExit(main())
