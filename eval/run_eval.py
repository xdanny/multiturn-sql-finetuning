"""
Run SQL benchmarks against an OpenAI-compatible chat endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI

from data.plan_contract import (
    ORACLE_PLANNER_DIAGNOSTIC,
    PREDICTED_PLANNER,
    predicted_planning_hint_from_plan,
    validate_prepared_record_contract,
)
from data.prepare import ORACLE_DIAGNOSTIC_WARNING
from eval.ragas_metrics import extract_sql, score_single_turn
from eval.result_manifest import build_result_manifest, write_result_manifest

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


def add_predicted_plan_to_last_user_message(
    messages: list[dict[str, str]], predicted_plan: dict[str, Any]
) -> list[dict[str, str]]:
    """Append non-oracle planner output to the current user turn."""

    updated = [dict(message) for message in messages]
    hint = predicted_planning_hint_from_plan(predicted_plan)
    for message in reversed(updated):
        if message.get("role") == "user":
            message["content"] = f"{message['content']}\n\n{hint}"
            return updated
    return [*updated, {"role": "user", "content": hint}]


def messages_for_generation(record: dict[str, Any]) -> list[dict[str, str]]:
    """Return the prompt sent to the model for one expanded eval turn."""

    messages = record["messages"]
    if record.get("evaluation_mode") == PREDICTED_PLANNER:
        predicted_plan = record.get("predicted_plan")
        if not predicted_plan:
            raise ValueError("predicted_planner eval records must include predicted_plan")
        messages = add_predicted_plan_to_last_user_message(messages, predicted_plan)
    return enforce_sql_only_instruction(messages)


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
    history_policy = record.get("history_policy")
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
        predicted_plan = (
            predicted_plans[turn_index] if turn_index < len(predicted_plans) else None
        )
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
                "history_policy": history_policy,
                "evaluation_mode": evaluation_mode,
                "planning_label_source": record.get("planning_label_source"),
                "uses_oracle_planning_hints": uses_oracle_planning_hints,
                "semantic_context_pruned_by_oracle_labels": semantic_context_pruned_by_oracle_labels,
                "oracle_diagnostic_warning": record.get("oracle_diagnostic_warning"),
                "gold_plan": gold_plans[turn_index] if turn_index < len(gold_plans) else None,
                "predicted_plan": predicted_plan,
                "predicted_plan_source": (
                    predicted_plan.get("prediction_source")
                    if isinstance(predicted_plan, dict) and predicted_plan.get("prediction_source")
                    else record.get("predicted_plan_source")
                ),
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
            if record.get("evaluation_mode"):
                validate_prepared_record_contract(record)
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


def summarize_eval_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics for result manifests and claim ledgers."""

    if not results:
        return {}
    strict_scores = [
        result["execution_score"]
        if result.get("strict_execution_score") is None
        else result["strict_execution_score"]
        for result in results
    ]
    value_scores = [
        result["execution_score"]
        if result.get("value_execution_score") is None
        else result["value_execution_score"]
        for result in results
    ]
    mean_generation_latency_ms = sum(
        float(result.get("generation_latency_ms") or 0.0) for result in results
    ) / len(results)
    metrics = {
        "execution_accuracy": sum(result["execution_score"] for result in results) / len(results),
        "strict_execution_accuracy": sum(strict_scores) / len(results),
        "value_execution_accuracy": sum(value_scores) / len(results),
        "syntax_accuracy": sum(float(bool(result.get("syntax_valid"))) for result in results)
        / len(results),
        "mean_generation_latency_ms": mean_generation_latency_ms,
        "mean_latency_ms": mean_generation_latency_ms,
        "sources": dict(Counter(str(result.get("source", "unknown")) for result in results)),
        "evaluation_modes": dict(
            Counter(str(result.get("evaluation_mode", "unknown")) for result in results)
        ),
    }
    history_policies = Counter(
        str(result.get("history_policy"))
        for result in results
        if result.get("history_policy")
    )
    if history_policies:
        metrics["history_policies"] = dict(history_policies)
        if len(history_policies) == 1:
            metrics["history_policy"] = next(iter(history_policies))
    if any(result.get("dialog_id") for result in results):
        metrics["dialog_count"] = len({result.get("dialog_id") for result in results})
        per_dialog: dict[str, list[float]] = {}
        for result in results:
            if result.get("dialog_id"):
                per_dialog.setdefault(str(result["dialog_id"]), []).append(result["execution_score"])
        metrics["interaction_match_rate"] = (
            sum(1.0 for scores in per_dialog.values() if all(score == 1.0 for score in scores))
            / len(per_dialog)
            if per_dialog
            else 0.0
        )
    return metrics


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
    manifest_output: Path | None = None,
    prompt_variant: str | None = None,
    command: Sequence[str] | None = None,
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
            messages=messages_for_generation(record),
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
    metrics = summarize_eval_metrics(results)
    if manifest_output is None:
        manifest_output = output.with_suffix(".manifest.json")
    evaluation_modes = metrics.get("evaluation_modes", {})
    evaluation_mode = (
        next(iter(evaluation_modes))
        if len(evaluation_modes) == 1
        else ",".join(sorted(evaluation_modes)) or "unknown"
    )
    manifest = build_result_manifest(
        run_id=output.stem,
        benchmark=benchmark,
        input_path=input_path,
        output_path=output,
        model_name=model_name,
        endpoint=endpoint,
        evaluation_mode=evaluation_mode,
        oracle_allowed=allow_oracle_plan,
        prompt_variant=prompt_variant,
        database_root=database_root,
        command=list(command or sys.argv),
        row_count=written,
        metrics=metrics,
    )
    write_result_manifest(manifest, manifest_output)
    mean_score = metrics.get("execution_accuracy", 0.0)
    print(f"Wrote {written} rows to {output}")
    print(f"Wrote result manifest to {manifest_output}")
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
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--prompt-variant", default=None)
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
        manifest_output=args.manifest_output,
        prompt_variant=args.prompt_variant,
        command=sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
