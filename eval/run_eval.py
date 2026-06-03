"""
Run SQL benchmarks against an OpenAI-compatible chat endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from datasets import load_dataset
from openai import OpenAI

from eval.ragas_metrics import extract_sql, score_single_turn
from eval.result_manifest import build_result_manifest, write_result_manifest

BENCHMARKS = ["prepared", "sparc", "bird_mini_dev"]
SQL_ONLY_INSTRUCTION = (
    "Return only one SQL query. Do not explain, reason step by step, use markdown, or include prose."
)
STRUCTURED_BRIEF_SQL_MARKER_RE = re.compile(r"(?im)^SQL:\s*")
ORACLE_PLAN_MARKERS = (
    "Oracle SQL planning hints",
    "SQL planning hints:",
)
SPLIT_PROVENANCE_FIELDS = (
    "split_id",
    "split_role",
    "split_row_id",
    "split_source_path",
    "split_source_sha256",
    "split_row_ids_sha256",
)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _usage_value(usage: Any, *names: str) -> int | None:
    for name in names:
        if usage is None:
            return None
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if value is not None:
            return int(value)
    return None


def usage_from_response(response: Any) -> dict[str, int | None]:
    """Extract OpenAI-compatible token usage when the endpoint returns it."""

    usage = getattr(response, "usage", None)
    prompt_tokens = _usage_value(usage, "prompt_tokens", "input_tokens")
    completion_tokens = _usage_value(usage, "completion_tokens", "output_tokens")
    total_tokens = _usage_value(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def estimate_generation_cost_usd(
    usage: dict[str, int | None],
    *,
    prompt_token_cost_usd_per_1k: float,
    completion_token_cost_usd_per_1k: float,
) -> float | None:
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if prompt_tokens is None and completion_tokens is None:
        return None
    return (
        ((prompt_tokens or 0) / 1000.0 * prompt_token_cost_usd_per_1k)
        + ((completion_tokens or 0) / 1000.0 * completion_token_cost_usd_per_1k)
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


def messages_for_generation(
    record: dict[str, Any], *, prompt_variant: str | None = None
) -> list[dict[str, str]]:
    """Return the prompt sent to the model for one expanded eval turn."""

    messages = record["messages"]
    if prompt_variant == "structured_brief_sql":
        return [dict(message) for message in messages]
    return enforce_sql_only_instruction(messages)


def extract_generated_sql(raw_generation: str, *, prompt_variant: str | None = None) -> str:
    """Extract SQL from one model generation under the active prompt contract."""

    generation = raw_generation
    if prompt_variant == "structured_brief_sql":
        marker = STRUCTURED_BRIEF_SQL_MARKER_RE.search(generation)
        if marker:
            generation = generation[marker.end() :]
    return extract_sql(generation)


def prompt_variant_for_record(
    record: dict[str, Any], *, prompt_variant: str | None = None
) -> str | None:
    """Return CLI prompt variant, falling back to self-describing prepared rows."""

    return prompt_variant or record.get("prompt_variant")


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
    evaluation_mode = record.get("evaluation_mode") or "unknown"
    split_provenance = {
        field: record[field] for field in SPLIT_PROVENANCE_FIELDS if record.get(field) is not None
    }
    turn_count = len(assistant_indices)
    expanded = []
    for turn_index, assistant_index in enumerate(assistant_indices):
        schema_label = (
            schema_link_labels[turn_index] if turn_index < len(schema_link_labels) else None
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
                **split_provenance,
                "history_policy": history_policy,
                "evaluation_mode": evaluation_mode,
                "prompt_variant": record.get("prompt_variant"),
                "uses_oracle_planning_hints": False,
                "semantic_context_pruned_by_oracle_labels": False,
                "gold_plan": schema_label,
                "schema_link_labels": schema_label,
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
                    "Oracle planner diagnostics have been removed from active evaluation."
                )
            if record_uses_oracle_plan(record) and allow_oracle_plan:
                raise ValueError("oracle planner diagnostics have been removed from active evaluation")
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
    raw_generation, latency_ms, _usage = generate_sql_with_usage(
        client,
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return raw_generation, latency_ms


def generate_sql_with_usage(
    client: OpenAI,
    *,
    model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> tuple[str, float, dict[str, int | None]]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    return response.choices[0].message.content or "", latency_ms, usage_from_response(response)


def write_results(records: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def summarize_eval_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metrics for result manifests and comparison reports."""

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
    split_rows = [result for result in results if result.get("split_row_id")]
    if split_rows:
        split_row_ids = [str(result["split_row_id"]) for result in split_rows]
        split_eval_turn_ids = [
            f"{result['split_row_id']}:{result.get('turn_index', 0)}" for result in split_rows
        ]
        unique_split_row_ids = _ordered_unique(split_row_ids)
        metrics["split_ids"] = dict(
            Counter(str(result.get("split_id", "unknown")) for result in split_rows)
        )
        metrics["split_roles"] = dict(
            Counter(str(result.get("split_role", "unknown")) for result in split_rows)
        )
        metrics["split_row_count"] = len(unique_split_row_ids)
        metrics["split_eval_turn_count"] = len(split_eval_turn_ids)
        metrics["split_row_ids_sha256"] = _sha256_json(unique_split_row_ids)
        metrics["split_eval_turn_ids_sha256"] = _sha256_json(split_eval_turn_ids)
        source_sha256s = _ordered_unique(
            str(result["split_source_sha256"])
            for result in split_rows
            if result.get("split_source_sha256")
        )
        if source_sha256s:
            metrics["split_source_sha256s"] = source_sha256s
    token_rows = [result for result in results if result.get("total_tokens") is not None]
    if token_rows:
        prompt_tokens = [int(result.get("prompt_tokens") or 0) for result in token_rows]
        completion_tokens = [int(result.get("completion_tokens") or 0) for result in token_rows]
        total_tokens = [int(result.get("total_tokens") or 0) for result in token_rows]
        estimated_costs = [
            float(result["estimated_generation_cost_usd"])
            for result in token_rows
            if result.get("estimated_generation_cost_usd") is not None
        ]
        metrics["token_usage_available_rows"] = len(token_rows)
        metrics["total_prompt_tokens"] = sum(prompt_tokens)
        metrics["total_completion_tokens"] = sum(completion_tokens)
        metrics["total_tokens"] = sum(total_tokens)
        metrics["mean_prompt_tokens"] = sum(prompt_tokens) / len(token_rows)
        metrics["mean_completion_tokens"] = sum(completion_tokens) / len(token_rows)
        metrics["mean_total_tokens"] = sum(total_tokens) / len(token_rows)
        if estimated_costs:
            metrics["total_estimated_generation_cost_usd"] = sum(estimated_costs)
            metrics["mean_estimated_generation_cost_usd"] = sum(estimated_costs) / len(
                estimated_costs
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
    prompt_token_cost_usd_per_1k: float = 0.0,
    completion_token_cost_usd_per_1k: float = 0.0,
) -> int:
    client = OpenAI(base_url=endpoint, api_key=api_key)
    records = load_benchmark_records(
        benchmark,
        input_path=input_path,
        limit=limit,
        allow_oracle_plan=allow_oracle_plan,
    )
    results = []
    for record in records:
        active_prompt_variant = prompt_variant_for_record(record, prompt_variant=prompt_variant)
        raw_generation, generation_latency_ms, token_usage = generate_sql_with_usage(
            client,
            model_name=model_name,
            messages=messages_for_generation(record, prompt_variant=active_prompt_variant),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        estimated_cost = estimate_generation_cost_usd(
            token_usage,
            prompt_token_cost_usd_per_1k=prompt_token_cost_usd_per_1k,
            completion_token_cost_usd_per_1k=completion_token_cost_usd_per_1k,
        )
        generated_sql = extract_generated_sql(
            raw_generation, prompt_variant=active_prompt_variant
        )
        database_path = database_path_for_record(record, database_root)
        score = score_single_turn(record["reference_sql"], generated_sql, database_path=database_path)
        results.append(
            {
                **record,
                "model_name": model_name,
                "prompt_variant": active_prompt_variant,
                "raw_generation": raw_generation,
                "generated_sql": generated_sql,
                "generation_latency_ms": generation_latency_ms,
                "prompt_tokens": token_usage["prompt_tokens"],
                "completion_tokens": token_usage["completion_tokens"],
                "total_tokens": token_usage["total_tokens"],
                "estimated_generation_cost_usd": estimated_cost,
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
    metrics["token_cost_rates_usd_per_1k"] = {
        "prompt": prompt_token_cost_usd_per_1k,
        "completion": completion_token_cost_usd_per_1k,
    }
    if manifest_output is None:
        manifest_output = output.with_suffix(".manifest.json")
    evaluation_modes = metrics.get("evaluation_modes", {})
    evaluation_mode = (
        next(iter(evaluation_modes))
        if len(evaluation_modes) == 1
        else ",".join(sorted(evaluation_modes)) or "unknown"
    )
    prompt_variants = {
        str(result.get("prompt_variant"))
        for result in results
        if result.get("prompt_variant") is not None
    }
    manifest_prompt_variant = (
        prompt_variant
        if prompt_variant is not None
        else next(iter(prompt_variants))
        if len(prompt_variants) == 1
        else None
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
        prompt_variant=manifest_prompt_variant,
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
    parser.add_argument("--prompt-token-cost-usd-per-1k", type=float, default=0.0)
    parser.add_argument("--completion-token-cost-usd-per-1k", type=float, default=0.0)
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
        allow_oracle_plan=False,
        manifest_output=args.manifest_output,
        prompt_variant=args.prompt_variant,
        prompt_token_cost_usd_per_1k=args.prompt_token_cost_usd_per_1k,
        completion_token_cost_usd_per_1k=args.completion_token_cost_usd_per_1k,
        command=sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
