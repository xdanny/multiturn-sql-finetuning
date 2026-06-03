"""
Evaluate multi-turn prepared dialogs with model-generated SQL history.

Unlike eval.run_eval, this runner does not teacher-force prior assistant SQL.
Each generated SQL query becomes the assistant message seen by later turns in
the same dialog. Reference SQL remains available only for scoring.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

from eval.ragas_metrics import extract_sql, score_single_turn
from eval.result_manifest import build_result_manifest, write_result_manifest
from eval.run_eval import (
    database_path_for_record,
    estimate_generation_cost_usd,
    generate_sql_with_usage,
    messages_for_generation,
    record_uses_oracle_plan,
    summarize_eval_metrics,
    write_results,
)

MODEL_GENERATED_SQL_ROLLOUT = "model_generated_sql_rollout"
GenerationResult = tuple[str, float] | tuple[str, float, dict[str, int | None]]
GenerateFn = Callable[[list[dict[str, str]]], GenerationResult]


def _normalize_generation_result(
    result: GenerationResult,
) -> tuple[str, float, dict[str, int | None]]:
    raw_generation = result[0]
    generation_latency_ms = result[1]
    token_usage = result[2] if len(result) == 3 else {}
    return (
        raw_generation,
        generation_latency_ms,
        {
            "prompt_tokens": token_usage.get("prompt_tokens"),
            "completion_tokens": token_usage.get("completion_tokens"),
            "total_tokens": token_usage.get("total_tokens"),
        },
    )


def load_rollout_prepared_records(
    path: Path,
    *,
    limit_dialogs: int | None = None,
    allow_oracle_plan: bool = False,
) -> list[dict[str, Any]]:
    """Load dialog-level prepared records for generated-history rollout."""

    records = []
    with path.open() as f:
        for line in f:
            if limit_dialogs is not None and len(records) >= limit_dialogs:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if record_uses_oracle_plan(record) and not allow_oracle_plan:
                raise ValueError(
                    f"{path} contains gold SQL-derived oracle planning hints. "
                    "Oracle planner diagnostics have been removed from rollout evaluation."
                )
            if record_uses_oracle_plan(record) and allow_oracle_plan:
                raise ValueError("oracle planner diagnostics have been removed from rollout evaluation")
            records.append(record)
    return records


def _assistant_turn_payloads(record: dict[str, Any]) -> list[dict[str, Any]]:
    payloads = []
    turn_index = 0
    schema_link_labels = record.get("schema_link_labels") or []
    for message_index, message in enumerate(record.get("messages", [])):
        if message.get("role") != "assistant":
            continue
        payloads.append(
            {
                "message_index": message_index,
                "turn_index": turn_index,
                "reference_sql": message.get("content", ""),
                "schema_link_labels": schema_link_labels[turn_index]
                if turn_index < len(schema_link_labels)
                else None,
            }
        )
        turn_index += 1
    if not payloads:
        raise ValueError("prepared record has no assistant reference SQL")
    return payloads


def _rollout_record(
    record: dict[str, Any],
    *,
    index: int,
    turn_payload: dict[str, Any],
    turn_count: int,
    history: list[dict[str, str]],
) -> dict[str, Any]:
    dialog_id = str(record.get("dialog_id") or record.get("id") or f"prepared-{index}")
    return {
        "id": f"{dialog_id}:{turn_payload['turn_index']}",
        "dialog_id": dialog_id,
        "turn_index": turn_payload["turn_index"],
        "turn_count": turn_count,
        "messages": [dict(message) for message in history],
        "reference_sql": turn_payload["reference_sql"],
        "source": record.get("source", "prepared"),
        "database_id": record.get("database_id"),
        "history_policy": MODEL_GENERATED_SQL_ROLLOUT,
        "original_history_policy": record.get("history_policy"),
        "evaluation_mode": record.get("evaluation_mode") or "unknown",
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
        "gold_plan": turn_payload.get("schema_link_labels"),
        "schema_link_labels": turn_payload.get("schema_link_labels"),
    }


def evaluate_rollout_records(
    records: list[dict[str, Any]],
    *,
    generate_fn: GenerateFn,
    model_name: str,
    database_root: Path | None,
    prompt_token_cost_usd_per_1k: float = 0.0,
    completion_token_cost_usd_per_1k: float = 0.0,
) -> list[dict[str, Any]]:
    """Run generated-history rollout over prepared dialog records."""

    results = []
    for record_index, record in enumerate(records):
        messages = record.get("messages", [])
        turn_payloads = _assistant_turn_payloads(record)
        turn_by_message_index = {
            payload["message_index"]: payload for payload in turn_payloads
        }
        seeded_history_turn_index = record.get("seeded_failure_turn_index")
        history: list[dict[str, str]] = []
        for message_index, message in enumerate(messages):
            if message.get("role") != "assistant":
                history.append(dict(message))
                continue

            payload = turn_by_message_index[message_index]
            if payload["turn_index"] == seeded_history_turn_index:
                history.append(dict(message))
                continue
            rollout_record = _rollout_record(
                record,
                index=record_index,
                turn_payload=payload,
                turn_count=len(turn_payloads),
                history=history,
            )
            raw_generation, generation_latency_ms, token_usage = _normalize_generation_result(
                generate_fn(messages_for_generation(rollout_record))
            )
            estimated_cost = estimate_generation_cost_usd(
                token_usage,
                prompt_token_cost_usd_per_1k=prompt_token_cost_usd_per_1k,
                completion_token_cost_usd_per_1k=completion_token_cost_usd_per_1k,
            )
            generated_sql = extract_sql(raw_generation)
            database_path = database_path_for_record(rollout_record, database_root)
            score = score_single_turn(
                rollout_record["reference_sql"],
                generated_sql,
                database_path=database_path,
            )
            result = {
                **rollout_record,
                "model_name": model_name,
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
            results.append(result)
            history.append({"role": "assistant", "content": generated_sql})
    return results


def run_rollout_eval(
    *,
    endpoint: str,
    model_name: str,
    input_path: Path,
    output: Path,
    limit_dialogs: int | None,
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
    records = load_rollout_prepared_records(
        input_path,
        limit_dialogs=limit_dialogs,
        allow_oracle_plan=allow_oracle_plan,
    )
    def endpoint_generate(messages: list[dict[str, str]]) -> GenerationResult:
        return generate_sql_with_usage(
            client,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    results = evaluate_rollout_records(
        records,
        generate_fn=endpoint_generate,
        model_name=model_name,
        database_root=database_root,
        prompt_token_cost_usd_per_1k=prompt_token_cost_usd_per_1k,
        completion_token_cost_usd_per_1k=completion_token_cost_usd_per_1k,
    )
    written = write_results(results, output)
    metrics = summarize_eval_metrics(results)
    metrics["history_policy"] = MODEL_GENERATED_SQL_ROLLOUT
    metrics["rollout_dialog_count"] = len(records)
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
    manifest = build_result_manifest(
        run_id=output.stem,
        benchmark="prepared_rollout",
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
    print(f"Wrote {written} rollout rows to {output}")
    print(f"Wrote rollout result manifest to {manifest_output}")
    print(f"Mean rollout execution score: {metrics.get('execution_accuracy', 0.0):.3f}")
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/rollout_eval.jsonl"))
    parser.add_argument("--limit-dialogs", type=int, default=None)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--prompt-variant", default=None)
    parser.add_argument("--prompt-token-cost-usd-per-1k", type=float, default=0.0)
    parser.add_argument("--completion-token-cost-usd-per-1k", type=float, default=0.0)
    args = parser.parse_args()
    return run_rollout_eval(
        endpoint=args.endpoint,
        model_name=args.model_name,
        input_path=args.input,
        output=args.output,
        limit_dialogs=args.limit_dialogs,
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
