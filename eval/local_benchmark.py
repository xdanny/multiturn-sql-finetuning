"""
Run local 5090 benchmarks for base Qwen and fine-tuned LoRA adapters.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.ragas_metrics import extract_sql, score_single_turn
from eval.run_eval import (
    SQL_ONLY_INSTRUCTION as _SQL_ONLY_INSTRUCTION,
)
from eval.run_eval import (
    database_path_for_record,
    enforce_sql_only_instruction,
    load_benchmark_records,
    messages_for_generation,
    write_results,
)

SQL_ONLY_INSTRUCTION = _SQL_ONLY_INSTRUCTION


def load_model_and_tokenizer(
    *,
    model_name: str,
    adapter_path: Path | None,
    max_memory_gb: int | None = None,
):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if max_memory_gb is not None:
        model_kwargs["max_memory"] = {0: f"{max_memory_gb}GiB", "cpu": "64GiB"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def prompt_from_messages(tokenizer, messages: list[dict[str, str]]) -> str:
    messages = enforce_sql_only_instruction(messages)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def generate_local_sql(
    model,
    tokenizer,
    *,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> tuple[str, float]:
    prompt = prompt_from_messages(tokenizer, messages)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency_ms = (time.perf_counter() - started) * 1000
    new_tokens = output[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, latency_ms


def run_local_benchmark(
    *,
    model_name: str,
    adapter_path: Path | None,
    benchmark: str,
    input_path: Path | None,
    output: Path,
    limit: int | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    database_root: Path | None,
) -> int:
    records = load_benchmark_records(
        benchmark,
        input_path=input_path,
        limit=limit,
    )
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        adapter_path=adapter_path,
        max_memory_gb=max_memory_gb,
    )

    results = []
    for record in records:
        raw_generation, generation_latency_ms = generate_local_sql(
            model,
            tokenizer,
            messages=messages_for_generation(record),
            max_new_tokens=max_new_tokens,
        )
        generated_sql = extract_sql(raw_generation)
        database_path = database_path_for_record(record, database_root)
        score = score_single_turn(record["reference_sql"], generated_sql, database_path=database_path)
        results.append(
            {
                **record,
                "model_name": model_name if adapter_path is None else f"{model_name}+{adapter_path}",
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
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--benchmark", choices=["prepared", "sparc", "bird_mini_dev"], required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    parser.add_argument("--database-root", type=Path, default=None)
    args = parser.parse_args()

    return run_local_benchmark(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        benchmark=args.benchmark,
        input_path=args.input,
        output=args.output,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
        database_root=args.database_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
