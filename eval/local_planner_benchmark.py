"""Local generation for planner JSON predictions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.planner_predict import (
    JSON_PLANNER_PREDICTIONS_SOURCE,
    parse_json_plan_prediction,
    planner_messages_for_record,
)
from eval.run_eval import load_prepared_records


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
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def generate_local_text(
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


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def run_local_planner_benchmark(
    *,
    model_name: str,
    adapter_path: Path | None,
    input_path: Path,
    output_path: Path,
    max_new_tokens: int,
    max_memory_gb: int | None,
) -> int:
    records = load_prepared_records(input_path, limit=None, allow_oracle_plan=False)
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        adapter_path=adapter_path,
        max_memory_gb=max_memory_gb,
    )
    rows: list[dict[str, Any]] = []
    for record in records:
        messages = planner_messages_for_record(record)
        raw_output, latency_ms = generate_local_text(
            model,
            tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
        rows.append(
            {
                "id": str(record["id"]),
                "dialog_id": str(record.get("dialog_id") or ""),
                "turn_index": int(record.get("turn_index") or 0),
                "database_id": record.get("database_id"),
                "model_name": model_name,
                "prediction_source": JSON_PLANNER_PREDICTIONS_SOURCE,
                "raw_planner_output": raw_output,
                "predicted_plan": parse_json_plan_prediction(
                    raw_output,
                    prediction_source=JSON_PLANNER_PREDICTIONS_SOURCE,
                ),
                "planner_latency_ms": latency_ms,
            }
        )
    written = _write_jsonl(rows, output_path)
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    args = parser.parse_args()
    return run_local_planner_benchmark(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        input_path=args.input,
        output_path=args.output,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
    )


if __name__ == "__main__":
    raise SystemExit(main())
