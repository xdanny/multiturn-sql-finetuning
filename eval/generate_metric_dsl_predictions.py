"""Generate metric-DSL or direct-SQL predictions from paired input rows."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from openai import OpenAI

from eval.ragas_metrics import extract_sql

GenerateFn = Callable[[list[dict[str, str]]], tuple[str, float]]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _validate_no_prompt_leakage(row: dict[str, Any]) -> None:
    if row.get("reference_sql_visible_to_model") or row.get("scoring_fields_visible_to_model"):
        raise ValueError(f"{row.get('id')} exposes scorer fields to the model prompt")
    prompt = json.dumps(row.get("messages") or [], sort_keys=True)
    for field in ("reference_sql", "gold_dsl"):
        value = row.get(field)
        if value and str(value) in prompt:
            raise ValueError(f"{row.get('id')} leaks {field} into the model prompt")


def generate_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
    generate_fn: GenerateFn,
) -> list[dict[str, Any]]:
    """Return prediction rows that preserve held-out scorer fields."""

    predictions = []
    for row in rows:
        _validate_no_prompt_leakage(row)
        messages = [dict(message) for message in row["messages"]]
        raw_generation, generation_latency_ms = generate_fn(messages)
        generation_target = row.get("generation_target")
        prediction = {
            **row,
            "model_name": model_name,
            "raw_generation": raw_generation,
            "generation_latency_ms": generation_latency_ms,
        }
        if generation_target == "metric_dsl":
            prediction["predicted_dsl"] = raw_generation.strip()
        elif generation_target == "direct_sql":
            prediction["generated_sql"] = extract_sql(raw_generation)
        else:
            raise ValueError(f"unknown generation_target: {generation_target}")
        predictions.append(prediction)
    return predictions


def endpoint_generate_fn(
    *,
    endpoint: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> GenerateFn:
    """Build an OpenAI-compatible chat completion generator."""

    client = OpenAI(base_url=endpoint, api_key=api_key)

    def generate(messages: list[dict[str, str]]) -> tuple[str, float]:
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

    return generate


def _chat_prompt(tokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def local_adapter_generate_fn(
    *,
    model_name: str,
    adapter_path: Path | None,
    max_tokens: int,
    max_memory_gb: int | None,
) -> GenerateFn:
    """Build a local Transformers/PEFT generator for metric fixture predictions."""

    import torch  # noqa: PLC0415
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path or model_name,
        trust_remote_code=True,
    )
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

    def generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        prompt = _chat_prompt(tokenizer, messages)
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - started) * 1000
        new_tokens = output[0, inputs["input_ids"].shape[-1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, latency_ms

    return generate


def run_generate_metric_dsl_predictions(
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    generate_fn: GenerateFn,
) -> int:
    rows = _load_jsonl(input_path)
    predictions = generate_prediction_rows(
        rows,
        model_name=model_name,
        generate_fn=generate_fn,
    )
    written = _write_jsonl(output_path, predictions)
    print(f"Wrote {written} prediction rows to {output_path}")
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--backend", choices=["endpoint", "local"], default="endpoint")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    args = parser.parse_args()
    if args.backend == "local":
        generate_fn = local_adapter_generate_fn(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            max_tokens=args.max_tokens,
            max_memory_gb=args.max_memory_gb,
        )
    else:
        generate_fn = endpoint_generate_fn(
            endpoint=args.endpoint,
            api_key=args.api_key,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

    return run_generate_metric_dsl_predictions(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        generate_fn=generate_fn,
    )


if __name__ == "__main__":
    raise SystemExit(main())
