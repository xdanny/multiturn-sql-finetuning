"""Local Transformers/PEFT chat generation helpers."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

GenerateFn = Callable[[list[dict[str, str]]], tuple[str, float]]


def chat_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    """Format chat messages without replacing the caller's system prompt."""

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
    """Build a local Transformers/PEFT generator for chat-completion style rows."""

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
        prompt = chat_prompt(tokenizer, messages)
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
