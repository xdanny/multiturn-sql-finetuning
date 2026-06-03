from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.finetune import (
    build_formatting_func,
    build_sft_config,
    load_jsonl_dataset,
)


def _write_train_file(path: Path) -> None:
    row = {
        "messages": [
            {"role": "system", "content": "You write SQL."},
            {"role": "user", "content": "List ids."},
            {"role": "assistant", "content": "SELECT id FROM t;"},
        ]
    }
    path.write_text(json.dumps(row) + "\n")


def test_load_jsonl_dataset_requires_messages(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"text": "missing messages"}) + "\n")

    with pytest.raises(ValueError, match="messages"):
        load_jsonl_dataset(path)


def test_load_jsonl_dataset_accepts_chat_messages(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    _write_train_file(path)

    dataset = load_jsonl_dataset(path)

    assert len(dataset) == 1
    assert dataset[0]["messages"][1]["content"] == "List ids."


def test_build_sft_config_applies_smoke_test_overrides() -> None:
    config = {
        "model": {"max_seq_length": 1024},
        "training": {
            "output_dir": "outputs/test",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "learning_rate": 1e-4,
            "eval_strategy": "steps",
            "report_to": "wandb",
        },
    }

    sft_config = build_sft_config(
        config,
        max_steps=1,
        output_dir=Path("outputs/override"),
        report_to="none",
    )

    assert sft_config.output_dir == "outputs/override"
    assert sft_config.max_steps == 1
    assert sft_config.report_to == []
    assert sft_config.max_length == 1024
    assert sft_config.eval_strategy.value == "no"


def test_build_sft_config_keeps_eval_strategy_with_eval_dataset() -> None:
    config = {
        "model": {"max_seq_length": 1024},
        "training": {"output_dir": "outputs/test", "eval_strategy": "steps", "eval_steps": 10},
    }

    sft_config = build_sft_config(config, has_eval_dataset=True)

    assert sft_config.eval_strategy.value == "steps"
    assert sft_config.eval_steps == 10


def test_build_formatting_func_uses_chat_template() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is False
            assert add_generation_prompt is False
            return "|".join(message["content"] for message in messages)

    formatter = build_formatting_func(Tokenizer())

    assert formatter({"messages": [{"role": "user", "content": "Question"}]}) == ["Question"]


def test_build_formatting_func_handles_batched_messages() -> None:
    class Tokenizer:
        chat_template = ""

    formatter = build_formatting_func(Tokenizer())

    assert formatter(
        {
            "messages": [
                [{"role": "user", "content": "One"}],
                [{"role": "assistant", "content": "Two"}],
            ]
        }
    ) == ["user: One", "assistant: Two"]
