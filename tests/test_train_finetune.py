from __future__ import annotations

import json
from pathlib import Path

import pytest

from train.finetune import (
    build_formatting_func,
    build_sft_config,
    load_jsonl_dataset,
    oracle_diagnostic_row_count,
    require_expected_dataset_metadata,
    require_oracle_diagnostic_acknowledgement,
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


def test_oracle_diagnostic_row_count_counts_marked_rows(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "q"}],
                        "evaluation_mode": "oracle_planner_diagnostic",
                    }
                ),
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "q"}],
                        "evaluation_mode": "non_oracle_generation",
                    }
                ),
            ]
        )
        + "\n"
    )

    dataset = load_jsonl_dataset(path)

    assert oracle_diagnostic_row_count(dataset) == 1


def test_oracle_diagnostic_rows_require_explicit_acknowledgement(tmp_path) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "q"}],
                "evaluation_mode": "oracle_planner_diagnostic",
            }
        )
        + "\n"
    )
    dataset = load_jsonl_dataset(path)

    with pytest.raises(ValueError, match="allow-oracle-diagnostic-data"):
        require_oracle_diagnostic_acknowledgement(
            dataset,
            dataset_name="training",
            allow_oracle_diagnostic_data=False,
        )

    assert (
        require_oracle_diagnostic_acknowledgement(
            dataset,
            dataset_name="training",
            allow_oracle_diagnostic_data=True,
        )
        == 1
    )


def test_require_expected_dataset_metadata_accepts_matching_rows(tmp_path) -> None:
    path = tmp_path / "metric_dsl_train.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [{"role": "user", "content": "q"}],
                "training_target": "metric_dsl",
                "evaluation_mode": "metric_dsl",
                "benchmark": "synthetic_metric_dsl_bootstrap",
            }
        )
        + "\n"
    )

    dataset = load_jsonl_dataset(path)

    require_expected_dataset_metadata(
        dataset,
        dataset_name="training",
        expected_training_target="metric_dsl",
        expected_evaluation_mode="metric_dsl",
        expected_benchmark="synthetic_metric_dsl_bootstrap",
    )


def test_require_expected_dataset_metadata_rejects_mixed_or_wrong_values(tmp_path) -> None:
    path = tmp_path / "metric_dsl_train.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "q"}],
                        "training_target": "metric_dsl",
                        "evaluation_mode": "metric_dsl",
                        "benchmark": "synthetic_metric_dsl_bootstrap",
                    }
                ),
                json.dumps(
                    {
                        "messages": [{"role": "user", "content": "q"}],
                        "training_target": "direct_sql_control",
                        "evaluation_mode": "non_oracle_generation",
                        "benchmark": "metric_dsl_direct_sql",
                    }
                ),
            ]
        )
        + "\n"
    )

    dataset = load_jsonl_dataset(path)

    with pytest.raises(ValueError, match="expected training_target=metric_dsl"):
        require_expected_dataset_metadata(
            dataset,
            dataset_name="training",
            expected_training_target="metric_dsl",
        )

    with pytest.raises(ValueError, match="expected evaluation_mode=metric_dsl"):
        require_expected_dataset_metadata(
            dataset,
            dataset_name="training",
            expected_evaluation_mode="metric_dsl",
        )

    with pytest.raises(ValueError, match="expected benchmark=synthetic_metric_dsl_bootstrap"):
        require_expected_dataset_metadata(
            dataset,
            dataset_name="training",
            expected_benchmark="synthetic_metric_dsl_bootstrap",
        )


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
