"""
Qwen bf16 LoRA fine-tuning on SQL chat data with Unsloth + TRL SFTTrainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset

from data.plan_contract import ORACLE_PLANNER_DIAGNOSTIC
from data.prepare import ORACLE_DIAGNOSTIC_WARNING


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def load_jsonl_dataset(path: Path) -> Dataset:
    if not path.exists():
        raise FileNotFoundError(f"training data not found: {path}")

    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"training data is empty: {path}")
    if any("messages" not in row for row in rows):
        raise ValueError(f"training data must contain a 'messages' column: {path}")
    return Dataset.from_list(rows)


def oracle_diagnostic_row_count(dataset: Dataset) -> int:
    """Count rows that include oracle planning hints or oracle-pruned context."""

    if "evaluation_mode" not in dataset.column_names:
        return 0
    return sum(1 for mode in dataset["evaluation_mode"] if mode == ORACLE_PLANNER_DIAGNOSTIC)


def require_oracle_diagnostic_acknowledgement(
    dataset: Dataset,
    *,
    dataset_name: str,
    allow_oracle_diagnostic_data: bool,
) -> int:
    """Reject oracle-labelled rows unless the run explicitly opts into diagnostics."""

    oracle_rows = oracle_diagnostic_row_count(dataset)
    if oracle_rows and not allow_oracle_diagnostic_data:
        raise ValueError(
            f"{oracle_rows}/{len(dataset)} {dataset_name} rows are oracle planner diagnostics. "
            "Pass --allow-oracle-diagnostic-data only when the run is explicitly a "
            "teacher-forced diagnostic, not a production-style fine-tune."
        )
    return oracle_rows


def _unique_non_null_values(dataset: Dataset, column_name: str) -> set[str]:
    if column_name not in dataset.column_names:
        return set()
    return {str(value) for value in dataset[column_name] if value is not None}


def require_expected_dataset_metadata(
    dataset: Dataset,
    *,
    dataset_name: str,
    expected_training_target: str | None = None,
    expected_evaluation_mode: str | None = None,
    expected_benchmark: str | None = None,
) -> None:
    """Reject rows whose declared experiment metadata does not match the run contract."""

    expected_fields = [
        ("training_target", expected_training_target),
        ("evaluation_mode", expected_evaluation_mode),
        ("benchmark", expected_benchmark),
    ]
    for field_name, expected_value in expected_fields:
        if expected_value is None:
            continue
        values = _unique_non_null_values(dataset, field_name)
        if values != {expected_value}:
            actual = ", ".join(sorted(values)) if values else "<missing>"
            raise ValueError(
                f"{dataset_name} rows expected {field_name}={expected_value}, got {actual}"
            )


def build_sft_config(
    config: dict[str, Any],
    *,
    has_eval_dataset: bool = False,
    max_steps: int | None = None,
    output_dir: Path | None = None,
    report_to: str | None = None,
):
    from transformers.utils import is_torch_bf16_gpu_available  # noqa: PLC0415
    from trl import SFTConfig  # noqa: PLC0415

    training_cfg = config["training"]
    model_cfg = config["model"]
    bf16 = training_cfg.get("bf16", True)
    tf32 = training_cfg.get("tf32", True)
    use_cpu = training_cfg.get("use_cpu")
    if use_cpu is None and bf16 and not is_torch_bf16_gpu_available():
        use_cpu = True
        bf16 = False
        tf32 = False
    kwargs = {
        "output_dir": str(output_dir) if output_dir is not None else training_cfg["output_dir"],
        "num_train_epochs": training_cfg.get("num_train_epochs", 1),
        "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 1),
        "learning_rate": training_cfg.get("learning_rate", 2e-4),
        "lr_scheduler_type": training_cfg.get("lr_scheduler_type", "cosine"),
        "warmup_steps": training_cfg.get("warmup_steps", 0),
        "weight_decay": training_cfg.get("weight_decay", 0.0),
        "max_grad_norm": training_cfg.get("max_grad_norm", 1.0),
        "optim": training_cfg.get("optim", "adamw_torch"),
        "bf16": bf16,
        "tf32": tf32,
        "use_cpu": use_cpu,
        "logging_steps": training_cfg.get("logging_steps", 10),
        "save_strategy": training_cfg.get("save_strategy", "steps"),
        "save_steps": training_cfg.get("save_steps", 500),
        "save_total_limit": training_cfg.get("save_total_limit", 3),
        "eval_strategy": training_cfg.get("eval_strategy", "no") if has_eval_dataset else "no",
        "eval_steps": training_cfg.get("eval_steps"),
        "seed": training_cfg.get("seed", 42),
        "report_to": report_to if report_to is not None else training_cfg.get("report_to", "none"),
        "max_length": model_cfg.get("max_seq_length", 2048),
        "packing": False,
    }
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    return SFTConfig(**{key: value for key, value in kwargs.items() if value is not None})


def load_unsloth_model(config: dict[str, Any]):
    import torch  # noqa: PLC0415
    from unsloth import FastLanguageModel  # noqa: PLC0415

    model_cfg = config["model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=model_cfg.get("load_in_4bit", False),
        load_in_16bit=model_cfg.get("load_in_16bit", True),
    )

    lora_cfg = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
        use_rslora=lora_cfg.get("use_rslora", False),
        use_gradient_checkpointing=lora_cfg.get("use_gradient_checkpointing", "unsloth"),
    )
    return model, tokenizer


def build_formatting_func(tokenizer):
    def format_messages(messages: list[dict[str, str]]) -> str:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def formatting_func(example: dict[str, Any]) -> list[str]:
        messages = example["messages"]
        if messages and isinstance(messages[0], list):
            return [format_messages(item) for item in messages]
        return [format_messages(messages)]

    return formatting_func


def train(
    *,
    config_path: Path,
    data_path: Path,
    eval_data_path: Path | None,
    dry_run: bool,
    validate_data_only: bool,
    max_steps: int | None,
    output_dir: Path | None,
    report_to: str | None,
    allow_oracle_diagnostic_data: bool,
    expected_training_target: str | None,
    expected_evaluation_mode: str | None,
    expected_benchmark: str | None,
) -> None:
    config = load_config(config_path)
    train_dataset = load_jsonl_dataset(data_path)
    eval_dataset = load_jsonl_dataset(eval_data_path) if eval_data_path else None

    print(f"Loaded config: {config_path}")
    print(f"Loaded train rows: {len(train_dataset)} from {data_path}")
    oracle_train_rows = require_oracle_diagnostic_acknowledgement(
        train_dataset,
        dataset_name="training",
        allow_oracle_diagnostic_data=allow_oracle_diagnostic_data,
    )
    require_expected_dataset_metadata(
        train_dataset,
        dataset_name="training",
        expected_training_target=expected_training_target,
        expected_evaluation_mode=expected_evaluation_mode,
        expected_benchmark=expected_benchmark,
    )
    if oracle_train_rows:
        print(
            "WARNING: "
            f"{oracle_train_rows}/{len(train_dataset)} training rows are oracle planner diagnostics. "
            f"{ORACLE_DIAGNOSTIC_WARNING}"
        )
    if eval_dataset is not None:
        print(f"Loaded eval rows: {len(eval_dataset)} from {eval_data_path}")
        oracle_eval_rows = require_oracle_diagnostic_acknowledgement(
            eval_dataset,
            dataset_name="eval",
            allow_oracle_diagnostic_data=allow_oracle_diagnostic_data,
        )
        require_expected_dataset_metadata(
            eval_dataset,
            dataset_name="eval",
            expected_training_target=expected_training_target,
            expected_evaluation_mode=expected_evaluation_mode,
            expected_benchmark=expected_benchmark,
        )
        if oracle_eval_rows:
            print(
                "WARNING: "
                f"{oracle_eval_rows}/{len(eval_dataset)} eval rows are oracle planner diagnostics. "
                f"{ORACLE_DIAGNOSTIC_WARNING}"
            )
    if validate_data_only:
        return

    model, tokenizer = load_unsloth_model(config)
    if dry_run:
        print("Dry-run: model loaded, LoRA attached, and data validated. Exiting before training.")
        return

    from trl import SFTTrainer  # noqa: PLC0415

    args = build_sft_config(
        config,
        has_eval_dataset=eval_dataset is not None,
        max_steps=max_steps,
        output_dir=output_dir,
        report_to=report_to,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=build_formatting_func(tokenizer),
    )
    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Saved final adapter to {final_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--eval-data", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Load model + data, don't train")
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="Validate prepared JSONL without importing GPU/model libraries",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Bounded smoke-test training")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config training output_dir")
    parser.add_argument("--report-to", default=None, help="Override Trainer report_to, e.g. none")
    parser.add_argument(
        "--allow-oracle-diagnostic-data",
        action="store_true",
        help=(
            "Allow training/eval JSONL that contains gold SQL-derived planning hints. "
            "Use only for explicit oracle diagnostic runs."
        ),
    )
    parser.add_argument("--expected-training-target", default=None)
    parser.add_argument("--expected-evaluation-mode", default=None)
    parser.add_argument("--expected-benchmark", default=None)
    args = parser.parse_args()

    train(
        config_path=args.config,
        data_path=args.data,
        eval_data_path=args.eval_data,
        dry_run=args.dry_run,
        validate_data_only=args.validate_data_only,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        report_to=args.report_to,
        allow_oracle_diagnostic_data=args.allow_oracle_diagnostic_data,
        expected_training_target=args.expected_training_target,
        expected_evaluation_mode=args.expected_evaluation_mode,
        expected_benchmark=args.expected_benchmark,
    )


if __name__ == "__main__":
    main()
