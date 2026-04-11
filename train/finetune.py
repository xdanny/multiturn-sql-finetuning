"""
Qwen 3.5 bf16 LoRA fine-tuning on multi-turn SQL with Unsloth + TRL SFTTrainer.

Config-driven: all hyperparams in configs/*.yaml. No CLI args beyond --config.

Usage:
    python -m train.finetune --config configs/qwen35_9b_5090.yaml
    python -m train.finetune --config configs/qwen35_4b_colab_l4.yaml

TODO: implement actual training loop once data/prepare.py is complete.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Load model + data, don't train")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Loaded config: {args.config}")
    print(yaml.dump(config, sort_keys=False, indent=2))

    # Lazy imports so --dry-run doesn't require GPU libs
    from unsloth import FastLanguageModel  # noqa: PLC0415
    import torch  # noqa: PLC0415

    # 1. Load model
    model_cfg = config["model"]
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=torch.bfloat16,
        load_in_4bit=model_cfg.get("load_in_4bit", False),
    )

    # 2. Attach LoRA
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

    if args.dry_run:
        print("Dry-run: model + LoRA attached successfully. Exiting before training.")
        return

    # 3. Load data — TODO: wire up data/prepare.py output
    raise NotImplementedError(
        "Training loop not yet implemented. Run data/prepare.py first, then wire dataset into SFTTrainer here."
    )


if __name__ == "__main__":
    main()
