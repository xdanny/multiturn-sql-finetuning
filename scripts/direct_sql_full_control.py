#!/usr/bin/env python3
"""Print or run the Roadmap Checkpoint 3 direct-SQL control workflow."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path("configs/direct_sql_full_non_oracle.yaml")
DEFAULT_ENDPOINT = "http://localhost:8000/v1"
DEFAULT_API_KEY = "EMPTY"
DEFAULT_LORA_MODEL_NAME = "direct_sql_full_non_oracle_lora"
STAGES = ("prepare", "validate", "train", "eval", "rollout", "analysis", "audit")


@dataclass(frozen=True)
class WorkflowStep:
    """One executable command in the direct-SQL control workflow."""

    stage: str
    name: str
    command: tuple[str, ...]

    def shell_command(self) -> str:
        return shlex.join(self.command)


def _uv_python_module_command(module: str, *args: str) -> tuple[str, ...]:
    return ("uv", "run", "--active", "--no-sync", "python", "-m", module, *args)


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _path_from_manifest_path(manifest_path: str) -> str:
    suffix = ".manifest.json"
    if manifest_path.endswith(suffix):
        return f"{manifest_path[: -len(suffix)]}.jsonl"
    return str(Path(manifest_path).with_suffix(".jsonl"))


def _append_optional_path(command: list[str], flag: str, path: Path | None) -> None:
    if path is not None:
        command.extend([flag, str(path)])


def _append_optional_int(command: list[str], flag: str, value: int | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _append_cost_flags(
    command: list[str],
    *,
    prompt_token_cost_usd_per_1k: float,
    completion_token_cost_usd_per_1k: float,
) -> None:
    if prompt_token_cost_usd_per_1k:
        command.extend(
            ["--prompt-token-cost-usd-per-1k", str(prompt_token_cost_usd_per_1k)]
        )
    if completion_token_cost_usd_per_1k:
        command.extend(
            ["--completion-token-cost-usd-per-1k", str(completion_token_cost_usd_per_1k)]
        )


def _prepared_entry(config: dict[str, Any], name: str) -> dict[str, Any]:
    entry = (config.get("prepared_data") or {}).get(name)
    if not isinstance(entry, dict):
        raise ValueError(f"prepared_data.{name} is required")
    return entry


def _artifact_entries(config: dict[str, Any], name: str) -> dict[str, Any]:
    artifact_config = config.get("checkpoint3_artifacts") or {}
    entries = artifact_config.get(name) or {}
    if not isinstance(entries, dict):
        raise ValueError(f"checkpoint3_artifacts.{name} must be an object")
    return entries


def build_workflow_steps(
    *,
    config_path: Path = DEFAULT_CONFIG,
    endpoint: str = DEFAULT_ENDPOINT,
    api_key: str = DEFAULT_API_KEY,
    base_model_name: str | None = None,
    lora_model_name: str = DEFAULT_LORA_MODEL_NAME,
    database_root: Path | None = None,
    eval_limit: int | None = None,
    rollout_limit_dialogs: int | None = None,
    prompt_token_cost_usd_per_1k: float = 0.0,
    completion_token_cost_usd_per_1k: float = 0.0,
) -> list[WorkflowStep]:
    """Build the command sequence needed to satisfy Checkpoint 3 artifacts."""

    config = _load_config(config_path)
    base_model_name = base_model_name or str((config.get("model") or {}).get("name") or "")
    if not base_model_name:
        raise ValueError("model.name is required when --base-model-name is not set")

    train_entry = _prepared_entry(config, "train")
    proxy_entry = _prepared_entry(config, "proxy_dev_seen")
    holdout_entry = _prepared_entry(config, "clean_local_holdout")
    result_manifests = _artifact_entries(config, "result_manifests")
    rollout_manifests = _artifact_entries(config, "generated_history_rollout_manifests")

    steps: list[WorkflowStep] = []
    for name, entry in (
        ("train", train_entry),
        ("proxy_dev_seen", proxy_entry),
        ("clean_local_holdout", holdout_entry),
    ):
        steps.append(
            WorkflowStep(
                stage="prepare",
                name=f"prepare_{name}",
                command=_uv_python_module_command(
                    "data.prepare_split",
                    "--split-id",
                    str(entry["split_id"]),
                    "--output",
                    str(entry["path"]),
                    "--manifest-output",
                    str(entry["manifest"]),
                ),
            )
        )

    for name, eval_entry in (
        ("proxy_dev_seen", proxy_entry),
        ("clean_local_holdout", holdout_entry),
    ):
        steps.append(
            WorkflowStep(
                stage="validate",
                name=f"validate_{name}",
                command=_uv_python_module_command(
                    "train.finetune",
                    "--config",
                    str(config_path),
                    "--data",
                    str(train_entry["path"]),
                    "--eval-data",
                    str(eval_entry["path"]),
                    "--validate-data-only",
                ),
            )
        )

    steps.append(
        WorkflowStep(
            stage="train",
            name="train_lora",
            command=_uv_python_module_command(
                "train.finetune",
                "--config",
                str(config_path),
                "--data",
                str(train_entry["path"]),
                "--eval-data",
                str(proxy_entry["path"]),
            ),
        )
    )

    eval_specs = (
        ("base_proxy_dev_seen", base_model_name, proxy_entry),
        ("lora_proxy_dev_seen", lora_model_name, proxy_entry),
        ("base_clean_holdout", base_model_name, holdout_entry),
        ("lora_clean_holdout", lora_model_name, holdout_entry),
    )
    for artifact_name, model_name, input_entry in eval_specs:
        artifact = result_manifests.get(artifact_name)
        if not isinstance(artifact, dict):
            raise ValueError(f"result_manifests.{artifact_name} is required")
        manifest_path = str(artifact["manifest"])
        command = list(
            _uv_python_module_command(
                "eval.run_eval",
                "--benchmark",
                "prepared",
                "--endpoint",
                endpoint,
                "--model-name",
                model_name,
                "--input",
                str(input_entry["path"]),
                "--output",
                _path_from_manifest_path(manifest_path),
                "--api-key",
                api_key,
                "--manifest-output",
                manifest_path,
            )
        )
        _append_optional_int(command, "--limit", eval_limit)
        _append_optional_path(command, "--database-root", database_root)
        _append_cost_flags(
            command,
            prompt_token_cost_usd_per_1k=prompt_token_cost_usd_per_1k,
            completion_token_cost_usd_per_1k=completion_token_cost_usd_per_1k,
        )
        steps.append(
            WorkflowStep(stage="eval", name=f"eval_{artifact_name}", command=tuple(command))
        )

    for artifact_name, model_name in (
        ("base", base_model_name),
        ("lora", lora_model_name),
    ):
        artifact = rollout_manifests.get(artifact_name)
        if not isinstance(artifact, dict):
            raise ValueError(f"generated_history_rollout_manifests.{artifact_name} is required")
        manifest_path = str(artifact["manifest"])
        command = list(
            _uv_python_module_command(
                "eval.rollout_eval",
                "--endpoint",
                endpoint,
                "--model-name",
                model_name,
                "--input",
                str(holdout_entry["path"]),
                "--output",
                _path_from_manifest_path(manifest_path),
                "--api-key",
                api_key,
                "--manifest-output",
                manifest_path,
            )
        )
        _append_optional_int(command, "--limit-dialogs", rollout_limit_dialogs)
        _append_optional_path(command, "--database-root", database_root)
        steps.append(
            WorkflowStep(
                stage="rollout",
                name=f"rollout_{artifact_name}",
                command=tuple(command),
            )
        )

    steps.append(
        WorkflowStep(
            stage="analysis",
            name="analyze_clean_holdout_failures",
            command=_uv_python_module_command(
                "eval.clean_holdout_failure_analysis",
                "--config",
                str(config_path),
            ),
        )
    )

    steps.append(
        WorkflowStep(
            stage="audit",
            name="audit_checkpoint3_artifacts",
            command=_uv_python_module_command(
                "eval.checkpoint3_artifact_audit",
                "--config",
                str(config_path),
            ),
        )
    )
    return steps


def selected_stages(values: Iterable[str]) -> set[str]:
    stages = set(values)
    if "all" in stages:
        return set(STAGES)
    unknown = stages.difference(STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {', '.join(sorted(unknown))}")
    return stages


def filter_steps(steps: Sequence[WorkflowStep], stages: set[str]) -> list[WorkflowStep]:
    return [step for step in steps if step.stage in stages]


def print_steps(steps: Sequence[WorkflowStep]) -> None:
    for index, step in enumerate(steps, start=1):
        print(f"# {index}. [{step.stage}] {step.name}")
        print(step.shell_command())


def run_steps(steps: Sequence[WorkflowStep]) -> None:
    for step in steps:
        print(f"$ {step.shell_command()}", flush=True)
        subprocess.run(step.command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", action="append", default=None, choices=("all", *STAGES))
    parser.add_argument("--run", action="store_true", help="Execute commands instead of printing them.")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--base-model-name", default=None)
    parser.add_argument("--lora-model-name", default=DEFAULT_LORA_MODEL_NAME)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--rollout-limit-dialogs", type=int, default=None)
    parser.add_argument("--prompt-token-cost-usd-per-1k", type=float, default=0.0)
    parser.add_argument("--completion-token-cost-usd-per-1k", type=float, default=0.0)
    args = parser.parse_args()

    stages = selected_stages(args.stage or ["all"])
    steps = filter_steps(
        build_workflow_steps(
            config_path=args.config,
            endpoint=args.endpoint,
            api_key=args.api_key,
            base_model_name=args.base_model_name,
            lora_model_name=args.lora_model_name,
            database_root=args.database_root,
            eval_limit=args.eval_limit,
            rollout_limit_dialogs=args.rollout_limit_dialogs,
            prompt_token_cost_usd_per_1k=args.prompt_token_cost_usd_per_1k,
            completion_token_cost_usd_per_1k=args.completion_token_cost_usd_per_1k,
        ),
        stages,
    )
    if args.run:
        run_steps(steps)
    else:
        print_steps(steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
