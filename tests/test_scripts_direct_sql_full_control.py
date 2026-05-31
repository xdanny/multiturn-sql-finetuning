from __future__ import annotations

from pathlib import Path

from scripts.direct_sql_full_control import build_workflow_steps, filter_steps, selected_stages


def _commands_by_name(config_path: Path) -> dict[str, tuple[str, ...]]:
    return {step.name: step.command for step in build_workflow_steps(config_path=config_path)}


def _flag_value(command: tuple[str, ...], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_build_workflow_steps_names_checkpoint3_artifact_paths() -> None:
    commands = _commands_by_name(Path("configs/direct_sql_full_non_oracle.yaml"))

    assert commands["prepare_train"] == (
        "uv",
        "run",
        "--active",
        "--no-sync",
        "python",
        "-m",
        "data.prepare_split",
        "--split-id",
        "cosql_train_v1",
        "--output",
        "data/processed/direct_sql_full/cosql_train_v1.jsonl",
        "--manifest-output",
        "data/processed/direct_sql_full/cosql_train_v1.manifest.json",
    )
    assert (
        "results/runs/direct_sql_full_non_oracle_control/"
        "base_proxy_dev_seen.manifest.json"
    ) in commands["eval_base_proxy_dev_seen"]
    assert (
        "results/runs/direct_sql_full_non_oracle_control/"
        "lora_proxy_dev_seen.manifest.json"
    ) in commands["eval_lora_proxy_dev_seen"]
    assert (
        "results/runs/direct_sql_full_non_oracle_control/"
        "base_clean_holdout.manifest.json"
    ) in commands["eval_base_clean_holdout"]
    assert (
        "results/runs/direct_sql_full_non_oracle_control/"
        "lora_clean_holdout.manifest.json"
    ) in commands["eval_lora_clean_holdout"]
    assert (
        "results/runs/direct_sql_full_non_oracle_control/"
        "base_generated_history_rollout.manifest.json"
    ) in commands["rollout_base"]
    assert (
        "results/runs/direct_sql_full_non_oracle_control/"
        "lora_generated_history_rollout.manifest.json"
    ) in commands["rollout_lora"]
    assert commands["audit_checkpoint3_artifacts"] == (
        "uv",
        "run",
        "--active",
        "--no-sync",
        "python",
        "-m",
        "eval.checkpoint3_artifact_audit",
        "--config",
        "configs/direct_sql_full_non_oracle.yaml",
    )


def test_build_workflow_steps_supports_smoke_limits_and_model_names() -> None:
    steps = build_workflow_steps(
        config_path=Path("configs/direct_sql_full_non_oracle.yaml"),
        endpoint="http://localhost:9000/v1",
        base_model_name="base-served",
        lora_model_name="lora-served",
        database_root=Path("data/raw/cosql_dataset/database"),
        eval_limit=5,
        rollout_limit_dialogs=3,
        prompt_token_cost_usd_per_1k=0.01,
        completion_token_cost_usd_per_1k=0.02,
    )
    commands = {step.name: step.command for step in steps}

    base_eval = commands["eval_base_proxy_dev_seen"]
    lora_eval = commands["eval_lora_clean_holdout"]
    rollout = commands["rollout_lora"]
    assert "http://localhost:9000/v1" in base_eval
    assert "base-served" in base_eval
    assert "lora-served" in lora_eval
    assert _flag_value(base_eval, "--limit") == "5"
    assert "--prompt-token-cost-usd-per-1k" in base_eval
    assert "--completion-token-cost-usd-per-1k" in base_eval
    assert "--database-root" in rollout
    assert _flag_value(rollout, "--limit-dialogs") == "3"


def test_filter_steps_selects_explicit_stages() -> None:
    steps = build_workflow_steps(config_path=Path("configs/direct_sql_full_non_oracle.yaml"))

    selected = filter_steps(steps, selected_stages(["eval", "audit"]))

    assert [step.stage for step in selected] == ["eval", "eval", "eval", "eval", "audit"]
