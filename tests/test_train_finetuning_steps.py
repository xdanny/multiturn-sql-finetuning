from __future__ import annotations

from pathlib import Path

from train.finetuning_steps import finetuning_step_summary, load_finetuning_steps

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_finetuning_steps_load_in_execution_order() -> None:
    steps = load_finetuning_steps(
        REPO_ROOT / "configs" / "finetuning_steps.yaml",
        method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
        protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
        repo_root=REPO_ROOT,
    )

    assert [step["step_id"] for step in steps] == [
        "direct_sql_control_smoke",
        "planner_first_sql_pair",
        "semantic_value_retrieval_pair",
        "metric_dsl_vs_direct_sql",
        "behavior_recovery_rollout_pair",
        "hosted_bird_interact_gate",
    ]
    metric = next(step for step in steps if step["step_id"] == "metric_dsl_vs_direct_sql")
    assert metric["method"] == "MEASURE()-preserving metric DSL"
    assert metric["train_rows_status"] == {
        "docs/data_artifacts/metric_dsl_training_rows.jsonl": True,
        "docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl": True,
    }
    assert "metric_dsl_beats_direct_sql" in metric["clears_claim_ids"]
    assert all(command.startswith("uv run --active --no-sync") for command in metric["commands"])


def test_finetuning_step_summary_exposes_readiness() -> None:
    summary = finetuning_step_summary(
        path=REPO_ROOT / "configs" / "finetuning_steps.yaml",
        method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
        protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
        repo_root=REPO_ROOT,
    )

    assert summary["step_count"] == 6
    steps = {step["step_id"]: step for step in summary["steps"]}
    assert steps["direct_sql_control_smoke"]["train_rows_ready"] is True
    assert steps["direct_sql_control_smoke"]["eval_rows_ready"] is True
    assert steps["hosted_bird_interact_gate"]["eval_rows_ready"] is True
    assert steps["hosted_bird_interact_gate"]["clears_claim_ids"] == [
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
    ]


def test_finetuning_step_loader_rejects_unknown_method(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: bad_step
    method: Missing method
    stage: train
    purpose: test
    benchmark_protocol_ids:
      - cosql_dev_100_teacher_forced_proxy
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    evidence_gate: manifest
    clears_claim_ids: []
    blocks_claim_ids: []
    leakage_boundary: no leakage
""",
        encoding="utf-8",
    )

    try:
        load_finetuning_steps(
            config,
            method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
            protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "bad_step: unknown method Missing method" in str(exc)
    else:
        raise AssertionError("unknown method should fail")
