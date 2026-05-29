from __future__ import annotations

from pathlib import Path

from train.finetuning_steps import (
    STAGE_ORDER,
    build_step_command_record,
    finetuning_step_summary,
    load_finetuning_steps,
    select_step_commands,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_finetuning_steps_load_in_execution_order() -> None:
    steps = load_finetuning_steps(
        REPO_ROOT / "configs" / "finetuning_steps.yaml",
        method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
        protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
        claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
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
    assert [step["stage"] for step in steps] == [
        "train_control",
        "endpoint_comparison",
        "endpoint_comparison",
        "train_and_compare",
        "train_and_rollout",
        "transfer_gate",
    ]
    assert [STAGE_ORDER[step["stage"]] for step in steps] == sorted(
        STAGE_ORDER[step["stage"]] for step in steps
    )
    assert steps[0]["requires_step_ids"] == ()
    assert steps[-1]["requires_step_ids"] == (
        "direct_sql_control_smoke",
        "planner_first_sql_pair",
        "semantic_value_retrieval_pair",
        "metric_dsl_vs_direct_sql",
        "behavior_recovery_rollout_pair",
    )
    metric = next(step for step in steps if step["step_id"] == "metric_dsl_vs_direct_sql")
    assert metric["method"] == "MEASURE()-preserving metric DSL"
    assert metric["train_rows_status"] == {
        "docs/data_artifacts/metric_dsl_training_rows.jsonl": True,
        "docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl": True,
    }
    assert "metric_dsl_beats_direct_sql" in metric["clears_claim_ids"]
    assert metric["measurement"]["primary_metric"] == (
        "compiled_value_accuracy_delta_vs_direct_sql"
    )
    assert metric["measurement"]["benchmark_metric_refs"] == (
        "synthetic_schema_rich_method_fixture:value_accuracy",
        "synthetic_schema_rich_method_fixture:measure_preservation",
    )
    assert "measure_preservation" in metric["measurement"]["supporting_metrics"]
    assert all(command.startswith("uv run --active --no-sync") for command in metric["commands"])
    assert all(
        command.startswith("uv run --active --no-sync")
        for command in metric["preflight_commands"]
    )
    assert all("--validate-data-only" in command for command in metric["preflight_commands"])


def test_finetuning_step_summary_exposes_readiness() -> None:
    summary = finetuning_step_summary(
        path=REPO_ROOT / "configs" / "finetuning_steps.yaml",
        method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
        protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
        claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
        repo_root=REPO_ROOT,
    )

    assert summary["step_count"] == 6
    steps = {step["step_id"]: step for step in summary["steps"]}
    assert steps["direct_sql_control_smoke"]["train_rows_ready"] is True
    assert steps["direct_sql_control_smoke"]["requires_step_ids"] == []
    assert steps["planner_first_sql_pair"]["requires_step_ids"] == [
        "direct_sql_control_smoke"
    ]
    assert steps["direct_sql_control_smoke"]["eval_rows_ready"] is True
    assert steps["direct_sql_control_smoke"]["cheap_preflight_available"] is True
    assert steps["planner_first_sql_pair"]["preflight_command_count"] == 1
    assert steps["semantic_value_retrieval_pair"]["preflight_command_count"] == 2
    assert steps["hosted_bird_interact_gate"]["eval_rows_ready"] is True
    assert steps["hosted_bird_interact_gate"]["clears_claim_ids"] == [
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
    ]
    assert steps["metric_dsl_vs_direct_sql"]["measurement"] == {
        "primary_metric": "compiled_value_accuracy_delta_vs_direct_sql",
        "benchmark_metric_refs": [
            "synthetic_schema_rich_method_fixture:value_accuracy",
            "synthetic_schema_rich_method_fixture:measure_preservation",
        ],
        "supporting_metrics": [
            "dsl_parse_rate",
            "dsl_compile_rate",
            "measure_preservation",
            "measure_f1",
            "dimension_f1",
            "filter_f1",
        ],
        "comparison_artifact": "results/metric_dsl/<run-id>.comparison.manifest.json",
        "promoted_when": (
            "compiled metric DSL beats direct SQL while preserving MEASURE(...) intent"
        ),
    }


def test_select_step_commands_returns_preflight_and_run_groups() -> None:
    preflight = select_step_commands(
        step_id="planner_first_sql_pair",
        command_group="preflight",
        path=REPO_ROOT / "configs" / "finetuning_steps.yaml",
        method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
        protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
        claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
        repo_root=REPO_ROOT,
    )
    run = select_step_commands(
        step_id="planner_first_sql_pair",
        command_group="run",
        path=REPO_ROOT / "configs" / "finetuning_steps.yaml",
        method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
        protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
        claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
        repo_root=REPO_ROOT,
    )
    all_commands = select_step_commands(
        step_id="planner_first_sql_pair",
        command_group="all",
        path=REPO_ROOT / "configs" / "finetuning_steps.yaml",
        method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
        protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
        claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
        repo_root=REPO_ROOT,
    )

    assert len(preflight) == 1
    assert "--preflight-only" in preflight[0]
    assert len(run) == 1
    assert "--preflight-only" not in run[0]
    assert all_commands == preflight + run


def test_select_step_commands_rejects_unknown_step() -> None:
    try:
        select_step_commands(
            step_id="missing_step",
            command_group="preflight",
            path=REPO_ROOT / "configs" / "finetuning_steps.yaml",
            method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
            protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "unknown finetuning step id: missing_step" in str(exc)
    else:
        raise AssertionError("unknown step id should fail")


def test_build_step_command_record_includes_handoff_metadata() -> None:
    record = build_step_command_record(
        step_id="semantic_value_retrieval_pair",
        command_group="preflight",
        path=REPO_ROOT / "configs" / "finetuning_steps.yaml",
        method_config_path=REPO_ROOT / "configs" / "finetuning_methods.yaml",
        protocol_config_path=REPO_ROOT / "configs" / "benchmark_protocols.yaml",
        claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
        repo_root=REPO_ROOT,
    )

    assert record["schema_version"] == 1
    assert record["step_id"] == "semantic_value_retrieval_pair"
    assert record["method"] == "Semantic-layer tuning"
    assert record["command_group"] == "preflight"
    assert record["requires_step_ids"] == ["direct_sql_control_smoke"]
    assert len(record["commands"]) == 2
    assert record["cheap_preflight_available"] is True
    assert record["benchmark_protocol_ids"] == ["cosql_dev_100_teacher_forced_proxy"]
    assert record["eval_rows"] == {"data/processed/eval_cosql_dev_100.jsonl": True}
    assert record["control_rows"] == {
        "data/processed/eval_cosql_dev_100.jsonl": True,
        "docs/data_artifacts/semantic_value_retrieval_inputs.manifest.json": True,
    }
    assert record["measurement"]["primary_metric"] == "value_accuracy_delta_vs_direct_sql"
    assert record["measurement"]["benchmark_metric_refs"] == [
        "cosql_dev_100_teacher_forced_proxy:value_accuracy"
    ]
    assert record["measurement"]["comparison_artifact"] == (
        "results/semantic_value_retrieval/<run-id>.comparison.manifest.json"
    )
    assert record["clears_claim_ids"] == ["semantic_value_retrieval_improves_sql"]
    assert "value index is database-derived" in record["leakage_boundary"]


def test_finetuning_step_loader_rejects_unknown_method(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: bad_step
    method: Missing method
    stage: train_control
    purpose: test
    benchmark_protocol_ids:
      - cosql_dev_100_teacher_forced_proxy
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    preflight_commands:
      - uv run --active --no-sync python -m train.finetune --validate-data-only
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "bad_step: unknown method Missing method" in str(exc)
    else:
        raise AssertionError("unknown method should fail")


def test_finetuning_step_loader_rejects_unknown_claim_id(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: bad_claim_step
    method: Direct SQL SFT
    stage: train_control
    purpose: test
    benchmark_protocol_ids:
      - cosql_dev_100_teacher_forced_proxy
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    preflight_commands:
      - uv run --active --no-sync python -m train.finetune --validate-data-only
    evidence_gate: manifest
    clears_claim_ids:
      - nonexistent_claim
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "bad_claim_step: unknown claim ids: nonexistent_claim" in str(exc)
    else:
        raise AssertionError("unknown claim id should fail")


def test_finetuning_step_loader_rejects_method_claim_mismatch(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: wrong_method_claim_step
    method: Direct SQL SFT
    stage: train_control
    purpose: test
    benchmark_protocol_ids:
      - cosql_dev_100_teacher_forced_proxy
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    preflight_commands:
      - uv run --active --no-sync python -m train.finetune --validate-data-only
    evidence_gate: manifest
    clears_claim_ids:
      - metric_dsl_beats_direct_sql
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert (
            "wrong_method_claim_step: claim ids are not declared by method Direct SQL SFT: "
            "metric_dsl_beats_direct_sql"
        ) in str(exc)
    else:
        raise AssertionError("method claim mismatch should fail")


def test_finetuning_step_loader_requires_preflight_commands(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: no_preflight_step
    method: Direct SQL SFT
    stage: train_control
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "no_preflight_step: missing preflight_commands" in str(exc)
    else:
        raise AssertionError("missing preflight commands should fail")


def test_finetuning_step_loader_requires_measurement_contract(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: no_measurement_step
    method: Direct SQL SFT
    stage: train_control
    purpose: test
    benchmark_protocol_ids:
      - cosql_dev_100_teacher_forced_proxy
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    preflight_commands:
      - uv run --active --no-sync python -m train.finetune --validate-data-only
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "no_measurement_step: missing measurement" in str(exc)
    else:
        raise AssertionError("missing measurement should fail")


def test_finetuning_step_loader_rejects_unknown_benchmark_metric_ref(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: bad_metric_ref_step
    method: Direct SQL SFT
    stage: train_control
    purpose: test
    benchmark_protocol_ids:
      - cosql_dev_100_teacher_forced_proxy
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    preflight_commands:
      - uv run --active --no-sync python -m train.finetune --validate-data-only
    evidence_gate: manifest
    measurement:
      primary_metric: fake_metric
      benchmark_metric_refs:
        - cosql_dev_100_teacher_forced_proxy:fake_metric
      supporting_metrics:
        - syntax_validity
      comparison_artifact: results/fake.json
      promoted_when: fake metric wins
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert (
            "bad_metric_ref_step: benchmark metric ref "
            "cosql_dev_100_teacher_forced_proxy:fake_metric is not a primary metric"
        ) in str(exc)
    else:
        raise AssertionError("unknown benchmark metric ref should fail")


def test_finetuning_step_loader_rejects_unknown_stage(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: bad_stage_step
    method: Direct SQL SFT
    stage: surprise_stage
    purpose: test
    benchmark_protocol_ids:
      - cosql_dev_100_teacher_forced_proxy
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    preflight_commands:
      - uv run --active --no-sync python -m train.finetune --validate-data-only
    evidence_gate: manifest
    measurement:
      primary_metric: value_accuracy
      benchmark_metric_refs:
        - cosql_dev_100_teacher_forced_proxy:value_accuracy
      supporting_metrics:
        - syntax_validity
      comparison_artifact: results/fake.json
      promoted_when: value accuracy is scored
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert "bad_stage_step: unknown stage surprise_stage" in str(exc)
    else:
        raise AssertionError("unknown stage should fail")


def test_finetuning_step_loader_rejects_backward_stage_order(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: hosted_first
    method: Hosted and BIRD-Interact comparison
    stage: transfer_gate
    purpose: test transfer first
    benchmark_protocol_ids:
      - bird_interact_same_protocol_transfer
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m eval.compare_hosted_baseline
    preflight_commands:
      - uv run --active --no-sync python -m train.finetuning_steps
    evidence_gate: manifest
    measurement:
      primary_metric: value_accuracy_delta_vs_hosted_baseline
      benchmark_metric_refs:
        - bird_interact_same_protocol_transfer:value_accuracy
      supporting_metrics:
        - task_success
      comparison_artifact: results/hosted/fake.json
      promoted_when: hosted comparison exists
    clears_claim_ids: []
    blocks_claim_ids: []
    leakage_boundary: no leakage
  - step_id: direct_later
    method: Direct SQL SFT
    stage: train_control
    purpose: test direct later
    benchmark_protocol_ids:
      - cosql_dev_100_teacher_forced_proxy
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    preflight_commands:
      - uv run --active --no-sync python -m train.finetune --validate-data-only
    evidence_gate: manifest
    measurement:
      primary_metric: value_accuracy
      benchmark_metric_refs:
        - cosql_dev_100_teacher_forced_proxy:value_accuracy
      supporting_metrics:
        - syntax_validity
      comparison_artifact: results/direct/fake.json
      promoted_when: direct control exists
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert (
            "direct_later: stage train_control appears after hosted_first "
            "with a later stage"
        ) in str(exc)
    else:
        raise AssertionError("backward stage order should fail")


def test_finetuning_step_loader_rejects_missing_dependency(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: missing_dependency_step
    method: Direct SQL SFT
    stage: train_control
    purpose: test missing dependency
    benchmark_protocol_ids:
      - cosql_dev_100_teacher_forced_proxy
    requires_step_ids:
      - absent_step
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    preflight_commands:
      - uv run --active --no-sync python -m train.finetune --validate-data-only
    evidence_gate: manifest
    measurement:
      primary_metric: value_accuracy
      benchmark_metric_refs:
        - cosql_dev_100_teacher_forced_proxy:value_accuracy
      supporting_metrics:
        - syntax_validity
      comparison_artifact: results/direct/fake.json
      promoted_when: direct control exists
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert (
            "missing_dependency_step: requires_step_ids must reference earlier steps: "
            "absent_step"
        ) in str(exc)
    else:
        raise AssertionError("missing dependency should fail")


def test_finetuning_step_loader_rejects_method_protocol_mismatch(tmp_path) -> None:
    config = tmp_path / "steps.yaml"
    config.write_text(
        """
schema_version: 1
steps:
  - step_id: wrong_protocol_step
    method: Direct SQL SFT
    stage: train_control
    purpose: test wrong protocol
    benchmark_protocol_ids:
      - synthetic_schema_rich_method_fixture
    requires_step_ids: []
    train_rows: []
    eval_rows: []
    control_rows: []
    commands:
      - uv run --active --no-sync python -m train.finetune
    preflight_commands:
      - uv run --active --no-sync python -m train.finetune --validate-data-only
    evidence_gate: manifest
    measurement:
      primary_metric: value_accuracy
      benchmark_metric_refs:
        - synthetic_schema_rich_method_fixture:value_accuracy
      supporting_metrics:
        - syntax_validity
      comparison_artifact: results/direct/fake.json
      promoted_when: direct control exists
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
            claim_ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
            repo_root=REPO_ROOT,
        )
    except ValueError as exc:
        assert (
            "wrong_protocol_step: benchmark protocols are not declared by method "
            "Direct SQL SFT: synthetic_schema_rich_method_fixture"
        ) in str(exc)
    else:
        raise AssertionError("method protocol mismatch should fail")
