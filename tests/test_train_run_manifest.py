from __future__ import annotations

import json
from pathlib import Path

import yaml

from train.finetune import train


def _write_config(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "model": {"name": "unit-model", "max_seq_length": 1024},
                "training": {"output_dir": "outputs/unit", "report_to": "none"},
                "lora": {
                    "r": 8,
                    "alpha": 16,
                    "dropout": 0.0,
                    "bias": "none",
                    "target_modules": ["q_proj"],
                },
            }
        )
    )


def _write_rows(path: Path, *, benchmark: str, training_target: str, evaluation_mode: str) -> None:
    row = {
        "messages": [
            {"role": "system", "content": "You write SQL."},
            {"role": "user", "content": "Show governed revenue by country."},
            {"role": "assistant", "content": "MEASURE(revenue) BY customer_country"},
        ],
        "benchmark": benchmark,
        "training_target": training_target,
        "evaluation_mode": evaluation_mode,
    }
    path.write_text(json.dumps(row) + "\n")


def test_validate_data_only_writes_stage_aware_training_manifest(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "outputs" / "metric_dsl"
    manifest_path = tmp_path / "training.manifest.json"
    _write_config(config_path)
    _write_rows(
        train_path,
        benchmark="synthetic_metric_dsl_bootstrap",
        training_target="metric_dsl",
        evaluation_mode="metric_dsl",
    )
    _write_rows(
        eval_path,
        benchmark="synthetic_metric_dsl_bootstrap",
        training_target="metric_dsl",
        evaluation_mode="metric_dsl",
    )

    train(
        config_path=config_path,
        data_path=train_path,
        eval_data_path=eval_path,
        dry_run=False,
        validate_data_only=True,
        max_steps=None,
        output_dir=output_dir,
        report_to="none",
        allow_oracle_diagnostic_data=False,
        expected_training_target="metric_dsl",
        expected_evaluation_mode="metric_dsl",
        expected_benchmark="synthetic_metric_dsl_bootstrap",
        training_manifest_output=manifest_path,
        run_id="metric-dsl-bootstrap",
    )

    manifest = json.loads(manifest_path.read_text())

    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "metric-dsl-bootstrap"
    assert manifest["status"] == "validated"
    assert manifest["stage"] == "metric_dsl"
    assert manifest["benchmark"] == "synthetic_metric_dsl_bootstrap"
    assert manifest["training_target"] == "metric_dsl"
    assert manifest["evaluation_mode"] == "metric_dsl"
    assert manifest["train_row_count"] == 1
    assert manifest["eval_row_count"] == 1
    assert manifest["train_data_path"] == str(train_path)
    assert manifest["eval_data_path"] == str(eval_path)
    assert manifest["train_data_sha256"]
    assert manifest["eval_data_sha256"]
    assert manifest["output_dir"] == str(output_dir)
    assert manifest["final_dir"] == str(output_dir / "final")
    assert manifest["config_path"] == str(config_path)
    assert manifest["config_sha256"]
    assert manifest["command"] == ["train.finetune", "validate-data-only"]


def test_validate_data_only_manifest_records_direct_sql_control_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    train_path = tmp_path / "train.jsonl"
    output_dir = tmp_path / "outputs" / "direct"
    manifest_path = tmp_path / "training.manifest.json"
    _write_config(config_path)
    _write_rows(
        train_path,
        benchmark="metric_dsl_direct_sql",
        training_target="direct_sql_control",
        evaluation_mode="non_oracle_generation",
    )

    train(
        config_path=config_path,
        data_path=train_path,
        eval_data_path=None,
        dry_run=False,
        validate_data_only=True,
        max_steps=None,
        output_dir=output_dir,
        report_to="none",
        allow_oracle_diagnostic_data=False,
        expected_training_target="direct_sql_control",
        expected_evaluation_mode="non_oracle_generation",
        expected_benchmark="metric_dsl_direct_sql",
        training_manifest_output=manifest_path,
        run_id="metric-dsl-direct-control",
    )

    manifest = json.loads(manifest_path.read_text())

    assert manifest["status"] == "validated"
    assert manifest["stage"] == "direct_sql_control"
    assert manifest["benchmark"] == "metric_dsl_direct_sql"
    assert manifest["training_target"] == "direct_sql_control"
    assert manifest["evaluation_mode"] == "non_oracle_generation"
    assert manifest["eval_row_count"] == 0


def test_validate_data_only_manifest_records_semantic_layer_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    train_path = tmp_path / "train.jsonl"
    output_dir = tmp_path / "outputs" / "semantic"
    manifest_path = tmp_path / "training.manifest.json"
    _write_config(config_path)
    _write_rows(
        train_path,
        benchmark="synthetic_semantic_layer",
        training_target="semantic_layer",
        evaluation_mode="non_oracle_generation",
    )

    train(
        config_path=config_path,
        data_path=train_path,
        eval_data_path=None,
        dry_run=False,
        validate_data_only=True,
        max_steps=None,
        output_dir=output_dir,
        report_to="none",
        allow_oracle_diagnostic_data=False,
        expected_training_target="semantic_layer",
        expected_evaluation_mode="non_oracle_generation",
        expected_benchmark="synthetic_semantic_layer",
        training_manifest_output=manifest_path,
        run_id="semantic-layer-bootstrap",
    )

    manifest = json.loads(manifest_path.read_text())

    assert manifest["status"] == "validated"
    assert manifest["stage"] == "semantic_layer"
    assert manifest["benchmark"] == "synthetic_semantic_layer"
    assert manifest["training_target"] == "semantic_layer"
    assert manifest["evaluation_mode"] == "non_oracle_generation"
    assert manifest["eval_row_count"] == 0


def test_validate_data_only_manifest_records_behavior_recovery_stage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    train_path = tmp_path / "train.jsonl"
    output_dir = tmp_path / "outputs" / "recovery"
    manifest_path = tmp_path / "training.manifest.json"
    _write_config(config_path)
    _write_rows(
        train_path,
        benchmark="synthetic_behavior_recovery",
        training_target="behavior_recovery",
        evaluation_mode="non_oracle_generation",
    )

    train(
        config_path=config_path,
        data_path=train_path,
        eval_data_path=None,
        dry_run=False,
        validate_data_only=True,
        max_steps=None,
        output_dir=output_dir,
        report_to="none",
        allow_oracle_diagnostic_data=False,
        expected_training_target="behavior_recovery",
        expected_evaluation_mode="non_oracle_generation",
        expected_benchmark="synthetic_behavior_recovery",
        training_manifest_output=manifest_path,
        run_id="behavior-recovery-bootstrap",
    )

    manifest = json.loads(manifest_path.read_text())

    assert manifest["status"] == "validated"
    assert manifest["stage"] == "behavior_recovery"
    assert manifest["benchmark"] == "synthetic_behavior_recovery"
    assert manifest["training_target"] == "behavior_recovery"
    assert manifest["evaluation_mode"] == "non_oracle_generation"
    assert manifest["eval_row_count"] == 0


def test_validate_data_only_manifest_records_behavior_recovery_prepared_proxy_stage(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "outputs" / "behavior_recovery_proxy"
    manifest_path = tmp_path / "training.manifest.json"
    _write_config(config_path)
    _write_rows(
        train_path,
        benchmark="prepared",
        training_target="behavior_recovery",
        evaluation_mode="non_oracle_generation",
    )
    _write_rows(
        eval_path,
        benchmark="prepared",
        training_target="behavior_recovery",
        evaluation_mode="non_oracle_generation",
    )

    train(
        config_path=config_path,
        data_path=train_path,
        eval_data_path=eval_path,
        dry_run=False,
        validate_data_only=True,
        max_steps=None,
        output_dir=output_dir,
        report_to="none",
        allow_oracle_diagnostic_data=False,
        expected_training_target="behavior_recovery",
        expected_evaluation_mode="non_oracle_generation",
        expected_benchmark="prepared",
        training_manifest_output=manifest_path,
        run_id="behavior-recovery-proxy",
    )

    manifest = json.loads(manifest_path.read_text())

    assert manifest["status"] == "validated"
    assert manifest["stage"] == "behavior_recovery"
    assert manifest["benchmark"] == "prepared"
    assert manifest["training_target"] == "behavior_recovery"
    assert manifest["evaluation_mode"] == "non_oracle_generation"
    assert manifest["eval_row_count"] == 1
