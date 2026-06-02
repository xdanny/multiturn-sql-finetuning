from __future__ import annotations

import json
from pathlib import Path

import yaml

from eval.roadmap_status import summarize_roadmap_status


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_registry(path: Path) -> None:
    experiments = [
        (
            "direct_sql_full_non_oracle_control",
            3,
            "lora_trained_pending_endpoint_eval",
            "direct_sql",
            "",
        ),
        (
            "predicted_planner_sql_vs_direct",
            5,
            "planner_slotaware_readiness_negative",
            "predicted_planner_sql",
            "direct_sql_full_non_oracle_control",
        ),
        (
            "semantic_value_retrieval_vs_direct",
            6,
            "semantic_pruned_clean_holdout_promoted",
            "semantic_value_retrieval",
            "direct_sql_full_non_oracle_control",
        ),
        (
            "metric_dsl_vs_direct_sql",
            7,
            "diagnostic_negative",
            "metric_dsl",
            "direct_sql_full_non_oracle_control",
        ),
        (
            "generated_history_recovery_vs_direct",
            8,
            "diagnostic_negative",
            "behavior_recovery",
            "direct_sql_full_non_oracle_control",
        ),
        (
            "hosted_bird_interact_transfer",
            9,
            "pending_local_winner",
            "hosted_transfer_comparison",
            "direct_sql_full_non_oracle_control",
        ),
    ]
    payload = {
        "schema_version": 1,
        "experiments": [
            {
                "experiment_id": experiment_id,
                "checkpoint": checkpoint,
                "status": status,
                "hypothesis_id": method,
                "hypothesis": f"{method} hypothesis",
                "train_split_id": "cosql_train_v1",
                "validation_split_id": "cosql_dev_100_proxy_seen_v1",
                "test_split_id": "cosql_dev_clean_holdout_v1",
                "dataset_role": "clean_local_holdout",
                "benchmark_protocol_id": "cosql_protocol",
                "model": "unsloth/Qwen3.5-9B",
                "adapter": "outputs/experiments/example/final",
                "method": method,
                "oracle_policy": "non_oracle_generation",
                "scorer": "eval.example:metric",
                "output_path": f"results/runs/{experiment_id}",
                "control_experiment_id": control_id or None,
                "primary_metric": "value_accuracy",
                "claim_boundary": "test fixture only",
            }
            for experiment_id, checkpoint, status, method, control_id in experiments
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _prepared_manifest(split_id: str, split_role: str) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "prepared_split_dataset",
        "split_id": split_id,
        "split_role": split_role,
        "row_count": 3,
        "oracle_policy": "non_oracle_generation",
        "evaluation_modes": {"non_oracle_generation": 3},
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
    }


def _result_manifest(split_id: str, split_role: str, row_hash: str) -> dict:
    return {
        "schema_version": 1,
        "benchmark": "prepared",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "row_count": 3,
        "output_sha256": "output",
        "metrics": {
            "value_execution_accuracy": 0.5,
            "strict_execution_accuracy": 0.5,
            "interaction_match_rate": 0.5,
            "mean_latency_ms": 1.0,
            "total_tokens": 100,
            "total_estimated_generation_cost_usd": 0.01,
            "split_ids": {split_id: 3},
            "split_roles": {split_role: 3},
            "split_row_ids_sha256": row_hash,
            "split_eval_turn_ids_sha256": f"{row_hash}-turns",
        },
    }


def _rollout_manifest() -> dict:
    return {
        "schema_version": 1,
        "benchmark": "generated_history_rollout",
        "oracle_allowed": False,
        "row_count": 2,
        "metrics": {
            "value_execution_accuracy": 0.5,
            "strict_execution_accuracy": 0.5,
            "mean_latency_ms": 1.0,
        },
    }


def _recorded_endpoint_evidence() -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "direct_sql_full_endpoint_evidence",
        "checkpoint3_artifact_audit": {"ok": True},
        "result_manifests": {
            name: {"row_count": 3}
            for name in (
                "base_proxy_dev_seen",
                "lora_proxy_dev_seen",
                "base_clean_holdout",
                "lora_clean_holdout",
                "base_generated_history_rollout",
                "lora_generated_history_rollout",
            )
        },
        "failure_analysis": {
            "artifact_type": "clean_holdout_failure_analysis",
            "split_role": "clean_local_holdout",
            "row_counts": {"base": 3, "lora": 3},
            "roadmap_method_hint_counts": {
                "base": {"planner_schema_linking": 2},
                "lora": {"planner_schema_linking": 1},
            },
        },
    }


def _write_checkpoint3_config(tmp_path: Path, *, complete: bool) -> Path:
    config_path = tmp_path / "configs" / "direct_sql_full_non_oracle.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "prepared_data": {
            "train": {
                "split_id": "cosql_train_v1",
                "path": "data/processed/direct_sql_full/cosql_train_v1.jsonl",
                "manifest": "data/processed/direct_sql_full/cosql_train_v1.manifest.json",
            },
            "proxy_dev_seen": {
                "split_id": "cosql_dev_100_proxy_seen_v1",
                "path": "data/processed/direct_sql_full/cosql_dev_100_proxy_seen_v1.jsonl",
                "manifest": (
                    "data/processed/direct_sql_full/"
                    "cosql_dev_100_proxy_seen_v1.manifest.json"
                ),
            },
            "clean_local_holdout": {
                "split_id": "cosql_dev_clean_holdout_v1",
                "path": "data/processed/direct_sql_full/cosql_dev_clean_holdout_v1.jsonl",
                "manifest": (
                    "data/processed/direct_sql_full/"
                    "cosql_dev_clean_holdout_v1.manifest.json"
                ),
            },
        },
        "checkpoint3_artifacts": {
            "required_eval_metrics": [
                "value_execution_accuracy",
                "strict_execution_accuracy",
                "interaction_match_rate",
                "mean_latency_ms",
                "total_tokens",
                "total_estimated_generation_cost_usd",
            ],
            "result_manifests": {
                "base_proxy_dev_seen": {
                    "model_role": "base",
                    "split_id": "cosql_dev_100_proxy_seen_v1",
                    "split_role": "proxy_dev_seen",
                    "manifest": (
                        "results/runs/direct_sql_full_non_oracle_control/"
                        "base_proxy_dev_seen.manifest.json"
                    ),
                },
                "lora_proxy_dev_seen": {
                    "model_role": "lora",
                    "split_id": "cosql_dev_100_proxy_seen_v1",
                    "split_role": "proxy_dev_seen",
                    "manifest": (
                        "results/runs/direct_sql_full_non_oracle_control/"
                        "lora_proxy_dev_seen.manifest.json"
                    ),
                },
                "base_clean_holdout": {
                    "model_role": "base",
                    "split_id": "cosql_dev_clean_holdout_v1",
                    "split_role": "clean_local_holdout",
                    "manifest": (
                        "results/runs/direct_sql_full_non_oracle_control/"
                        "base_clean_holdout.manifest.json"
                    ),
                },
                "lora_clean_holdout": {
                    "model_role": "lora",
                    "split_id": "cosql_dev_clean_holdout_v1",
                    "split_role": "clean_local_holdout",
                    "manifest": (
                        "results/runs/direct_sql_full_non_oracle_control/"
                        "lora_clean_holdout.manifest.json"
                    ),
                },
            },
            "generated_history_rollout_manifests": {
                "base": {
                    "model_role": "base",
                    "required_metrics": [
                        "value_execution_accuracy",
                        "strict_execution_accuracy",
                        "mean_latency_ms",
                    ],
                    "manifest": (
                        "results/runs/direct_sql_full_non_oracle_control/"
                        "base_generated_history_rollout.manifest.json"
                    ),
                },
                "lora": {
                    "model_role": "lora",
                    "required_metrics": [
                        "value_execution_accuracy",
                        "strict_execution_accuracy",
                        "mean_latency_ms",
                    ],
                    "manifest": (
                        "results/runs/direct_sql_full_non_oracle_control/"
                        "lora_generated_history_rollout.manifest.json"
                    ),
                },
            },
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if not complete:
        return config_path

    prepared_dir = tmp_path / "data" / "processed" / "direct_sql_full"
    prepared_roles = {
        "cosql_train_v1": "train",
        "cosql_dev_100_proxy_seen_v1": "proxy_dev_seen",
        "cosql_dev_clean_holdout_v1": "clean_local_holdout",
    }
    for split_id, split_role in prepared_roles.items():
        (prepared_dir / f"{split_id}.jsonl").parent.mkdir(parents=True, exist_ok=True)
        (prepared_dir / f"{split_id}.jsonl").write_text("{}\n", encoding="utf-8")
        _write_json(prepared_dir / f"{split_id}.manifest.json", _prepared_manifest(split_id, split_role))

    results_dir = tmp_path / "results" / "runs" / "direct_sql_full_non_oracle_control"
    for model_role in ("base", "lora"):
        _write_json(
            results_dir / f"{model_role}_proxy_dev_seen.manifest.json",
            _result_manifest("cosql_dev_100_proxy_seen_v1", "proxy_dev_seen", "proxy"),
        )
        _write_json(
            results_dir / f"{model_role}_clean_holdout.manifest.json",
            _result_manifest("cosql_dev_clean_holdout_v1", "clean_local_holdout", "holdout"),
        )
        _write_json(
            results_dir / f"{model_role}_generated_history_rollout.manifest.json",
            _rollout_manifest(),
        )
    return config_path


def test_summarize_roadmap_status_counts_current_checkpoints(tmp_path: Path) -> None:
    registry = tmp_path / "configs" / "experiments.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    _write_registry(registry)
    checkpoint3 = _write_checkpoint3_config(tmp_path, complete=True)

    summary = summarize_roadmap_status(
        experiment_registry_path=registry,
        checkpoint3_config_path=checkpoint3,
    )

    assert summary["status_counts"] == {"complete": 5, "in_progress": 4, "pending": 1}
    by_checkpoint = {row["checkpoint"]: row for row in summary["checkpoints"]}
    assert by_checkpoint[3]["status"] == "complete"
    assert "docs/training_runs/direct_sql_full_lora_20260531.json" in by_checkpoint[3]["evidence"]
    assert "eval.planner_readiness promotion policy" in by_checkpoint[5]["evidence"]
    assert (
        "docs/training_runs/planner_schema_context_repair_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_projection_prior_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_generic_column_prior_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_sft_data_path_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_sft_1000_readiness_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_sft_1000_sql_limit24_negative_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_sft_1000_sql_failure_analysis_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_projection_order_contract_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_projection_order_metric_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_orderaware_readiness_rerun_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_projection_sequence_prompt_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_sft_sequence_instruction_dataset_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_sequence_sft_1000_readiness_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_sequence_useronly_readiness_20260531.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_schema_label_gold_source_20260601.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_sft_schema_label_source_dataset_20260601.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_schema_label_source_sft_1000_20260601.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_schema_label_source_readiness_20260601.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_alias_normalized_readiness_20260601.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_output_slot_contract_20260602.json"
        in by_checkpoint[5]["evidence"]
    )
    assert (
        "docs/training_runs/planner_slotaware_readiness_20260602.json"
        in by_checkpoint[5]["evidence"]
    )
    assert by_checkpoint[5]["open_items"] == [
        "update planner-SFT targets and prompt/schema to predict ordered output_slots directly",
    ]
    assert (
        "eval.compare_semantic_value_retrieval promotion policy"
        in by_checkpoint[6]["evidence"]
    )
    assert (
        "docs/training_runs/semantic_value_clean_holdout_preflight_20260602.json"
        in by_checkpoint[6]["evidence"]
    )
    assert (
        "docs/training_runs/semantic_value_clean_holdout_full_20260602.json"
        in by_checkpoint[6]["evidence"]
    )
    assert (
        "docs/training_runs/semantic_value_regression_diagnosis_20260602.json"
        in by_checkpoint[6]["evidence"]
    )
    assert (
        "docs/training_runs/semantic_value_pruned_preflight_20260602.json"
        in by_checkpoint[6]["evidence"]
    )
    assert (
        "docs/training_runs/semantic_value_pruned_full_20260602.json"
        in by_checkpoint[6]["evidence"]
    )
    assert by_checkpoint[6]["status"] == "complete"
    assert by_checkpoint[6]["open_items"] == []
    assert (
        "eval.compare_metric_dsl_direct_sql promotion policy"
        in by_checkpoint[7]["evidence"]
    )
    assert by_checkpoint[7]["open_items"] == [
        "Metric DSL clean-holdout promotion policy must pass"
    ]
    assert by_checkpoint[9]["status"] == "pending"
    assert by_checkpoint[9]["open_items"] == [
        "hosted transfer waits for a local clean-holdout winner"
    ]


def test_summarize_roadmap_status_keeps_checkpoint3_in_progress_until_artifacts_exist(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "configs" / "experiments.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    _write_registry(registry)
    checkpoint3 = _write_checkpoint3_config(tmp_path, complete=False)

    summary = summarize_roadmap_status(
        experiment_registry_path=registry,
        checkpoint3_config_path=checkpoint3,
    )

    by_checkpoint = {row["checkpoint"]: row for row in summary["checkpoints"]}
    assert by_checkpoint[3]["status"] == "in_progress"
    assert any("missing manifest" in issue for issue in by_checkpoint[3]["open_items"])


def test_summarize_roadmap_status_accepts_recorded_endpoint_evidence(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "configs" / "experiments.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    _write_registry(registry)
    checkpoint3 = _write_checkpoint3_config(tmp_path, complete=False)
    evidence_path = tmp_path / "docs" / "training_runs" / "direct_sql_full_eval.json"
    _write_json(evidence_path, _recorded_endpoint_evidence())

    summary = summarize_roadmap_status(
        experiment_registry_path=registry,
        checkpoint3_config_path=checkpoint3,
        checkpoint3_evidence_path=evidence_path,
    )

    assert summary["status_counts"] == {"complete": 6, "in_progress": 3, "pending": 1}
    by_checkpoint = {row["checkpoint"]: row for row in summary["checkpoints"]}
    assert by_checkpoint[3]["status"] == "complete"
    assert by_checkpoint[3]["open_items"] == []
    assert str(evidence_path) in by_checkpoint[3]["evidence"]
    assert by_checkpoint[4]["status"] == "complete"
    assert by_checkpoint[4]["open_items"] == []
    assert str(evidence_path) in by_checkpoint[4]["evidence"]
