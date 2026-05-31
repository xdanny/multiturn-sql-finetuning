from __future__ import annotations

import json
from pathlib import Path

import yaml

from eval.checkpoint3_artifact_audit import audit_checkpoint3_artifacts


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _prepared_manifest(split_id: str, split_role: str, row_count: int) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "prepared_split_dataset",
        "split_id": split_id,
        "split_role": split_role,
        "row_count": row_count,
        "oracle_policy": "non_oracle_generation",
        "evaluation_modes": {"non_oracle_generation": row_count},
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
    }


def _result_manifest(*, split_id: str, split_role: str, row_hash: str, turn_hash: str) -> dict:
    return {
        "schema_version": 1,
        "benchmark": "prepared",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "row_count": 3,
        "output_sha256": "output-hash",
        "metrics": {
            "value_execution_accuracy": 0.5,
            "strict_execution_accuracy": 0.4,
            "interaction_match_rate": 0.25,
            "mean_latency_ms": 12.5,
            "total_tokens": 1000,
            "total_estimated_generation_cost_usd": 0.02,
            "split_ids": {split_id: 3},
            "split_roles": {split_role: 3},
            "split_row_ids_sha256": row_hash,
            "split_eval_turn_ids_sha256": turn_hash,
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
            "mean_latency_ms": 10.0,
        },
    }


def _write_checkpoint3_fixture(tmp_path: Path) -> Path:
    config_path = tmp_path / "configs" / "direct_sql_full_non_oracle.yaml"
    prepared_dir = tmp_path / "data" / "processed" / "direct_sql_full"
    results_dir = tmp_path / "results" / "runs" / "direct_sql_full_non_oracle_control"
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
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    prepared_payloads = {
        "cosql_train_v1": ("train", 2159),
        "cosql_dev_100_proxy_seen_v1": ("proxy_dev_seen", 100),
        "cosql_dev_clean_holdout_v1": ("clean_local_holdout", 193),
    }
    prepared_dir.mkdir(parents=True, exist_ok=True)
    for split_id, (split_role, row_count) in prepared_payloads.items():
        (prepared_dir / f"{split_id}.jsonl").write_text("{}\n", encoding="utf-8")
        _write_json(
            prepared_dir / f"{split_id}.manifest.json",
            _prepared_manifest(split_id, split_role, row_count),
        )

    _write_json(
        results_dir / "base_proxy_dev_seen.manifest.json",
        _result_manifest(
            split_id="cosql_dev_100_proxy_seen_v1",
            split_role="proxy_dev_seen",
            row_hash="proxy-rows",
            turn_hash="proxy-turns",
        ),
    )
    _write_json(
        results_dir / "lora_proxy_dev_seen.manifest.json",
        _result_manifest(
            split_id="cosql_dev_100_proxy_seen_v1",
            split_role="proxy_dev_seen",
            row_hash="proxy-rows",
            turn_hash="proxy-turns",
        ),
    )
    _write_json(
        results_dir / "base_clean_holdout.manifest.json",
        _result_manifest(
            split_id="cosql_dev_clean_holdout_v1",
            split_role="clean_local_holdout",
            row_hash="holdout-rows",
            turn_hash="holdout-turns",
        ),
    )
    _write_json(
        results_dir / "lora_clean_holdout.manifest.json",
        _result_manifest(
            split_id="cosql_dev_clean_holdout_v1",
            split_role="clean_local_holdout",
            row_hash="holdout-rows",
            turn_hash="holdout-turns",
        ),
    )
    _write_json(results_dir / "base_generated_history_rollout.manifest.json", _rollout_manifest())
    _write_json(results_dir / "lora_generated_history_rollout.manifest.json", _rollout_manifest())
    return config_path


def test_checkpoint3_artifact_audit_accepts_complete_artifacts(tmp_path) -> None:
    config_path = _write_checkpoint3_fixture(tmp_path)

    result = audit_checkpoint3_artifacts(config_path)

    assert result.ok
    assert result.issues == ()


def test_checkpoint3_artifact_audit_rejects_row_mismatch(tmp_path) -> None:
    config_path = _write_checkpoint3_fixture(tmp_path)
    lora_holdout = (
        tmp_path
        / "results"
        / "runs"
        / "direct_sql_full_non_oracle_control"
        / "lora_clean_holdout.manifest.json"
    )
    payload = json.loads(lora_holdout.read_text(encoding="utf-8"))
    payload["metrics"]["split_eval_turn_ids_sha256"] = "different-turns"
    _write_json(lora_holdout, payload)

    result = audit_checkpoint3_artifacts(config_path)

    assert not result.ok
    assert "clean_local_holdout: base/lora split_eval_turn_ids_sha256 mismatch" in result.issues


def test_checkpoint3_artifact_audit_rejects_missing_cost_metric(tmp_path) -> None:
    config_path = _write_checkpoint3_fixture(tmp_path)
    base_proxy = (
        tmp_path
        / "results"
        / "runs"
        / "direct_sql_full_non_oracle_control"
        / "base_proxy_dev_seen.manifest.json"
    )
    payload = json.loads(base_proxy.read_text(encoding="utf-8"))
    del payload["metrics"]["total_estimated_generation_cost_usd"]
    _write_json(base_proxy, payload)

    result = audit_checkpoint3_artifacts(config_path)

    assert not result.ok
    assert (
        "result_manifests.base_proxy_dev_seen: missing metric "
        "total_estimated_generation_cost_usd"
    ) in result.issues
