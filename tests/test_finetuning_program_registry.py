from __future__ import annotations

import json
from pathlib import Path

from data.finetuning_program_registry import (
    DEFAULT_MANIFEST_OUTPUT,
    DEFAULT_OUTPUT,
    build_finetuning_program_registry,
    write_finetuning_program_registry,
)


def test_build_finetuning_program_registry_covers_blog_method_ideas() -> None:
    registry = build_finetuning_program_registry()

    assert [entry["stage_id"] for entry in registry["stages"]] == [0, 1, 2, 3, 4, 5, 6]
    assert [entry["method_key"] for entry in registry["stages"]] == [
        "direct_sql_control",
        "planner_supervision",
        "predicted_planner_sql",
        "semantic_layer",
        "metric_dsl",
        "behavior_recovery",
        "hosted_and_bird_benchmark",
    ]

    stage_by_key = {entry["method_key"]: entry for entry in registry["stages"]}
    assert stage_by_key["metric_dsl"]["blog_idea"] == "MEASURE()-preserving DSL"
    assert stage_by_key["semantic_layer"]["blog_idea"] == "semantic-layer state"
    assert stage_by_key["behavior_recovery"]["blog_idea"] == "generated-history recovery"
    assert stage_by_key["hosted_and_bird_benchmark"]["blog_idea"] == "hosted and BIRD-Interact benchmark gate"

    assert stage_by_key["direct_sql_control"]["trainer_contract"] == {
        "evaluation_mode": "non_oracle_generation",
        "expected_benchmark": "prepared",
        "expected_training_target": "direct_sql_control",
    }
    assert stage_by_key["metric_dsl"]["trainer_contract"] == {
        "evaluation_mode": "metric_dsl",
        "expected_benchmark": "synthetic_metric_dsl_bootstrap",
        "expected_training_target": "metric_dsl",
    }
    assert stage_by_key["behavior_recovery"]["leakage_boundaries"] == [
        "no_future_turn_content",
        "no_reference_sql_in_prompt",
        "no_repair_labels_in_prompt",
        "generated_history_claims_require_rollout_eval",
    ]
    assert stage_by_key["hosted_and_bird_benchmark"]["trainer_contract"] is None
    assert stage_by_key["hosted_and_bird_benchmark"]["comparison_contracts"] == [
        "eval.run_hosted_baseline_comparison",
        "eval.run_bird_interact_comparison",
    ]


def test_write_finetuning_program_registry_writes_checked_artifact_shape(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "finetuning_program_registry.json"
    manifest_path = tmp_path / "finetuning_program_registry.manifest.json"

    manifest = write_finetuning_program_registry(
        output_path=output_path,
        manifest_path=manifest_path,
        command=["uv", "run", "python", "-m", "data.finetuning_program_registry"],
    )

    payload = json.loads(output_path.read_text())
    written_manifest = json.loads(manifest_path.read_text())

    assert payload["artifact_type"] == "finetuning_program_registry"
    assert payload["stage_count"] == 7
    assert payload["blog_idea_count"] == 7
    assert payload["evaluation_modes"] == [
        "metric_dsl",
        "non_oracle_generation",
        "predicted_planner",
    ]
    assert payload["benchmarks"] == [
        "bird_interact_transfer",
        "metric_dsl_direct_sql",
        "prepared",
        "synthetic_behavior_recovery",
        "synthetic_metric_dsl_bootstrap",
        "synthetic_semantic_layer",
    ]
    assert manifest["artifact_type"] == "finetuning_program_registry"
    assert manifest["stage_count"] == 7
    assert manifest["output_sha256"]
    assert manifest["command"] == ["uv", "run", "python", "-m", "data.finetuning_program_registry"]
    assert written_manifest == manifest


def test_checked_in_finetuning_program_registry_is_current(tmp_path: Path) -> None:
    output_path = tmp_path / "finetuning_program_registry.json"
    manifest_path = tmp_path / "finetuning_program_registry.manifest.json"

    manifest = write_finetuning_program_registry(
        output_path=output_path,
        manifest_path=manifest_path,
    )

    checked_in_payload = json.loads(DEFAULT_OUTPUT.read_text())
    checked_in_manifest = json.loads(DEFAULT_MANIFEST_OUTPUT.read_text())

    assert checked_in_payload == json.loads(output_path.read_text())
    for field in [
        "artifact_type",
        "stage_count",
        "blog_idea_count",
        "evaluation_modes",
        "benchmarks",
    ]:
        assert checked_in_manifest[field] == manifest[field]
