from __future__ import annotations

import json
from pathlib import Path

from data.finetuning_program_registry import (
    DEFAULT_MANIFEST_OUTPUT,
    DEFAULT_OUTPUT,
    DEFAULT_SCORECARD_OUTPUT,
    build_finetuning_program_registry,
    build_finetuning_stage_scorecard,
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
    assert stage_by_key["metric_dsl"]["learning_focus"] == "preserve governed metric intent before SQL compilation"
    assert stage_by_key["predicted_planner_sql"]["win_condition"] == "same-row SQL execution beats the direct SQL control"
    assert stage_by_key["direct_sql_control"]["current_evidence"]["status"] == "measured"
    assert "best checked-in direct-SQL proxy value accuracy is 0.63" in stage_by_key["direct_sql_control"]["current_evidence"]["summary"]
    assert stage_by_key["planner_supervision"]["current_evidence"]["status"] == "measured"
    assert "macro planner score is 0.571" in stage_by_key["planner_supervision"]["current_evidence"]["summary"]
    assert stage_by_key["predicted_planner_sql"]["current_evidence"]["status"] == "pending"
    assert "no predicted_planner result manifest" in stage_by_key["predicted_planner_sql"]["current_evidence"]["summary"]
    assert stage_by_key["predicted_planner_sql"]["current_evidence"]["next_required_artifact"] == "same-protocol endpoint SQL result manifest"
    assert "uv run python -m eval.run_predicted_planner_comparison" in stage_by_key["predicted_planner_sql"]["current_evidence"]["next_command"]
    assert stage_by_key["semantic_layer"]["current_evidence"]["status"] == "artifacts_ready"
    assert "no checked-in same-row semantic comparison result manifest yet" in stage_by_key["semantic_layer"]["current_evidence"]["summary"]
    assert stage_by_key["semantic_layer"]["current_evidence"]["next_required_artifact"] == "same-row semantic comparison result manifest"
    assert "uv run python -m eval.run_local_semantic_layer_comparison" in stage_by_key["semantic_layer"]["current_evidence"]["next_command"]
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
    scorecard_path = tmp_path / "finetuning_stage_scorecard.md"

    manifest = write_finetuning_program_registry(
        output_path=output_path,
        manifest_path=manifest_path,
        scorecard_output_path=scorecard_path,
        command=["uv", "run", "python", "-m", "data.finetuning_program_registry"],
    )

    payload = json.loads(output_path.read_text())
    written_manifest = json.loads(manifest_path.read_text())
    scorecard = scorecard_path.read_text()

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
    assert manifest["scorecard_output_path"] == str(scorecard_path)
    assert manifest["scorecard_output_sha256"]
    assert manifest["command"] == ["uv", "run", "python", "-m", "data.finetuning_program_registry"]
    assert written_manifest == manifest
    assert scorecard.startswith("# Finetuning Stage Scorecard")
    assert "## Stage 4: MEASURE()-preserving metric DSL" in scorecard
    assert "- Win condition: same-row compiled SQL beats the direct SQL control on metric-heavy rows." in scorecard
    assert "No wide summary table appears here on purpose." in scorecard


def test_build_finetuning_stage_scorecard_is_human_readable() -> None:
    registry = build_finetuning_program_registry()

    scorecard = build_finetuning_stage_scorecard(registry)

    assert "# Finetuning Stage Scorecard" in scorecard
    assert "## Stage 3: Semantic-layer tuning" in scorecard
    assert "- Learns: governed entities, joins, grain, and value meaning that raw schema text misses." in scorecard
    assert "- Win condition: same-row comparison beats the direct SQL control without oracle pruning." in scorecard
    assert "- Current evidence: prepared semantic artifacts exist, but no checked-in same-row semantic comparison result manifest yet." in scorecard
    assert "- Current evidence: prepared artifacts exist, but pending claim because no predicted_planner result manifest." in scorecard
    assert "- Next evidence: same-row semantic comparison result manifest via `uv run python -m eval.run_local_semantic_layer_comparison`." in scorecard
    assert "- Next evidence: same-protocol endpoint SQL result manifest via `uv run python -m eval.run_predicted_planner_comparison`." in scorecard
    assert "## Stage 6: Hosted and BIRD-Interact comparison" in scorecard
    assert "- Control arm: best_local_candidate_from_stage_0_to_5" in scorecard
    assert "No wide summary table appears here on purpose." in scorecard


def test_checked_in_finetuning_program_registry_is_current(tmp_path: Path) -> None:
    output_path = tmp_path / "finetuning_program_registry.json"
    manifest_path = tmp_path / "finetuning_program_registry.manifest.json"
    scorecard_path = tmp_path / "finetuning_stage_scorecard.md"

    manifest = write_finetuning_program_registry(
        output_path=output_path,
        manifest_path=manifest_path,
        scorecard_output_path=scorecard_path,
    )

    checked_in_payload = json.loads(DEFAULT_OUTPUT.read_text())
    checked_in_manifest = json.loads(DEFAULT_MANIFEST_OUTPUT.read_text())
    checked_in_scorecard = DEFAULT_SCORECARD_OUTPUT.read_text()

    assert checked_in_payload == json.loads(output_path.read_text())
    assert checked_in_scorecard == scorecard_path.read_text()
    for field in [
        "artifact_type",
        "stage_count",
        "blog_idea_count",
        "evaluation_modes",
        "benchmarks",
        "scorecard_output_sha256",
    ]:
        assert checked_in_manifest[field] == manifest[field]
