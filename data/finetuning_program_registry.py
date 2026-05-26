"""Generate a structured registry for the repo's finetuning program."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

ARTIFACT_TYPE = "finetuning_program_registry"
DEFAULT_OUTPUT = Path("docs/data_artifacts/finetuning_program_registry.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/finetuning_program_registry.manifest.json")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _stages() -> list[dict[str, Any]]:
    return [
        {
            "stage_id": 0,
            "stage_name": "Direct SQL control",
            "method_key": "direct_sql_control",
            "blog_idea": "direct SQL control",
            "training_target": "direct_sql_control",
            "primary_benchmark": "prepared",
            "benchmark_surfaces": ["prepared"],
            "control_arm": None,
            "prepared_artifacts": [
                "data.prepare",
                "data/processed/eval_cosql_dev_100.jsonl",
            ],
            "comparison_contracts": [
                "eval.run_eval",
                "eval.claim_ledger",
            ],
            "trainer_contract": {
                "expected_training_target": "direct_sql_control",
                "evaluation_mode": "non_oracle_generation",
                "expected_benchmark": "prepared",
            },
            "leakage_boundaries": [
                "no_future_turn_content",
                "no_oracle_planner_hints",
                "no_oracle_pruned_semantic_context",
            ],
            "claim_boundary": "Proxy-only control arm until later same-row comparisons or Stage 6 benchmarks beat hosted baselines.",
        },
        {
            "stage_id": 1,
            "stage_name": "Planner supervision",
            "method_key": "planner_supervision",
            "blog_idea": "planner-first intermediate state",
            "training_target": "planner_supervision",
            "primary_benchmark": "prepared",
            "benchmark_surfaces": ["prepared"],
            "control_arm": "direct_sql_control",
            "prepared_artifacts": [
                "data.prepare",
                "eval.planner_eval",
                "gold_plan",
            ],
            "comparison_contracts": [
                "eval.planner_eval",
                "eval.planner_optimize",
                "eval.run_local_planner_eval",
            ],
            "trainer_contract": {
                "expected_training_target": "planner_supervision",
                "evaluation_mode": "non_oracle_generation",
                "expected_benchmark": "prepared",
            },
            "leakage_boundaries": [
                "no_future_turn_content",
                "gold_plan_allowed_only_as_label",
                "no_reference_sql_in_prompt",
            ],
            "claim_boundary": "Planner quality is a prerequisite surface, not a SQL win by itself.",
        },
        {
            "stage_id": 2,
            "stage_name": "Predicted-planner SQL",
            "method_key": "predicted_planner_sql",
            "blog_idea": "planner-to-SQL execution",
            "training_target": "predicted_planner",
            "primary_benchmark": "prepared",
            "benchmark_surfaces": ["prepared"],
            "control_arm": "direct_sql_control",
            "prepared_artifacts": [
                "eval.planner_eval",
                "data/processed/eval_cosql_dev_predicted_planner_100.jsonl",
            ],
            "comparison_contracts": [
                "eval.run_predicted_planner_comparison",
                "eval.run_local_predicted_planner_comparison",
            ],
            "trainer_contract": {
                "expected_training_target": "predicted_planner",
                "evaluation_mode": "predicted_planner",
                "expected_benchmark": "prepared",
            },
            "leakage_boundaries": [
                "no_future_turn_content",
                "predicted_plan_must_be_non_oracle",
                "same_row_pairing_against_direct_sql",
            ],
            "claim_boundary": "Only a positive same-row comparison against direct SQL can clear this method claim.",
        },
        {
            "stage_id": 3,
            "stage_name": "Semantic-layer tuning",
            "method_key": "semantic_layer",
            "blog_idea": "semantic-layer state",
            "training_target": "semantic_layer",
            "primary_benchmark": "synthetic_semantic_layer",
            "benchmark_surfaces": ["synthetic_semantic_layer"],
            "control_arm": "direct_sql_control",
            "prepared_artifacts": [
                "data.semantic_layer_dataset",
                "data.semantic_proxy_dataset",
                "docs/data_artifacts/value_index_cosql_dev_100.jsonl",
            ],
            "comparison_contracts": [
                "eval.run_local_semantic_layer_comparison",
                "eval.run_local_semantic_proxy_comparison",
            ],
            "trainer_contract": {
                "expected_training_target": "semantic_layer",
                "evaluation_mode": "non_oracle_generation",
                "expected_benchmark": "synthetic_semantic_layer",
            },
            "leakage_boundaries": [
                "no_future_turn_content",
                "no_reference_sql_in_prompt",
                "no_expected_rows_in_prompt",
                "no_gold_metric_dsl_in_prompt",
            ],
            "claim_boundary": "Semantic artifacts only matter if same-row comparison beats the direct control without oracle pruning.",
        },
        {
            "stage_id": 4,
            "stage_name": "MEASURE()-preserving metric DSL",
            "method_key": "metric_dsl",
            "blog_idea": "MEASURE()-preserving DSL",
            "training_target": "metric_dsl",
            "primary_benchmark": "synthetic_metric_dsl_bootstrap",
            "benchmark_surfaces": [
                "synthetic_metric_dsl_bootstrap",
                "metric_dsl_direct_sql",
            ],
            "control_arm": "direct_sql_control",
            "prepared_artifacts": [
                "data.metric_dsl_dataset",
                "data.metric_dsl_direct_sql_dataset",
                "docs/data_artifacts/metric_dsl_training_rows.jsonl",
            ],
            "comparison_contracts": [
                "eval.metric_dsl_eval",
                "eval.run_metric_dsl_comparison",
                "eval.run_local_metric_dsl_comparison",
            ],
            "trainer_contract": {
                "expected_training_target": "metric_dsl",
                "evaluation_mode": "metric_dsl",
                "expected_benchmark": "synthetic_metric_dsl_bootstrap",
            },
            "leakage_boundaries": [
                "no_future_turn_content",
                "no_reference_sql_in_prompt",
                "no_compiled_sql_in_prompt",
                "measure_preservation_scored_separately_from_sql",
            ],
            "claim_boundary": "A DSL parse/compile win is not enough; compiled SQL must beat same-row direct SQL on metric-heavy rows.",
        },
        {
            "stage_id": 5,
            "stage_name": "Generated-history recovery",
            "method_key": "behavior_recovery",
            "blog_idea": "generated-history recovery",
            "training_target": "behavior_recovery",
            "primary_benchmark": "synthetic_behavior_recovery",
            "benchmark_surfaces": ["synthetic_behavior_recovery", "prepared"],
            "control_arm": "direct_sql_control",
            "prepared_artifacts": [
                "data.behavior_recovery_dataset",
                "data.behavior_recovery_proxy_dataset",
                "generated_history_trace",
            ],
            "comparison_contracts": [
                "eval.run_local_behavior_recovery_comparison",
                "eval.run_local_rollout_comparison",
                "eval.compare_rollout_history",
            ],
            "trainer_contract": {
                "expected_training_target": "behavior_recovery",
                "evaluation_mode": "non_oracle_generation",
                "expected_benchmark": "synthetic_behavior_recovery",
            },
            "leakage_boundaries": [
                "no_future_turn_content",
                "no_reference_sql_in_prompt",
                "no_repair_labels_in_prompt",
                "generated_history_claims_require_rollout_eval",
            ],
            "claim_boundary": "Teacher-forced history cannot support a recovery claim; rollout comparison is required.",
        },
        {
            "stage_id": 6,
            "stage_name": "Hosted and BIRD-Interact comparison",
            "method_key": "hosted_and_bird_benchmark",
            "blog_idea": "hosted and BIRD-Interact benchmark gate",
            "training_target": None,
            "primary_benchmark": "bird_interact_transfer",
            "benchmark_surfaces": ["prepared", "bird_interact_transfer"],
            "control_arm": "best_local_candidate_from_stage_0_to_5",
            "prepared_artifacts": [
                "data.hosted_baseline_dataset",
                "data.bird_interact_transfer_dataset",
            ],
            "comparison_contracts": [
                "eval.run_hosted_baseline_comparison",
                "eval.run_bird_interact_comparison",
            ],
            "trainer_contract": None,
            "leakage_boundaries": [
                "no_future_turn_content",
                "non_oracle_generation_only",
                "result_manifests_must_match_frozen_input_contract_hash",
            ],
            "claim_boundary": "This is the only stage that can support local-vs-hosted or BIRD-Interact competitiveness language.",
        },
    ]


def build_finetuning_program_registry() -> dict[str, Any]:
    stages = _stages()
    evaluation_modes = sorted(
        {
            stage["trainer_contract"]["evaluation_mode"]
            for stage in stages
            if stage.get("trainer_contract") is not None
        }
    )
    benchmarks = sorted(
        {
            benchmark
            for stage in stages
            for benchmark in stage.get("benchmark_surfaces", [stage["primary_benchmark"]])
        }
    )
    registry = {
        "artifact_type": ARTIFACT_TYPE,
        "source_docs": [
            "docs/research_goal.md",
            "docs/methodology.md",
            "docs/finetuning_ladder.md",
        ],
        "stage_count": len(stages),
        "blog_idea_count": len(stages),
        "evaluation_modes": evaluation_modes,
        "benchmarks": benchmarks,
        "stages": stages,
    }
    return registry


def write_finetuning_program_registry(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    registry = build_finetuning_program_registry()
    _write_json(output_path, registry)
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "source_docs": registry["source_docs"],
        "stage_count": registry["stage_count"],
        "blog_idea_count": registry["blog_idea_count"],
        "evaluation_modes": registry["evaluation_modes"],
        "benchmarks": registry["benchmarks"],
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    manifest = write_finetuning_program_registry(
        output_path=args.output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(f"Wrote {manifest['stage_count']} finetuning stages to {args.output}")
    print(f"Wrote manifest to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
