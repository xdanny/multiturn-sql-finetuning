"""Generate a structured registry for the repo's finetuning program."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_TYPE = "finetuning_program_registry"
DEFAULT_OUTPUT = Path("docs/data_artifacts/finetuning_program_registry.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/finetuning_program_registry.manifest.json")
DEFAULT_SCORECARD_OUTPUT = Path("docs/data_artifacts/finetuning_stage_scorecard.md")
DEFAULT_CLAIM_LEDGER = Path("docs/claim_ledgers/cosql_dev_100.jsonl")


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
            "learning_focus": "direct SQL behavior with no planner or semantic shortcuts",
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
            "evidence_contract": {
                "claim_ids": [
                    "qwen35_9b_base_cosql_dev_100turns",
                    "multiturn_sql_100_cosql_dev_100turns",
                ],
                "artifact_paths": [
                    "docs/result_manifests/cosql_dev_100_proxy.json",
                ],
                "mode": "best_value_accuracy",
                "next_required_artifact": None,
                "next_command": None,
            },
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
            "win_condition": "establish a stable non-oracle control arm on the fixed prepared slice",
            "claim_boundary": "Proxy-only control arm until later same-row comparisons or Stage 6 benchmarks beat hosted baselines.",
        },
        {
            "stage_id": 1,
            "stage_name": "Planner supervision",
            "method_key": "planner_supervision",
            "blog_idea": "planner-first intermediate state",
            "learning_focus": "tables, columns, joins, projection shape, and duplicate policy before SQL generation",
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
            "evidence_contract": {
                "claim_ids": ["planner_lexical_schema_baseline"],
                "artifact_paths": [
                    "docs/planner_baseline_cosql_dev_100_summary.json",
                ],
                "mode": "planner_macro",
                "next_required_artifact": None,
                "next_command": None,
            },
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
            "win_condition": "planner-quality metrics improve before SQL generation is promoted",
            "claim_boundary": "Planner quality is a prerequisite surface, not a SQL win by itself.",
        },
        {
            "stage_id": 2,
            "stage_name": "Predicted-planner SQL",
            "method_key": "predicted_planner_sql",
            "blog_idea": "planner-to-SQL execution",
            "learning_focus": "SQL generation conditioned on a non-oracle predicted plan",
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
            "evidence_contract": {
                "claim_ids": ["predicted_planner_sql_execution"],
                "artifact_paths": [
                    "data/processed/eval_cosql_dev_predicted_planner_100.jsonl",
                ],
                "mode": "pending_claim",
                "next_required_artifact": "same-protocol endpoint SQL result manifest",
                "next_command": "uv run python -m eval.run_predicted_planner_comparison",
            },
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
            "win_condition": "same-row SQL execution beats the direct SQL control",
            "claim_boundary": "Only a positive same-row comparison against direct SQL can clear this method claim.",
        },
        {
            "stage_id": 3,
            "stage_name": "Semantic-layer tuning",
            "method_key": "semantic_layer",
            "blog_idea": "semantic-layer state",
            "learning_focus": "governed entities, joins, grain, and value meaning that raw schema text misses",
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
            "evidence_contract": {
                "artifact_paths": [
                    "docs/result_manifests/semantic_proxy_vs_direct_sql.json",
                    "docs/data_artifacts/semantic_layer_training_rows.manifest.json",
                    "docs/data_artifacts/semantic_layer_training_run.manifest.json",
                    "docs/data_artifacts/semantic_proxy.manifest.json",
                    "docs/data_artifacts/semantic_proxy_direct_sql.manifest.json",
                    "docs/data_artifacts/semantic_layer_direct_sql_training_run.manifest.json",
                    "docs/semantic_layer_comparison_preflight.json",
                ],
                "mode": "comparison_manifest",
                "comparison_metric_prefix": "semantic_proxy",
                "artifacts_ready_summary": "prepared semantic artifacts exist, but no checked-in same-row semantic comparison result manifest yet.",
                "next_required_artifact": "stage-specific semantic comparison manifest from semantic training artifacts",
                "next_command": "uv run python -m eval.run_local_semantic_layer_comparison",
            },
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
            "win_condition": "same-row comparison beats the direct SQL control without oracle pruning",
            "claim_boundary": "Semantic artifacts only matter if same-row comparison beats the direct control without oracle pruning.",
        },
        {
            "stage_id": 4,
            "stage_name": "MEASURE()-preserving metric DSL",
            "method_key": "metric_dsl",
            "blog_idea": "MEASURE()-preserving DSL",
            "learning_focus": "preserve governed metric intent before SQL compilation",
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
            "evidence_contract": {
                "claim_ids": [
                    "metric_dsl_evaluation_manifest",
                    "metric_dsl_beats_direct_sql",
                ],
                "artifact_paths": [
                    "docs/data_artifacts/metric_dsl_training_rows.manifest.json",
                    "docs/data_artifacts/metric_dsl_direct_sql_training_rows.manifest.json",
                ],
                "mode": "pending_claim",
                "next_required_artifact": "compared metric_dsl manifest with direct-SQL baseline",
                "next_command": "uv run python -m eval.run_local_metric_dsl_comparison",
            },
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
            "win_condition": "same-row compiled SQL beats the direct SQL control on metric-heavy rows",
            "claim_boundary": "A DSL parse/compile win is not enough; compiled SQL must beat same-row direct SQL on metric-heavy rows.",
        },
        {
            "stage_id": 5,
            "stage_name": "Generated-history recovery",
            "method_key": "behavior_recovery",
            "blog_idea": "generated-history recovery",
            "learning_focus": "repair, retry, and recovery behavior after the model's own earlier SQL",
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
            "evidence_contract": {
                "claim_ids": [
                    "model_generated_history_rollout",
                    "rollout_beats_teacher_forced_history",
                ],
                "artifact_paths": [
                    "docs/data_artifacts/behavior_recovery_training_rows.manifest.json",
                    "docs/data_artifacts/behavior_recovery_proxy.manifest.json",
                ],
                "mode": "pending_claim",
                "next_required_artifact": "same-model rollout and teacher-forced comparison manifests",
                "next_command": "uv run python -m eval.run_local_rollout_comparison",
            },
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
            "win_condition": "rollout comparison beats the same checkpoint under teacher-forced history",
            "claim_boundary": "Teacher-forced history cannot support a recovery claim; rollout comparison is required.",
        },
        {
            "stage_id": 6,
            "stage_name": "Hosted and BIRD-Interact comparison",
            "method_key": "hosted_and_bird_benchmark",
            "blog_idea": "hosted and BIRD-Interact benchmark gate",
            "learning_focus": "none; this stage validates whether the best earlier local method transfers to real benchmark gates",
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
            "evidence_contract": {
                "claim_ids": [
                    "hosted_sota_same_protocol",
                    "local_beats_hosted_same_protocol",
                    "bird_interact_local_vs_hosted",
                ],
                "artifact_paths": [
                    "docs/data_artifacts/hosted_baseline.manifest.json",
                    "docs/data_artifacts/bird_interact_transfer.manifest.json",
                ],
                "mode": "pending_claim",
                "next_required_artifact": "hosted-model and BIRD-Interact result manifests on frozen contracts",
                "next_command": "uv run python -m eval.run_hosted_baseline_comparison",
            },
            "trainer_contract": None,
            "leakage_boundaries": [
                "no_future_turn_content",
                "non_oracle_generation_only",
                "result_manifests_must_match_frozen_input_contract_hash",
            ],
            "win_condition": "the best local candidate holds up against hosted or BIRD-Interact baselines on frozen contracts",
            "claim_boundary": "This is the only stage that can support local-vs-hosted or BIRD-Interact competitiveness language.",
        },
    ]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_claim_rows() -> dict[str, dict[str, Any]]:
    path = _resolve_repo_path(DEFAULT_CLAIM_LEDGER)
    if not path.exists():
        return {}
    return {str(row.get("claim_id")): row for row in _load_jsonl(path) if row.get("claim_id")}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _format_metric(value: float) -> str:
    return f"{value:.3f}" if value < 0.6 else f"{value:.2f}"


def _summarize_current_evidence(stage: dict[str, Any], claim_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    contract = stage.get("evidence_contract") or {}
    artifact_paths = [str(path) for path in contract.get("artifact_paths", [])]
    existing_artifacts = [path for path in artifact_paths if _resolve_repo_path(path).exists()]
    comparison_artifact = artifact_paths[0] if artifact_paths else None
    claim_ids = [str(claim_id) for claim_id in contract.get("claim_ids", [])]
    rows = [claim_rows[claim_id] for claim_id in claim_ids if claim_id in claim_rows]
    mode = contract.get("mode")

    if mode == "best_value_accuracy" and rows:
        best = max(
            (float(row.get("value_execution_accuracy") or 0.0) for row in rows),
            default=0.0,
        )
        return {
            "status": "measured",
            "summary": f"measured proxy results exist; best checked-in direct-SQL proxy value accuracy is {_format_metric(best)}.",
            "claim_ids": claim_ids,
            "artifact_paths": existing_artifacts,
            "next_required_artifact": contract.get("next_required_artifact"),
            "next_command": contract.get("next_command"),
        }
    if mode == "planner_macro" and rows:
        row = rows[0]
        score = float(row.get("macro_planner_score") or 0.0)
        return {
            "status": "measured",
            "summary": f"planner-quality evidence exists; current checked-in macro planner score is {_format_metric(score)}.",
            "claim_ids": claim_ids,
            "artifact_paths": existing_artifacts,
            "next_required_artifact": contract.get("next_required_artifact"),
            "next_command": contract.get("next_command"),
        }
    if mode == "pending_claim" and rows:
        blocking_reason = next(
            (str(row.get("blocking_reason")) for row in rows if row.get("blocking_reason")),
            "required result artifacts are still missing",
        )
        prefix = "prepared artifacts exist, but " if existing_artifacts else ""
        return {
            "status": "pending",
            "summary": f"{prefix}pending claim because {blocking_reason}.",
            "claim_ids": claim_ids,
            "artifact_paths": existing_artifacts,
            "next_required_artifact": contract.get("next_required_artifact"),
            "next_command": contract.get("next_command"),
        }
    if mode == "artifacts_only" and existing_artifacts:
        return {
            "status": "artifacts_ready",
            "summary": str(contract.get("artifacts_ready_summary") or "prepared artifacts exist, but no checked-in result manifest yet."),
            "claim_ids": claim_ids,
            "artifact_paths": existing_artifacts,
            "next_required_artifact": contract.get("next_required_artifact"),
            "next_command": contract.get("next_command"),
        }
    if mode == "comparison_manifest" and existing_artifacts:
        comparison_path = _resolve_repo_path(comparison_artifact) if comparison_artifact else None
        if comparison_path is not None and comparison_path.exists():
            compared = _load_json(comparison_path)
            metrics = compared.get("metrics") or {}
            prefix = str(contract.get("comparison_metric_prefix") or "")
            value_delta = float(metrics.get(f"{prefix}_value_delta_vs_direct_sql") or 0.0)
            strict_delta = float(metrics.get(f"{prefix}_strict_delta_vs_direct_sql") or 0.0)
            row_count = int(metrics.get(f"{prefix}_comparable_row_count") or compared.get("row_count") or 0)
            value_text = f"{value_delta:+.2f}"
            strict_text = f"{strict_delta:+.2f}"
            return {
                "status": "measured",
                "summary": f"measured semantic proxy comparison exists; value delta vs direct SQL is {value_text} on {row_count} rows, while strict delta is {strict_text}.",
                "claim_ids": claim_ids,
                "artifact_paths": existing_artifacts,
                "next_required_artifact": contract.get("next_required_artifact"),
                "next_command": contract.get("next_command"),
            }
        remaining_artifacts = [
            path for path in artifact_paths[1:] if _resolve_repo_path(path).exists()
        ]
        if remaining_artifacts:
            return {
                "status": "artifacts_ready",
                "summary": str(contract.get("artifacts_ready_summary") or "prepared artifacts exist, but no checked-in result manifest yet."),
                "claim_ids": claim_ids,
                "artifact_paths": existing_artifacts,
                "next_required_artifact": contract.get("next_required_artifact"),
                "next_command": contract.get("next_command"),
            }
    return {
        "status": "missing",
        "summary": "no checked-in evidence artifact found for this stage yet.",
        "claim_ids": claim_ids,
        "artifact_paths": existing_artifacts,
        "next_required_artifact": contract.get("next_required_artifact"),
        "next_command": contract.get("next_command"),
    }


def build_finetuning_program_registry() -> dict[str, Any]:
    stages = _stages()
    claim_rows = _load_claim_rows()
    stages = [
        {
            **stage,
            "current_evidence": _summarize_current_evidence(stage, claim_rows),
        }
        for stage in stages
    ]
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


def build_finetuning_stage_scorecard(registry: dict[str, Any] | None = None) -> str:
    registry = registry or build_finetuning_program_registry()
    lines = [
        "# Finetuning Stage Scorecard",
        "",
        "This is the short companion to the finetuning ladder and the JSON registry.",
        "Each stage answers the same practical questions: what the model learns,",
        "what it is compared against, what benchmark surface it uses, and what",
        "would count as a real win.",
        "",
        "No wide summary table appears here on purpose.",
        "",
    ]
    for stage in registry["stages"]:
        lines.extend(
            [
                f"## Stage {stage['stage_id']}: {stage['stage_name']}",
                "",
                f"- Learns: {stage['learning_focus']}.",
                f"- Benchmark surfaces: {', '.join(stage['benchmark_surfaces'])}.",
                f"- Control arm: {stage['control_arm'] or 'none; this is the control arm'}.",
                f"- Prepared artifacts: {', '.join(stage['prepared_artifacts'])}.",
                f"- Comparison gate: {', '.join(stage['comparison_contracts'])}.",
                f"- Win condition: {stage['win_condition']}.",
                f"- Leakage to forbid: {', '.join(stage['leakage_boundaries'])}.",
                f"- Current evidence: {stage['current_evidence']['summary']}",
                *(
                    [
                        f"- Next evidence: {stage['current_evidence']['next_required_artifact']} via `{stage['current_evidence']['next_command']}`."
                    ]
                    if stage["current_evidence"].get("next_required_artifact")
                    and stage["current_evidence"].get("next_command")
                    else []
                ),
                f"- Claim boundary: {stage['claim_boundary']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_finetuning_program_registry(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    scorecard_output_path: Path = DEFAULT_SCORECARD_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    registry = build_finetuning_program_registry()
    _write_json(output_path, registry)
    scorecard_output_path.parent.mkdir(parents=True, exist_ok=True)
    scorecard_output_path.write_text(build_finetuning_stage_scorecard(registry))
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "source_docs": registry["source_docs"],
        "stage_count": registry["stage_count"],
        "blog_idea_count": registry["blog_idea_count"],
        "evaluation_modes": registry["evaluation_modes"],
        "benchmarks": registry["benchmarks"],
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "scorecard_output_path": str(scorecard_output_path),
        "scorecard_output_sha256": sha256_file(scorecard_output_path),
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--scorecard-output", type=Path, default=DEFAULT_SCORECARD_OUTPUT)
    args = parser.parse_args()

    manifest = write_finetuning_program_registry(
        output_path=args.output,
        manifest_path=args.manifest_output,
        scorecard_output_path=args.scorecard_output,
        command=sys.argv,
    )
    print(f"Wrote {manifest['stage_count']} finetuning stages to {args.output}")
    print(f"Wrote stage scorecard to {args.scorecard_output}")
    print(f"Wrote manifest to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
