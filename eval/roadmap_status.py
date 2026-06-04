"""Summarize roadmap checkpoint status from current evidence contracts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from eval.checkpoint3_artifact_audit import audit_checkpoint3_artifacts
from eval.experiment_registry import (
    DEFAULT_EXPERIMENT_REGISTRY,
    experiment_registry_map,
)

DEFAULT_CHECKPOINT3_CONFIG = Path("configs/direct_sql_full_non_oracle.yaml")
DEFAULT_CHECKPOINT3_EVIDENCE = Path("docs/training_runs/direct_sql_full_eval_20260531.json")
DEFAULT_STRUCTURED_BRIEF_COMPARISON_EVIDENCE = Path(
    "docs/training_runs/structured_brief_clean_holdout_comparison_20260602.json"
)
DEFAULT_METRIC_DSL_READINESS_EVIDENCE = Path(
    "docs/training_runs/metric_dsl_clean_holdout_readiness_20260602.json"
)
DEFAULT_METRIC_DSL_GOLD_LABEL_EVIDENCE = Path(
    "docs/training_runs/metric_dsl_gold_labels_20260602.json"
)
DEFAULT_METRIC_DSL_PREDICTION_INPUT_EVIDENCE = Path(
    "docs/training_runs/metric_dsl_clean_holdout_prediction_inputs_20260602.json"
)
DEFAULT_METRIC_DSL_COMPARISON_EVIDENCE = Path(
    "docs/training_runs/metric_dsl_clean_holdout_comparison_20260602.json"
)
DEFAULT_GENERATED_HISTORY_RECOVERY_READINESS_EVIDENCE = Path(
    "docs/training_runs/generated_history_recovery_readiness_20260602.json"
)
DEFAULT_GENERATED_HISTORY_RECOVERY_COMPARISON_EVIDENCE = Path(
    "docs/training_runs/generated_history_recovery_clean_holdout_comparison_20260603.json"
)
DEFAULT_HOSTED_TRANSFER_COMPARISON_EVIDENCE = Path(
    "docs/training_runs/hosted_transfer_openrouter_sonnet_4_6_20260603.json"
)
DEFAULT_SEMANTIC_CONTEXT_TRANSFER_INPUT_MANIFEST = Path(
    "docs/data_artifacts/semantic_context_transfer_cp10_limit12.manifest.json"
)
DEFAULT_SEMANTIC_CONTEXT_TRANSFER_PREFLIGHT = Path(
    "docs/training_runs/semantic_context_transfer_cp10_limit12_preflight_20260604.json"
)
DEFAULT_SEMANTIC_CONTEXT_TRANSFER_LOCAL_EVIDENCE = Path(
    "docs/training_runs/semantic_context_transfer_structured_brief_local_20260604.json"
)
DEFAULT_SEMANTIC_CONTEXT_TRANSFER_RAW_QWEN_EVIDENCE = Path(
    "docs/training_runs/semantic_context_transfer_raw_qwen_local_20260604.json"
)
METRIC_DSL_MIN_COMPARABLE_ROWS = 24


CHECKPOINTS: dict[int, str] = {
    0: "Freeze The Current State",
    1: "Simplify The Research Loop",
    2: "Establish Honest Dataset Roles",
    3: "Rebuild Baselines At Real Scale",
    4: "Let Failure Analysis Choose Methods",
    5: "Structured Query Brief SFT",
    6: "Semantic Layer And Value Grounding",
    7: "Metric DSL",
    8: "Generated-History Recovery",
    9: "Hosted And Target Benchmark Transfer",
    10: "Semantic Context Transfer",
}


def _entry(
    checkpoint: int,
    *,
    status: str,
    evidence: Sequence[str],
    open_items: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "checkpoint": checkpoint,
        "name": CHECKPOINTS[checkpoint],
        "status": status,
        "evidence": list(evidence),
        "open_items": list(open_items),
    }


def _experiment_status(
    experiments: Mapping[str, dict[str, Any]], experiment_id: str
) -> str:
    return str(experiments[experiment_id]["status"])


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_from_config_root(config_path: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return config_path.resolve().parents[1] / path


def _positive_int(value: Any) -> bool:
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def _recorded_checkpoint3_complete(evidence: Mapping[str, Any] | None) -> bool:
    if not evidence:
        return False
    if evidence.get("artifact_type") != "direct_sql_full_endpoint_evidence":
        return False
    if not (evidence.get("checkpoint3_artifact_audit") or {}).get("ok"):
        return False
    manifests = evidence.get("result_manifests") or {}
    required = {
        "base_proxy_dev_seen",
        "lora_proxy_dev_seen",
        "base_clean_holdout",
        "lora_clean_holdout",
        "base_generated_history_rollout",
        "lora_generated_history_rollout",
    }
    return required.issubset(manifests) and all(
        _positive_int((manifests.get(name) or {}).get("row_count")) for name in required
    )


def _analysis_manifest_path(checkpoint3_config_path: Path) -> Path:
    base_dir = checkpoint3_config_path.resolve().parents[1]
    config = yaml.safe_load(checkpoint3_config_path.read_text(encoding="utf-8"))
    entries = (config.get("checkpoint3_artifacts") or {}).get("result_manifests") or {}
    base_entry = entries.get("base_clean_holdout") or {}
    base_manifest = Path(base_entry.get("manifest", ""))
    if not base_manifest.is_absolute():
        base_manifest = base_dir / base_manifest
    return base_manifest.parent / "clean_holdout_failure_analysis" / "manifest.json"


def _analysis_complete(analysis: Mapping[str, Any] | None) -> bool:
    if not analysis:
        return False
    if analysis.get("artifact_type") != "clean_holdout_failure_analysis":
        return False
    if analysis.get("split_role") != "clean_local_holdout":
        return False
    row_counts = analysis.get("row_counts") or {}
    if not (_positive_int(row_counts.get("base")) and _positive_int(row_counts.get("lora"))):
        return False
    hints = analysis.get("roadmap_method_hint_counts") or {}
    return bool((hints.get("base") or {}) or (hints.get("lora") or {}))


def _structured_brief_claim_complete(evidence: Mapping[str, Any] | None) -> bool:
    if not evidence:
        return False
    if evidence.get("artifact_type") != "structured_brief_clean_holdout_comparison_claim_manifest":
        return False
    if not (evidence.get("promotion") or {}).get("promotion_ready"):
        return False
    comparison = evidence.get("comparison") or {}
    if not _positive_int(comparison.get("comparable_turns")):
        return False
    value_delta = float(comparison.get("structured_brief_value_delta_vs_direct_sql") or 0.0)
    strict_delta = float(comparison.get("structured_brief_strict_delta_vs_direct_sql") or 0.0)
    leakage_boundary = evidence.get("leakage_boundary") or {}
    return (
        value_delta > 0.0
        and strict_delta >= 0.0
        and leakage_boundary.get("oracle_policy") == "non_oracle_generation"
        and leakage_boundary.get("clean_holdout_reference_sql_visible_to_model") is False
        and leakage_boundary.get("scorer_labels_visible_to_model") is False
    )


def _metric_dsl_readiness_recorded(evidence: Mapping[str, Any] | None) -> bool:
    if not evidence:
        return False
    if evidence.get("artifact_type") != "metric_dsl_clean_holdout_candidate_summary":
        return False
    if not _positive_int(evidence.get("candidate_count")):
        return False
    split_roles = evidence.get("split_roles") or {}
    return _positive_int(split_roles.get("clean_local_holdout"))


def _metric_dsl_gold_labels_recorded(evidence: Mapping[str, Any] | None) -> bool:
    if not evidence:
        return False
    if evidence.get("artifact_type") != "metric_dsl_gold_label_summary":
        return False
    if not _positive_int(evidence.get("labelled_count")):
        return False
    split_roles = evidence.get("split_roles") or {}
    return _positive_int(split_roles.get("clean_local_holdout"))


def _metric_dsl_gold_labels_meet_comparison_floor(
    evidence: Mapping[str, Any] | None,
) -> bool:
    if not _metric_dsl_gold_labels_recorded(evidence):
        return False
    return int(evidence.get("labelled_count") or 0) >= METRIC_DSL_MIN_COMPARABLE_ROWS


def _metric_dsl_prediction_inputs_recorded(evidence: Mapping[str, Any] | None) -> bool:
    if not evidence:
        return False
    if evidence.get("artifact_type") != "metric_dsl_clean_holdout_prediction_input_summary":
        return False
    paired_row_count = int(evidence.get("paired_row_count") or 0)
    if paired_row_count < METRIC_DSL_MIN_COMPARABLE_ROWS:
        return False
    split_roles = evidence.get("split_roles") or {}
    return int(split_roles.get("clean_local_holdout") or 0) >= paired_row_count


def _metric_dsl_comparison_recorded(evidence: Mapping[str, Any] | None) -> bool:
    if not evidence:
        return False
    if evidence.get("artifact_type") != "metric_dsl_clean_holdout_comparison_summary":
        return False
    comparable_row_count = int(evidence.get("comparable_row_count") or 0)
    if comparable_row_count < METRIC_DSL_MIN_COMPARABLE_ROWS:
        return False
    split_roles = evidence.get("split_roles") or {}
    return int(split_roles.get("clean_local_holdout") or 0) >= comparable_row_count


def _metric_dsl_comparison_promoted(evidence: Mapping[str, Any] | None) -> bool:
    return _metric_dsl_comparison_recorded(evidence) and bool(
        evidence.get("metric_dsl_promotion_ready")
    )


def _metric_dsl_open_items(
    readiness_evidence: Mapping[str, Any] | None,
    gold_label_evidence: Mapping[str, Any] | None,
    prediction_input_evidence: Mapping[str, Any] | None,
    comparison_evidence: Mapping[str, Any] | None,
) -> list[str]:
    items = []
    gold_labels_ready = _metric_dsl_gold_labels_meet_comparison_floor(gold_label_evidence)
    prediction_inputs_ready = _metric_dsl_prediction_inputs_recorded(
        prediction_input_evidence
    )
    comparison_recorded = _metric_dsl_comparison_recorded(comparison_evidence)
    comparison_promoted = _metric_dsl_comparison_promoted(comparison_evidence)
    if _metric_dsl_readiness_recorded(readiness_evidence):
        for blocker in readiness_evidence.get("readiness_blockers") or {}:
            if gold_labels_ready and blocker == "structured gold Metric DSL labels missing":
                continue
            items.append(str(blocker))
    if comparison_promoted or comparison_recorded:
        return items
    if gold_labels_ready:
        if prediction_inputs_ready:
            items.append("Metric DSL generated outputs are missing")
        else:
            items.append("Metric DSL prediction inputs are missing")
    else:
        items.append(
            f"at least {METRIC_DSL_MIN_COMPARABLE_ROWS} structured gold Metric DSL labels needed"
        )
    items.append("Metric DSL clean-holdout promotion policy must pass")
    return items


def _generated_history_recovery_readiness_recorded(
    evidence: Mapping[str, Any] | None,
) -> bool:
    if not evidence:
        return False
    if evidence.get("artifact_type") != "generated_history_recovery_readiness_summary":
        return False
    if not _positive_int(evidence.get("candidate_dialog_count")):
        return False
    split_roles = evidence.get("split_roles") or {}
    return _positive_int(split_roles.get("clean_local_holdout"))


def _generated_history_recovery_open_items(
    evidence: Mapping[str, Any] | None,
    comparison_evidence: Mapping[str, Any] | None,
) -> list[str]:
    if _generated_history_recovery_comparison_recorded(comparison_evidence):
        return []
    items = []
    if _generated_history_recovery_readiness_recorded(evidence):
        items.extend(str(blocker) for blocker in (evidence.get("readiness_blockers") or {}))
    items.append("multi-dialog generated-history recovery win is still missing")
    return items


def _generated_history_recovery_comparison_recorded(
    evidence: Mapping[str, Any] | None,
) -> bool:
    if not evidence:
        return False
    if evidence.get("artifact_type") != "generated_history_recovery_clean_holdout_comparison_summary":
        return False
    comparable_row_count = int(evidence.get("comparable_row_count") or 0)
    evaluated_dialog_count = int(evidence.get("evaluated_dialog_count") or 0)
    if comparable_row_count <= 0 or evaluated_dialog_count <= 1:
        return False
    split_roles = evidence.get("split_roles") or {}
    if int(split_roles.get("clean_local_holdout") or 0) != comparable_row_count:
        return False
    value_delta = float(evidence.get("behavior_recovery_value_delta_vs_direct_sql") or 0.0)
    return (
        value_delta > 0.0
        and evidence.get("oracle_policy") == "non_oracle_generation"
        and evidence.get("history_policy") == "model_generated_sql_rollout"
        and evidence.get("reference_sql_visible_to_model_prompt") is False
        and evidence.get("scorer_labels_visible_to_model_prompt") is False
    )


def _hosted_transfer_comparison_recorded(evidence: Mapping[str, Any] | None) -> bool:
    if not evidence:
        return False
    if evidence.get("artifact_type") != "hosted_transfer_comparison_summary":
        return False
    if int(evidence.get("checkpoint") or 0) != 9:
        return False
    comparable_row_count = int(evidence.get("comparable_row_count") or 0)
    if comparable_row_count <= 0:
        return False
    split_roles = evidence.get("split_roles") or {}
    if int(split_roles.get("clean_local_holdout") or 0) != comparable_row_count:
        return False
    for field in (
        "local_value_delta_vs_base_qwen",
        "local_strict_delta_vs_base_qwen",
        "local_value_delta_vs_hosted",
        "local_strict_delta_vs_hosted",
        "local_base_qwen_value_execution_accuracy",
        "local_base_qwen_strict_execution_accuracy",
        "hosted_value_execution_accuracy",
        "hosted_strict_execution_accuracy",
    ):
        if evidence.get(field) is None:
            return False
    hosted_cost = evidence.get("hosted_total_estimated_generation_cost_usd")
    hosted_latency = evidence.get("hosted_mean_latency_ms")
    return (
        evidence.get("oracle_policy") == "non_oracle_generation"
        and evidence.get("history_policy") == "model_generated_sql_rollout"
        and evidence.get("local_same_row_identity_verified") is True
        and evidence.get("reference_sql_visible_to_model_prompt") is False
        and evidence.get("scorer_labels_visible_to_model_prompt") is False
        and evidence.get("future_turns_visible_to_model_prompt") is False
        and str(evidence.get("hosted_endpoint") or "").startswith("https://")
        and bool(evidence.get("hosted_model_name"))
        and bool(evidence.get("local_base_qwen_model_name"))
        and bool(evidence.get("input_sha256"))
        and hosted_cost is not None
        and float(hosted_cost) >= 0.0
        and hosted_latency is not None
        and float(hosted_latency) > 0.0
    )


def _semantic_context_transfer_inputs_recorded(
    manifest: Mapping[str, Any] | None,
    preflight: Mapping[str, Any] | None,
) -> bool:
    if not manifest or not preflight:
        return False
    if manifest.get("artifact_type") != "semantic_context_transfer_rollout_inputs":
        return False
    if preflight.get("artifact_type") != "semantic_context_transfer_preflight":
        return False
    if int(manifest.get("checkpoint") or 0) != 10:
        return False
    if int(manifest.get("dialog_count") or 0) <= 0:
        return False
    if int(preflight.get("row_count") or 0) <= 0:
        return False
    return (
        manifest.get("preflight_status") == "ready_for_semantic_context_rollout_pair"
        and preflight.get("status") == "ready_for_semantic_context_rollout_pair"
        and manifest.get("rollout_target_history_policy") == "model_generated_sql_rollout"
        and preflight.get("rollout_target_history_policy") == "model_generated_sql_rollout"
        and preflight.get("value_index_index_source") == "database_contents"
        and manifest.get("oracle_policy") == "non_oracle_generation"
    )


def _semantic_context_transfer_local_evidence_recorded(
    evidence: Mapping[str, Any] | None,
) -> bool:
    if not evidence:
        return False
    normal_context = evidence.get("normal_context") or {}
    semantic_context = evidence.get("semantic_context") or {}
    comparison = evidence.get("comparison") or {}
    return (
        evidence.get("artifact_type") == "semantic_context_transfer_local_evidence"
        and int(evidence.get("checkpoint") or 0) == 10
        and evidence.get("evaluation_mode") == "non_oracle_generation"
        and evidence.get("oracle_allowed") is False
        and evidence.get("history_policy") == "model_generated_sql_rollout"
        and int(evidence.get("row_count") or 0) > 0
        and normal_context.get("prompt_variant") == "normal_schema_context"
        and semantic_context.get("prompt_variant")
        == "schema_context_plus_database_value_retrieval"
        and comparison.get("comparison_role") in {"best_local", "raw_qwen"}
        and "semantic_context_value_delta_vs_normal" in comparison
        and "semantic_context_strict_delta_vs_normal" in comparison
        and bool(comparison.get("comparison_manifest_path"))
    )


def summarize_roadmap_status(
    *,
    experiment_registry_path: Path = DEFAULT_EXPERIMENT_REGISTRY,
    checkpoint3_config_path: Path = DEFAULT_CHECKPOINT3_CONFIG,
    checkpoint3_evidence_path: Path = DEFAULT_CHECKPOINT3_EVIDENCE,
    structured_brief_comparison_evidence_path: Path = (
        DEFAULT_STRUCTURED_BRIEF_COMPARISON_EVIDENCE
    ),
    metric_dsl_readiness_evidence_path: Path = DEFAULT_METRIC_DSL_READINESS_EVIDENCE,
    metric_dsl_gold_label_evidence_path: Path = DEFAULT_METRIC_DSL_GOLD_LABEL_EVIDENCE,
    metric_dsl_prediction_input_evidence_path: Path = (
        DEFAULT_METRIC_DSL_PREDICTION_INPUT_EVIDENCE
    ),
    metric_dsl_comparison_evidence_path: Path = DEFAULT_METRIC_DSL_COMPARISON_EVIDENCE,
    generated_history_recovery_readiness_evidence_path: Path = (
        DEFAULT_GENERATED_HISTORY_RECOVERY_READINESS_EVIDENCE
    ),
    generated_history_recovery_comparison_evidence_path: Path = (
        DEFAULT_GENERATED_HISTORY_RECOVERY_COMPARISON_EVIDENCE
    ),
    hosted_transfer_comparison_evidence_path: Path = (
        DEFAULT_HOSTED_TRANSFER_COMPARISON_EVIDENCE
    ),
    semantic_context_transfer_input_manifest_path: Path = (
        DEFAULT_SEMANTIC_CONTEXT_TRANSFER_INPUT_MANIFEST
    ),
    semantic_context_transfer_preflight_path: Path = (
        DEFAULT_SEMANTIC_CONTEXT_TRANSFER_PREFLIGHT
    ),
    semantic_context_transfer_local_evidence_path: Path = (
        DEFAULT_SEMANTIC_CONTEXT_TRANSFER_LOCAL_EVIDENCE
    ),
    semantic_context_transfer_raw_qwen_evidence_path: Path = (
        DEFAULT_SEMANTIC_CONTEXT_TRANSFER_RAW_QWEN_EVIDENCE
    ),
) -> dict[str, Any]:
    """Return checkpoint statuses derived from current repo evidence."""

    experiments = experiment_registry_map(experiment_registry_path)
    checkpoint3 = audit_checkpoint3_artifacts(checkpoint3_config_path)
    resolved_checkpoint3_evidence_path = _resolve_from_config_root(
        checkpoint3_config_path, checkpoint3_evidence_path
    )
    recorded_checkpoint3 = _load_optional_json(resolved_checkpoint3_evidence_path)
    recorded_checkpoint3_complete = _recorded_checkpoint3_complete(recorded_checkpoint3)
    checkpoint3_status = (
        "complete" if checkpoint3.ok or recorded_checkpoint3_complete else "in_progress"
    )
    checkpoint3_open_items = [] if recorded_checkpoint3_complete else list(checkpoint3.issues)
    checkpoint3_evidence = [
        str(checkpoint3_config_path),
        "eval.checkpoint3_artifact_audit",
        "docs/training_runs/direct_sql_full_lora_20260531.json",
        f"experiment_status={_experiment_status(experiments, 'direct_sql_full_non_oracle_control')}",
    ]
    if recorded_checkpoint3:
        checkpoint3_evidence.append(str(checkpoint3_evidence_path))

    live_analysis = _load_optional_json(_analysis_manifest_path(checkpoint3_config_path))
    recorded_analysis = (
        recorded_checkpoint3.get("failure_analysis") if recorded_checkpoint3 else None
    )
    checkpoint4_complete = _analysis_complete(live_analysis) or _analysis_complete(
        recorded_analysis
    )
    checkpoint4_evidence = [
        "eval.clean_holdout_failure_analysis",
        "scripts.direct_sql_full_control analysis stage",
    ]
    if checkpoint4_complete:
        checkpoint4_evidence.append(
            "results/runs/direct_sql_full_non_oracle_control/clean_holdout_failure_analysis/manifest.json"
        )
        if recorded_checkpoint3:
            checkpoint4_evidence.append(str(checkpoint3_evidence_path))
    checkpoint4_open_items = (
        []
        if checkpoint4_complete
        else ["clean-holdout failure analysis waits for Checkpoint 3 base and LoRA manifests"]
    )
    resolved_structured_brief_evidence_path = _resolve_from_config_root(
        experiment_registry_path, structured_brief_comparison_evidence_path
    )
    structured_brief_evidence = _load_optional_json(resolved_structured_brief_evidence_path)
    structured_brief_complete = _structured_brief_claim_complete(structured_brief_evidence)
    checkpoint5_open_items = (
        []
        if structured_brief_complete
        else [
            "structured-brief clean-holdout comparison claim manifest is missing or not promotion-ready"
        ]
    )
    resolved_metric_dsl_readiness_path = _resolve_from_config_root(
        experiment_registry_path, metric_dsl_readiness_evidence_path
    )
    metric_dsl_readiness = _load_optional_json(resolved_metric_dsl_readiness_path)
    metric_dsl_readiness_recorded = _metric_dsl_readiness_recorded(metric_dsl_readiness)
    resolved_metric_dsl_gold_label_path = _resolve_from_config_root(
        experiment_registry_path, metric_dsl_gold_label_evidence_path
    )
    metric_dsl_gold_labels = _load_optional_json(resolved_metric_dsl_gold_label_path)
    metric_dsl_gold_labels_recorded = _metric_dsl_gold_labels_recorded(
        metric_dsl_gold_labels
    )
    resolved_metric_dsl_prediction_input_path = _resolve_from_config_root(
        experiment_registry_path, metric_dsl_prediction_input_evidence_path
    )
    metric_dsl_prediction_inputs = _load_optional_json(
        resolved_metric_dsl_prediction_input_path
    )
    metric_dsl_prediction_inputs_recorded = _metric_dsl_prediction_inputs_recorded(
        metric_dsl_prediction_inputs
    )
    resolved_metric_dsl_comparison_path = _resolve_from_config_root(
        experiment_registry_path, metric_dsl_comparison_evidence_path
    )
    metric_dsl_comparison = _load_optional_json(resolved_metric_dsl_comparison_path)
    metric_dsl_comparison_recorded = _metric_dsl_comparison_recorded(
        metric_dsl_comparison
    )
    metric_dsl_complete = metric_dsl_comparison_recorded
    metric_dsl_evidence = [
        f"experiment_status={_experiment_status(experiments, 'metric_dsl_vs_direct_sql')}",
        "eval.compare_metric_dsl_direct_sql promotion policy",
        "parser/evaluator and two-row fixtures exist",
    ]
    if metric_dsl_readiness_recorded:
        metric_dsl_evidence.extend(
            [
                "data.metric_dsl_clean_holdout_readiness",
                str(metric_dsl_readiness_evidence_path),
            ]
        )
    if metric_dsl_gold_labels_recorded:
        metric_dsl_evidence.extend(
            [
                "data.metric_dsl_gold_labels",
                str(metric_dsl_gold_label_evidence_path),
            ]
        )
    if metric_dsl_prediction_inputs_recorded:
        metric_dsl_evidence.extend(
            [
                "data.metric_dsl_clean_holdout_prediction_inputs",
                str(metric_dsl_prediction_input_evidence_path),
            ]
        )
    if metric_dsl_comparison_recorded:
        metric_dsl_evidence.extend(
            [
                "eval.run_metric_dsl_comparison clean-holdout generated-output comparison",
                str(metric_dsl_comparison_evidence_path),
            ]
        )
    resolved_generated_history_recovery_readiness_path = _resolve_from_config_root(
        experiment_registry_path, generated_history_recovery_readiness_evidence_path
    )
    generated_history_recovery_readiness = _load_optional_json(
        resolved_generated_history_recovery_readiness_path
    )
    generated_history_recovery_readiness_recorded = (
        _generated_history_recovery_readiness_recorded(
            generated_history_recovery_readiness
        )
    )
    resolved_generated_history_recovery_comparison_path = _resolve_from_config_root(
        experiment_registry_path, generated_history_recovery_comparison_evidence_path
    )
    generated_history_recovery_comparison = _load_optional_json(
        resolved_generated_history_recovery_comparison_path
    )
    generated_history_recovery_comparison_recorded = (
        _generated_history_recovery_comparison_recorded(
            generated_history_recovery_comparison
        )
    )
    generated_history_recovery_evidence = [
        f"experiment_status={_experiment_status(experiments, 'generated_history_recovery_vs_direct')}",
        "rollout evaluators and one-row recovery diagnostics exist",
    ]
    if generated_history_recovery_readiness_recorded:
        generated_history_recovery_evidence.extend(
            [
                "data.generated_history_recovery_readiness",
                str(generated_history_recovery_readiness_evidence_path),
            ]
        )
    if generated_history_recovery_comparison_recorded:
        generated_history_recovery_evidence.extend(
            [
                "eval.run_behavior_recovery_comparison clean-holdout generated-history comparison",
                str(generated_history_recovery_comparison_evidence_path),
            ]
        )
    resolved_hosted_transfer_comparison_path = _resolve_from_config_root(
        experiment_registry_path, hosted_transfer_comparison_evidence_path
    )
    hosted_transfer_comparison = _load_optional_json(
        resolved_hosted_transfer_comparison_path
    )
    hosted_transfer_comparison_recorded = _hosted_transfer_comparison_recorded(
        hosted_transfer_comparison
    )
    hosted_transfer_evidence = [
        f"experiment_status={_experiment_status(experiments, 'hosted_bird_interact_transfer')}",
        "external target split manifests are pending records",
    ]
    if hosted_transfer_comparison_recorded:
        hosted_transfer_evidence.extend(
            [
                "OpenRouter anthropic/claude-sonnet-4.6 same-protocol hosted comparator",
                "raw Qwen same-row local baseline",
                "eval.compare_hosted_baseline same-row hosted comparison",
                str(hosted_transfer_comparison_evidence_path),
            ]
        )
    hosted_transfer_open_items = (
        []
        if hosted_transfer_comparison_recorded
        else ["hosted transfer needs an explicit target protocol and same-row hosted run"]
    )
    resolved_semantic_context_transfer_input_manifest_path = _resolve_from_config_root(
        experiment_registry_path,
        semantic_context_transfer_input_manifest_path,
    )
    semantic_context_transfer_input_manifest = _load_optional_json(
        resolved_semantic_context_transfer_input_manifest_path
    )
    resolved_semantic_context_transfer_preflight_path = _resolve_from_config_root(
        experiment_registry_path,
        semantic_context_transfer_preflight_path,
    )
    semantic_context_transfer_preflight = _load_optional_json(
        resolved_semantic_context_transfer_preflight_path
    )
    semantic_context_transfer_inputs_recorded = (
        _semantic_context_transfer_inputs_recorded(
            semantic_context_transfer_input_manifest,
            semantic_context_transfer_preflight,
        )
    )
    resolved_semantic_context_transfer_local_evidence_path = _resolve_from_config_root(
        experiment_registry_path,
        semantic_context_transfer_local_evidence_path,
    )
    semantic_context_transfer_local_evidence = _load_optional_json(
        resolved_semantic_context_transfer_local_evidence_path
    )
    semantic_context_transfer_local_evidence_recorded = (
        _semantic_context_transfer_local_evidence_recorded(
            semantic_context_transfer_local_evidence
        )
    )
    resolved_semantic_context_transfer_raw_qwen_evidence_path = (
        _resolve_from_config_root(
            experiment_registry_path,
            semantic_context_transfer_raw_qwen_evidence_path,
        )
    )
    semantic_context_transfer_raw_qwen_evidence = _load_optional_json(
        resolved_semantic_context_transfer_raw_qwen_evidence_path
    )
    semantic_context_transfer_raw_qwen_evidence_recorded = (
        _semantic_context_transfer_local_evidence_recorded(
            semantic_context_transfer_raw_qwen_evidence
        )
    )
    semantic_context_transfer_evidence = [
        f"experiment_status={_experiment_status(experiments, 'semantic_context_transfer_with_hosted')}",
        "eval.semantic_context_transfer preflight and normal-vs-semantic comparison contracts",
        "eval.run_semantic_context_transfer_rollouts endpoint runner contract",
        "eval.finalize_semantic_context_transfer comparison finalizer contract",
        "eval.summarize_semantic_context_transfer four-manifest evidence summary contract",
        "local and hosted models must receive the same non-oracle context class",
    ]
    if semantic_context_transfer_inputs_recorded:
        semantic_context_transfer_evidence.extend(
            [
                "data.semantic_context_transfer_inputs bounded clean-holdout input builder",
                str(semantic_context_transfer_input_manifest_path),
                str(semantic_context_transfer_preflight_path),
            ]
        )
    if semantic_context_transfer_local_evidence_recorded:
        semantic_context_transfer_evidence.extend(
            [
                "structured-brief local semantic-context rollout evidence",
                str(semantic_context_transfer_local_evidence_path),
            ]
        )
    if semantic_context_transfer_raw_qwen_evidence_recorded:
        semantic_context_transfer_evidence.extend(
            [
                "raw Qwen local semantic-context rollout evidence",
                str(semantic_context_transfer_raw_qwen_evidence_path),
            ]
        )

    entries = [
        _entry(
            0,
            status="complete",
            evidence=[
                "docs/current_research_inventory.md",
                "gate artifacts and old readiness workflow removed from active loop",
            ],
        ),
        _entry(
            1,
            status="complete",
            evidence=[
                str(experiment_registry_path),
                "active guardrails are split integrity, leakage prevention, row matching, and manifests",
            ],
        ),
        _entry(
            2,
            status="complete",
            evidence=[
                "data/splits/cosql_train_v1.json",
                "data/splits/cosql_dev_100_proxy_seen_v1.json",
                "data/splits/cosql_dev_clean_holdout_v1.json",
                "pending external-target split manifests",
            ],
        ),
        _entry(
            3,
            status=checkpoint3_status,
            evidence=checkpoint3_evidence,
            open_items=checkpoint3_open_items,
        ),
        _entry(
            4,
            status="complete" if checkpoint4_complete else "in_progress",
            evidence=checkpoint4_evidence,
            open_items=checkpoint4_open_items,
        ),
        _entry(
            5,
            status="complete" if structured_brief_complete else "in_progress",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'structured_brief_sql_vs_direct')}",
                "data.structured_brief_training_rows",
                "docs/training_runs/structured_brief_train_data_path_20260602.json",
                str(structured_brief_comparison_evidence_path),
                "docs/training_runs/structured_brief_full_20260602.json",
                "eval.compare_structured_brief_direct_sql promotion policy",
                "docs/research_roadmap.md structured brief reset",
                "old predicted-planner comparison artifacts removed from active evidence",
            ],
            open_items=checkpoint5_open_items,
        ),
        _entry(
            6,
            status="complete",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'semantic_value_retrieval_vs_direct')}",
                "eval.compare_semantic_value_retrieval promotion policy",
                "database-derived value index and semantic retrieval inputs exist",
                "docs/training_runs/semantic_value_clean_holdout_preflight_20260602.json",
                "docs/training_runs/semantic_value_clean_holdout_full_20260602.json",
                "docs/training_runs/semantic_value_regression_diagnosis_20260602.json",
                "docs/training_runs/semantic_value_pruned_preflight_20260602.json",
                "docs/training_runs/semantic_value_pruned_full_20260602.json",
            ],
        ),
        _entry(
            7,
            status="complete" if metric_dsl_complete else "in_progress",
            evidence=metric_dsl_evidence,
            open_items=_metric_dsl_open_items(
                metric_dsl_readiness,
                metric_dsl_gold_labels,
                metric_dsl_prediction_inputs,
                metric_dsl_comparison,
            ),
        ),
        _entry(
            8,
            status=(
                "complete"
                if generated_history_recovery_comparison_recorded
                else "in_progress"
            ),
            evidence=generated_history_recovery_evidence,
            open_items=_generated_history_recovery_open_items(
                generated_history_recovery_readiness,
                generated_history_recovery_comparison,
            ),
        ),
        _entry(
            9,
            status="complete" if hosted_transfer_comparison_recorded else "pending",
            evidence=hosted_transfer_evidence,
            open_items=hosted_transfer_open_items,
        ),
        _entry(
            10,
            status="in_progress",
            evidence=semantic_context_transfer_evidence,
            open_items=[
                "run OpenRouter Claude Sonnet 4.6 with semantic/value context",
                "compare semantic-context deltas against normal-context controls for each model family",
                "compare the best local semantic-context arm against Sonnet with the same semantic-context input",
            ],
        ),
    ]
    return {
        "schema_version": 1,
        "artifact_type": "roadmap_checkpoint_status",
        "experiment_registry_path": str(experiment_registry_path),
        "checkpoint3_config_path": str(checkpoint3_config_path),
        "status_counts": {
            status: sum(1 for entry in entries if entry["status"] == status)
            for status in ("complete", "in_progress", "pending")
        },
        "checkpoints": entries,
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Roadmap Checkpoint Status",
        "",
        f"- Complete: {summary['status_counts']['complete']}",
        f"- In progress: {summary['status_counts']['in_progress']}",
        f"- Pending: {summary['status_counts']['pending']}",
        "",
    ]
    for checkpoint in summary["checkpoints"]:
        lines.append(
            f"## Checkpoint {checkpoint['checkpoint']}: {checkpoint['name']} "
            f"({checkpoint['status']})"
        )
        lines.append("")
        lines.append("Evidence:")
        lines.extend(f"- {item}" for item in checkpoint["evidence"])
        if checkpoint["open_items"]:
            lines.append("")
            lines.append("Open items:")
            lines.extend(f"- {item}" for item in checkpoint["open_items"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=DEFAULT_EXPERIMENT_REGISTRY)
    parser.add_argument("--checkpoint3-config", type=Path, default=DEFAULT_CHECKPOINT3_CONFIG)
    parser.add_argument(
        "--checkpoint3-evidence",
        type=Path,
        default=DEFAULT_CHECKPOINT3_EVIDENCE,
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()

    summary = summarize_roadmap_status(
        experiment_registry_path=args.experiments,
        checkpoint3_config_path=args.checkpoint3_config,
        checkpoint3_evidence_path=args.checkpoint3_evidence,
    )
    if args.format == "markdown":
        print(_render_markdown(summary), end="")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
