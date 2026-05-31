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


CHECKPOINTS: dict[int, str] = {
    0: "Freeze The Current State",
    1: "Simplify The Research Loop",
    2: "Establish Honest Dataset Roles",
    3: "Rebuild Baselines At Real Scale",
    4: "Let Failure Analysis Choose Methods",
    5: "Planner First, But Non-Oracle",
    6: "Semantic Layer And Value Grounding",
    7: "Metric DSL",
    8: "Generated-History Recovery",
    9: "Hosted And Target Benchmark Transfer",
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


def summarize_roadmap_status(
    *,
    experiment_registry_path: Path = DEFAULT_EXPERIMENT_REGISTRY,
    checkpoint3_config_path: Path = DEFAULT_CHECKPOINT3_CONFIG,
    checkpoint3_evidence_path: Path = DEFAULT_CHECKPOINT3_EVIDENCE,
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
            status="in_progress",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'predicted_planner_sql_vs_direct')}",
                "eval.planner_readiness promotion policy",
                "docs/training_runs/lexical_predicted_planner_limit24.json",
                "docs/training_runs/planner_schema_context_repair_20260531.json",
            ],
            open_items=[
                (
                    "improve non-oracle column linking and projection quality before "
                    "another predicted-planner SQL run"
                )
            ],
        ),
        _entry(
            6,
            status="in_progress",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'semantic_value_retrieval_vs_direct')}",
                "eval.compare_semantic_value_retrieval promotion policy",
                "database-derived value index and semantic retrieval inputs exist",
            ],
            open_items=["semantic clean-holdout promotion policy must pass"],
        ),
        _entry(
            7,
            status="in_progress",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'metric_dsl_vs_direct_sql')}",
                "eval.compare_metric_dsl_direct_sql promotion policy",
                "parser/evaluator and two-row fixtures exist",
            ],
            open_items=["Metric DSL clean-holdout promotion policy must pass"],
        ),
        _entry(
            8,
            status="in_progress",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'generated_history_recovery_vs_direct')}",
                "rollout evaluators and one-row recovery diagnostics exist",
            ],
            open_items=["multi-dialog generated-history recovery win is still missing"],
        ),
        _entry(
            9,
            status="pending",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'hosted_bird_interact_transfer')}",
                "external target split manifests are pending records",
            ],
            open_items=["hosted transfer waits for a local clean-holdout winner"],
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
