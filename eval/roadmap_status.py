"""Summarize roadmap checkpoint status from current evidence contracts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from eval.checkpoint3_artifact_audit import audit_checkpoint3_artifacts
from eval.experiment_registry import (
    DEFAULT_EXPERIMENT_REGISTRY,
    experiment_registry_map,
)

DEFAULT_CHECKPOINT3_CONFIG = Path("configs/direct_sql_full_non_oracle.yaml")


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


def summarize_roadmap_status(
    *,
    experiment_registry_path: Path = DEFAULT_EXPERIMENT_REGISTRY,
    checkpoint3_config_path: Path = DEFAULT_CHECKPOINT3_CONFIG,
) -> dict[str, Any]:
    """Return checkpoint statuses derived from current repo evidence."""

    experiments = experiment_registry_map(experiment_registry_path)
    checkpoint3 = audit_checkpoint3_artifacts(checkpoint3_config_path)
    checkpoint3_status = "complete" if checkpoint3.ok else "in_progress"
    checkpoint3_open_items = list(checkpoint3.issues)

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
            evidence=[
                str(checkpoint3_config_path),
                "eval.checkpoint3_artifact_audit",
                f"experiment_status={_experiment_status(experiments, 'direct_sql_full_non_oracle_control')}",
            ],
            open_items=checkpoint3_open_items,
        ),
        _entry(
            4,
            status="in_progress",
            evidence=[
                "eval.clean_holdout_failure_analysis",
                "scripts.direct_sql_full_control analysis stage",
            ],
            open_items=[
                "clean-holdout failure analysis waits for Checkpoint 3 base and LoRA manifests"
            ],
        ),
        _entry(
            5,
            status="in_progress",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'predicted_planner_sql_vs_direct')}",
                "docs/training_runs/lexical_predicted_planner_limit24.json",
            ],
            open_items=[
                "planner quality must improve before another predicted-planner SQL run"
            ],
        ),
        _entry(
            6,
            status="in_progress",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'semantic_value_retrieval_vs_direct')}",
                "database-derived value index and semantic retrieval inputs exist",
            ],
            open_items=["same-row clean-holdout SQL win is still missing"],
        ),
        _entry(
            7,
            status="in_progress",
            evidence=[
                f"experiment_status={_experiment_status(experiments, 'metric_dsl_vs_direct_sql')}",
                "parser/evaluator and two-row fixtures exist",
            ],
            open_items=["metric-heavy held-out comparison against direct SQL is still missing"],
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
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    args = parser.parse_args()

    summary = summarize_roadmap_status(
        experiment_registry_path=args.experiments,
        checkpoint3_config_path=args.checkpoint3_config,
    )
    if args.format == "markdown":
        print(_render_markdown(summary), end="")
    else:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
