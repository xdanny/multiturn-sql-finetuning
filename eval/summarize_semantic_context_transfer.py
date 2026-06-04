"""Summarize completed Checkpoint 10 semantic-context transfer evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

ARTIFACT_TYPE = "semantic_context_transfer_evidence_summary"
CONTEXT_COMPARISON_ARTIFACT_TYPE = "semantic_context_transfer_comparison"
NON_ORACLE_GENERATION = "non_oracle_generation"
PREPARED_ROLLOUT = "prepared_rollout"
MODEL_GENERATED_SQL_ROLLOUT = "model_generated_sql_rollout"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _metric(manifest: dict[str, Any], name: str) -> float:
    value = (manifest.get("metrics") or {}).get(name)
    if value is None:
        raise ValueError(f"manifest {manifest.get('run_id')} is missing metric {name}")
    return float(value)


def _validate_context_comparison(manifest: dict[str, Any], *, label: str) -> None:
    if manifest.get("artifact_type") != CONTEXT_COMPARISON_ARTIFACT_TYPE:
        raise ValueError(f"{label} must be a semantic context transfer comparison")
    if manifest.get("oracle_allowed") or manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError(f"{label} must be non-oracle")
    if manifest.get("benchmark") != PREPARED_ROLLOUT:
        raise ValueError(f"{label} must use benchmark=prepared_rollout")
    if (manifest.get("metrics") or {}).get("history_policy") != MODEL_GENERATED_SQL_ROLLOUT:
        raise ValueError(f"{label} must use model-generated rollout history")
    for metric in (
        "value_execution_accuracy",
        "strict_execution_accuracy",
        "syntax_accuracy",
        "normal_context_value_execution_accuracy",
        "normal_context_strict_execution_accuracy",
        "normal_context_syntax_accuracy",
        "semantic_context_value_delta_vs_normal",
        "semantic_context_strict_delta_vs_normal",
        "semantic_context_syntax_delta_vs_normal",
        "semantic_context_transfer_comparable_row_count",
    ):
        _metric(manifest, metric)


def _validate_hosted_comparison(manifest: dict[str, Any], *, label: str) -> None:
    if manifest.get("oracle_allowed") or manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError(f"{label} hosted comparison must be non-oracle")
    if manifest.get("benchmark") != PREPARED_ROLLOUT:
        raise ValueError(f"{label} hosted comparison must use benchmark=prepared_rollout")
    if (manifest.get("metrics") or {}).get("history_policy") != MODEL_GENERATED_SQL_ROLLOUT:
        raise ValueError(f"{label} hosted comparison must use model-generated rollout history")
    for metric in (
        "value_execution_accuracy",
        "strict_execution_accuracy",
        "hosted_value_execution_accuracy",
        "hosted_strict_execution_accuracy",
        "local_value_delta_vs_hosted",
        "local_strict_delta_vs_hosted",
        "hosted_comparable_row_count",
    ):
        _metric(manifest, metric)


def _context_metrics(manifest: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    metrics = manifest.get("metrics") or {}
    semantic_value = _metric(manifest, "value_execution_accuracy")
    semantic_strict = _metric(manifest, "strict_execution_accuracy")
    semantic_syntax = _metric(manifest, "syntax_accuracy")
    normal_value = _metric(manifest, "normal_context_value_execution_accuracy")
    normal_strict = _metric(manifest, "normal_context_strict_execution_accuracy")
    normal_syntax = _metric(manifest, "normal_context_syntax_accuracy")
    return {
        f"{prefix}_model_name": manifest.get("model_name"),
        f"{prefix}_normal_value_execution_accuracy": normal_value,
        f"{prefix}_semantic_value_execution_accuracy": semantic_value,
        f"{prefix}_semantic_value_delta_vs_normal": semantic_value - normal_value,
        f"{prefix}_normal_strict_execution_accuracy": normal_strict,
        f"{prefix}_semantic_strict_execution_accuracy": semantic_strict,
        f"{prefix}_semantic_strict_delta_vs_normal": semantic_strict - normal_strict,
        f"{prefix}_normal_syntax_accuracy": normal_syntax,
        f"{prefix}_semantic_syntax_accuracy": semantic_syntax,
        f"{prefix}_semantic_syntax_delta_vs_normal": semantic_syntax - normal_syntax,
        f"{prefix}_semantic_context_helped": bool(metrics.get("semantic_context_helped")),
        f"{prefix}_context_comparable_row_count": int(
            metrics.get("semantic_context_transfer_comparable_row_count") or 0
        ),
    }


def _hosted_gap_metrics(
    *,
    normal_hosted_comparison: dict[str, Any],
    semantic_hosted_comparison: dict[str, Any],
) -> dict[str, float]:
    normal_local_delta = _metric(normal_hosted_comparison, "local_value_delta_vs_hosted")
    semantic_local_delta = _metric(
        semantic_hosted_comparison, "local_value_delta_vs_hosted"
    )
    normal_strict_delta = _metric(
        normal_hosted_comparison, "local_strict_delta_vs_hosted"
    )
    semantic_strict_delta = _metric(
        semantic_hosted_comparison, "local_strict_delta_vs_hosted"
    )
    return {
        "normal_context_local_value_delta_vs_hosted": normal_local_delta,
        "semantic_context_local_value_delta_vs_hosted": semantic_local_delta,
        "normal_context_hosted_value_gap": -normal_local_delta,
        "semantic_context_hosted_value_gap": -semantic_local_delta,
        "semantic_context_hosted_value_gap_delta_vs_normal": (
            -semantic_local_delta - (-normal_local_delta)
        ),
        "normal_context_local_strict_delta_vs_hosted": normal_strict_delta,
        "semantic_context_local_strict_delta_vs_hosted": semantic_strict_delta,
        "normal_context_hosted_strict_gap": -normal_strict_delta,
        "semantic_context_hosted_strict_gap": -semantic_strict_delta,
        "semantic_context_hosted_strict_gap_delta_vs_normal": (
            -semantic_strict_delta - (-normal_strict_delta)
        ),
    }


def summarize_semantic_context_transfer_evidence(
    *,
    local_context_comparison: dict[str, Any],
    hosted_context_comparison: dict[str, Any],
    normal_local_vs_hosted_comparison: dict[str, Any],
    semantic_local_vs_hosted_comparison: dict[str, Any],
) -> dict[str, Any]:
    """Return a compact Checkpoint 10 evidence summary."""

    _validate_context_comparison(local_context_comparison, label="local context")
    _validate_context_comparison(hosted_context_comparison, label="hosted context")
    _validate_hosted_comparison(
        normal_local_vs_hosted_comparison,
        label="normal-context",
    )
    _validate_hosted_comparison(
        semantic_local_vs_hosted_comparison,
        label="semantic-context",
    )

    local_count = int(
        (local_context_comparison.get("metrics") or {}).get(
            "semantic_context_transfer_comparable_row_count"
        )
        or 0
    )
    hosted_count = int(
        (hosted_context_comparison.get("metrics") or {}).get(
            "semantic_context_transfer_comparable_row_count"
        )
        or 0
    )
    normal_hosted_count = int(
        (normal_local_vs_hosted_comparison.get("metrics") or {}).get(
            "hosted_comparable_row_count"
        )
        or 0
    )
    semantic_hosted_count = int(
        (semantic_local_vs_hosted_comparison.get("metrics") or {}).get(
            "hosted_comparable_row_count"
        )
        or 0
    )
    if len({local_count, hosted_count, normal_hosted_count, semantic_hosted_count}) != 1:
        raise ValueError("all CP10 comparison manifests must use the same row count")

    summary = {
        "schema_version": 1,
        "artifact_type": ARTIFACT_TYPE,
        "checkpoint": 10,
        "claim_boundary": (
            "Semantic context transfer evidence summary only; promote no broader "
            "hosted or external benchmark claim without reviewing blockers."
        ),
        "oracle_policy": NON_ORACLE_GENERATION,
        "history_policy": MODEL_GENERATED_SQL_ROLLOUT,
        "comparable_row_count": local_count,
        **_context_metrics(local_context_comparison, prefix="local"),
        **_context_metrics(hosted_context_comparison, prefix="hosted"),
        **_hosted_gap_metrics(
            normal_hosted_comparison=normal_local_vs_hosted_comparison,
            semantic_hosted_comparison=semantic_local_vs_hosted_comparison,
        ),
    }
    local_delta = float(summary["local_semantic_value_delta_vs_normal"])
    hosted_delta = float(summary["hosted_semantic_value_delta_vs_normal"])
    semantic_gap_delta = float(summary["semantic_context_hosted_value_gap_delta_vs_normal"])
    summary["local_semantic_delta_exceeds_hosted_delta"] = local_delta > hosted_delta
    summary["hosted_value_gap_narrowed"] = semantic_gap_delta < 0.0
    if (
        bool(summary["local_semantic_context_helped"])
        and bool(summary["hosted_value_gap_narrowed"])
        and bool(summary["local_semantic_delta_exceeds_hosted_delta"])
    ):
        promotion_status = "gap_narrowed"
    elif bool(summary["local_semantic_context_helped"]):
        promotion_status = "local_context_helped_gap_not_narrowed"
    elif bool(summary["hosted_semantic_context_helped"]):
        promotion_status = "hosted_context_helped_more_than_local"
    else:
        promotion_status = "negative_or_inconclusive"
    summary["promotion_status"] = promotion_status
    summary["promotion_ready"] = promotion_status == "gap_narrowed"
    return summary


def summarize_semantic_context_transfer_evidence_files(
    *,
    local_context_comparison_path: Path,
    hosted_context_comparison_path: Path,
    normal_local_vs_hosted_comparison_path: Path,
    semantic_local_vs_hosted_comparison_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Load completed manifests, write a compact Checkpoint 10 summary."""

    summary = summarize_semantic_context_transfer_evidence(
        local_context_comparison=_load_json(local_context_comparison_path),
        hosted_context_comparison=_load_json(hosted_context_comparison_path),
        normal_local_vs_hosted_comparison=_load_json(
            normal_local_vs_hosted_comparison_path
        ),
        semantic_local_vs_hosted_comparison=_load_json(
            semantic_local_vs_hosted_comparison_path
        ),
    )
    summary["input_artifacts"] = {
        "local_context_comparison_path": str(local_context_comparison_path),
        "local_context_comparison_sha256": sha256_file(local_context_comparison_path),
        "hosted_context_comparison_path": str(hosted_context_comparison_path),
        "hosted_context_comparison_sha256": sha256_file(hosted_context_comparison_path),
        "normal_local_vs_hosted_comparison_path": str(
            normal_local_vs_hosted_comparison_path
        ),
        "normal_local_vs_hosted_comparison_sha256": sha256_file(
            normal_local_vs_hosted_comparison_path
        ),
        "semantic_local_vs_hosted_comparison_path": str(
            semantic_local_vs_hosted_comparison_path
        ),
        "semantic_local_vs_hosted_comparison_sha256": sha256_file(
            semantic_local_vs_hosted_comparison_path
        ),
    }
    _write_json(output_path, summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-context-comparison", type=Path, required=True)
    parser.add_argument("--hosted-context-comparison", type=Path, required=True)
    parser.add_argument("--normal-local-vs-hosted-comparison", type=Path, required=True)
    parser.add_argument("--semantic-local-vs-hosted-comparison", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_semantic_context_transfer_evidence_files(
        local_context_comparison_path=args.local_context_comparison,
        hosted_context_comparison_path=args.hosted_context_comparison,
        normal_local_vs_hosted_comparison_path=args.normal_local_vs_hosted_comparison,
        semantic_local_vs_hosted_comparison_path=args.semantic_local_vs_hosted_comparison,
        output_path=args.output,
    )
    print(f"Wrote semantic context transfer evidence summary to {args.output}")
    print(f"Promotion status: {summary['promotion_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
