"""Finalize Checkpoint 10 semantic-context transfer evidence from completed runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.compare_hosted_baseline import compare_hosted_baseline_manifest_files
from eval.semantic_context_transfer import compare_semantic_context_transfer_manifest_files
from eval.summarize_semantic_context_transfer import (
    summarize_semantic_context_transfer_evidence_files,
)

DEFAULT_OUTPUT_DIR = Path("results/semantic_context_transfer_cp10")
DEFAULT_RUN_ID = "cp10_semantic_context_transfer"


def finalize_semantic_context_transfer(
    *,
    local_normal_manifest: Path,
    local_semantic_manifest: Path,
    hosted_normal_manifest: Path,
    hosted_semantic_manifest: Path,
    preflight_manifest: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    run_id: str = DEFAULT_RUN_ID,
    repo_root: Path = Path("."),
    local_comparison_role: str = "best_local",
    hosted_comparison_role: str = "hosted_sonnet",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write all final CP10 comparison artifacts from four rollout manifests."""

    output_dir.mkdir(parents=True, exist_ok=True)
    command = list(command or sys.argv)
    local_context_comparison_path = (
        output_dir / f"{run_id}.local.semantic_context_comparison.manifest.json"
    )
    hosted_context_comparison_path = (
        output_dir / f"{run_id}.hosted.semantic_context_comparison.manifest.json"
    )
    normal_local_vs_hosted_path = (
        output_dir / f"{run_id}.normal.local_vs_hosted.manifest.json"
    )
    semantic_local_vs_hosted_path = (
        output_dir / f"{run_id}.semantic.local_vs_hosted.manifest.json"
    )
    summary_path = output_dir / f"{run_id}.evidence_summary.json"

    local_context_comparison = compare_semantic_context_transfer_manifest_files(
        normal_manifest_path=local_normal_manifest,
        semantic_manifest_path=local_semantic_manifest,
        output_path=local_context_comparison_path,
        comparison_role=local_comparison_role,
        preflight_manifest_path=preflight_manifest,
        repo_root=repo_root,
    )
    hosted_context_comparison = compare_semantic_context_transfer_manifest_files(
        normal_manifest_path=hosted_normal_manifest,
        semantic_manifest_path=hosted_semantic_manifest,
        output_path=hosted_context_comparison_path,
        comparison_role=hosted_comparison_role,
        preflight_manifest_path=preflight_manifest,
        repo_root=repo_root,
    )
    normal_local_vs_hosted = compare_hosted_baseline_manifest_files(
        local_manifest_path=local_normal_manifest,
        hosted_manifest_path=hosted_normal_manifest,
        output_path=normal_local_vs_hosted_path,
        repo_root=repo_root,
    )
    semantic_local_vs_hosted = compare_hosted_baseline_manifest_files(
        local_manifest_path=local_semantic_manifest,
        hosted_manifest_path=hosted_semantic_manifest,
        output_path=semantic_local_vs_hosted_path,
        repo_root=repo_root,
    )
    summary = summarize_semantic_context_transfer_evidence_files(
        local_context_comparison_path=local_context_comparison_path,
        hosted_context_comparison_path=hosted_context_comparison_path,
        normal_local_vs_hosted_comparison_path=normal_local_vs_hosted_path,
        semantic_local_vs_hosted_comparison_path=semantic_local_vs_hosted_path,
        output_path=summary_path,
    )
    summary["finalizer"] = {
        "module": "eval.finalize_semantic_context_transfer",
        "command": command,
        "local_normal_manifest": str(local_normal_manifest),
        "local_semantic_manifest": str(local_semantic_manifest),
        "hosted_normal_manifest": str(hosted_normal_manifest),
        "hosted_semantic_manifest": str(hosted_semantic_manifest),
        "preflight_manifest": str(preflight_manifest),
        "local_context_comparison_path": str(local_context_comparison_path),
        "hosted_context_comparison_path": str(hosted_context_comparison_path),
        "normal_local_vs_hosted_comparison_path": str(normal_local_vs_hosted_path),
        "semantic_local_vs_hosted_comparison_path": str(semantic_local_vs_hosted_path),
        "summary_path": str(summary_path),
        "local_context_blockers": (
            local_context_comparison.get("metrics") or {}
        ).get("semantic_context_transfer_blockers"),
        "hosted_context_blockers": (
            hosted_context_comparison.get("metrics") or {}
        ).get("semantic_context_transfer_blockers"),
        "normal_local_value_delta_vs_hosted": (
            normal_local_vs_hosted.get("metrics") or {}
        ).get("local_value_delta_vs_hosted"),
        "semantic_local_value_delta_vs_hosted": (
            semantic_local_vs_hosted.get("metrics") or {}
        ).get("local_value_delta_vs_hosted"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-normal-manifest", type=Path, required=True)
    parser.add_argument("--local-semantic-manifest", type=Path, required=True)
    parser.add_argument("--hosted-normal-manifest", type=Path, required=True)
    parser.add_argument("--hosted-semantic-manifest", type=Path, required=True)
    parser.add_argument("--preflight-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--local-comparison-role", default="best_local")
    parser.add_argument("--hosted-comparison-role", default="hosted_sonnet")
    args = parser.parse_args()

    summary = finalize_semantic_context_transfer(
        local_normal_manifest=args.local_normal_manifest,
        local_semantic_manifest=args.local_semantic_manifest,
        hosted_normal_manifest=args.hosted_normal_manifest,
        hosted_semantic_manifest=args.hosted_semantic_manifest,
        preflight_manifest=args.preflight_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        repo_root=args.repo_root,
        local_comparison_role=args.local_comparison_role,
        hosted_comparison_role=args.hosted_comparison_role,
        command=sys.argv,
    )
    print(f"Wrote semantic context transfer evidence summary to {args.output_dir}")
    print(f"Promotion status: {summary['promotion_status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
