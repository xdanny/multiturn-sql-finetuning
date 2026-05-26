"""Validate BIRD-Interact transfer manifests before local-vs-hosted comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_hosted_baseline import compare_hosted_baseline_manifest_files


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_bird_interact_result_manifest(result_manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(result_manifest_path)
    benchmark = str(manifest.get("benchmark") or "")
    if "bird_interact" not in benchmark:
        raise ValueError("result manifest must use a bird_interact benchmark")
    if manifest.get("evaluation_mode") != "non_oracle_generation":
        raise ValueError("BIRD-Interact comparison requires evaluation_mode=non_oracle_generation")
    return manifest


def run_bird_interact_comparison(
    *,
    local_result_manifest: Path,
    hosted_result_manifest: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> int:
    validate_bird_interact_result_manifest(local_result_manifest)
    validate_bird_interact_result_manifest(hosted_result_manifest)
    compare_hosted_baseline_manifest_files(
        local_manifest_path=local_result_manifest,
        hosted_manifest_path=hosted_result_manifest,
        output_path=output_path,
        repo_root=repo_root,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-result-manifest", type=Path, required=True)
    parser.add_argument("--hosted-result-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    return run_bird_interact_comparison(
        local_result_manifest=args.local_result_manifest,
        hosted_result_manifest=args.hosted_result_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
