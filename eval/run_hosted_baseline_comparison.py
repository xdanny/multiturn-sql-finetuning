"""Validate a local prepared candidate before comparing it with a hosted baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_hosted_baseline import compare_hosted_baseline_manifest_files
from eval.stage6_protocol import validate_result_manifest_against_contract


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_local_candidate_manifest(training_manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(training_manifest_path)
    if manifest.get("benchmark") != "prepared":
        raise ValueError("local candidate training manifest must use benchmark=prepared")
    if manifest.get("evaluation_mode") != "non_oracle_generation":
        raise ValueError(
            "local candidate training manifest must use evaluation_mode=non_oracle_generation"
        )
    return manifest


def run_hosted_baseline_comparison(
    *,
    local_training_manifest: Path,
    local_result_manifest: Path,
    hosted_result_manifest: Path,
    contract_manifest: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> int:
    validate_local_candidate_manifest(local_training_manifest)
    validate_result_manifest_against_contract(
        result_manifest_path=local_result_manifest,
        contract_manifest_path=contract_manifest,
        expected_benchmark="prepared",
        label="local hosted-baseline",
    )
    validate_result_manifest_against_contract(
        result_manifest_path=hosted_result_manifest,
        contract_manifest_path=contract_manifest,
        expected_benchmark="prepared",
        label="hosted baseline",
    )
    compare_hosted_baseline_manifest_files(
        local_manifest_path=local_result_manifest,
        hosted_manifest_path=hosted_result_manifest,
        output_path=output_path,
        repo_root=repo_root,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-training-manifest", type=Path, required=True)
    parser.add_argument("--local-result-manifest", type=Path, required=True)
    parser.add_argument("--hosted-result-manifest", type=Path, required=True)
    parser.add_argument(
        "--contract-manifest",
        type=Path,
        default=Path("docs/data_artifacts/hosted_baseline.manifest.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    return run_hosted_baseline_comparison(
        local_training_manifest=args.local_training_manifest,
        local_result_manifest=args.local_result_manifest,
        hosted_result_manifest=args.hosted_result_manifest,
        contract_manifest=args.contract_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
