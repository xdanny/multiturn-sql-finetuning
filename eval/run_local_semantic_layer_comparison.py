"""Run local semantic-layer and direct-SQL control eval as one comparable pair."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.compare_semantic_layer_direct_sql import (
    compare_semantic_layer_direct_sql_manifest_files,
)
from eval.local_sql_pair import (
    SqlPairSpec,
    run_local_sql_pair_generation_and_eval,
    validate_sql_pair_training_manifests,
)

SPEC = SqlPairSpec(
    method_name="semantic_layer",
    method_stage="semantic_layer",
    method_benchmark="synthetic_semantic_layer",
    method_prompt_variant="semantic_layer",
    direct_benchmark="synthetic_semantic_layer_direct_sql",
    comparison_label="semantic_layer",
)

def validate_local_semantic_layer_training_manifests(
    *,
    semantic_training_manifest: Path,
    direct_training_manifest: Path,
) -> dict[str, object]:
    validated = validate_sql_pair_training_manifests(
        method_training_manifest=semantic_training_manifest,
        direct_training_manifest=direct_training_manifest,
        spec=SPEC,
    )
    return {
        "semantic_manifest": validated["method_manifest"],
        "direct_manifest": validated["direct_manifest"],
        "semantic_input_path": validated["method_input_path"],
        "direct_input_path": validated["direct_input_path"],
        "row_count": validated["row_count"],
    }


def run_local_semantic_layer_comparison(
    *,
    semantic_training_manifest: Path,
    direct_training_manifest: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    semantic_adapter_path: Path | None,
    direct_adapter_path: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    repo_root: Path = Path("."),
    fixtures_path: Path | None = None,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    validated = validate_sql_pair_training_manifests(
        method_training_manifest=semantic_training_manifest,
        direct_training_manifest=direct_training_manifest,
        spec=SPEC,
    )
    outputs = run_local_sql_pair_generation_and_eval(
        validated=validated,
        spec=SPEC,
        output_dir=output_dir,
        run_id=run_id,
        model_name=model_name,
        method_adapter_path=semantic_adapter_path,
        direct_adapter_path=direct_adapter_path,
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
        fixtures_path=fixtures_path,
    )
    compare_semantic_layer_direct_sql_manifest_files(
        semantic_manifest_path=outputs["method_manifest_output"],
        direct_manifest_path=outputs["direct_manifest_output"],
        output_path=outputs["compared_output"],
        repo_root=repo_root,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic-training-manifest", type=Path, required=True)
    parser.add_argument("--direct-training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--semantic-adapter-path", type=Path, default=None)
    parser.add_argument("--direct-adapter-path", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--fixtures", type=Path, default=None)
    args = parser.parse_args()
    return run_local_semantic_layer_comparison(
        semantic_training_manifest=args.semantic_training_manifest,
        direct_training_manifest=args.direct_training_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        semantic_adapter_path=args.semantic_adapter_path,
        direct_adapter_path=args.direct_adapter_path,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
        repo_root=args.repo_root,
        fixtures_path=args.fixtures,
    )


if __name__ == "__main__":
    raise SystemExit(main())
