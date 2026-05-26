"""Run local Stage 4 generation, scoring, and comparison for paired adapters."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.compare_metric_dsl_direct_sql import compare_metric_dsl_direct_sql_manifest_files
from eval.direct_sql_eval import run_direct_sql_eval
from eval.local_metric_dsl_benchmark import run_local_metric_dsl_benchmark
from eval.metric_dsl_eval import run_metric_dsl_eval
from eval.run_metric_dsl_comparison import validate_metric_dsl_comparison_inputs


def run_local_metric_dsl_comparison(
    *,
    metric_training_manifest: Path,
    direct_training_manifest: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    metric_adapter_path: Path | None,
    direct_adapter_path: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    repo_root: Path = Path("."),
    fixtures_path: Path | None = None,
) -> int:
    validated = validate_metric_dsl_comparison_inputs(
        metric_training_manifest=metric_training_manifest,
        direct_training_manifest=direct_training_manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_predictions = output_dir / f"{run_id}.metric_dsl.predictions.jsonl"
    metric_results = output_dir / f"{run_id}.metric_dsl.jsonl"
    metric_manifest_output = output_dir / f"{run_id}.metric_dsl.manifest.json"
    direct_predictions = output_dir / f"{run_id}.direct_sql.predictions.jsonl"
    direct_results = output_dir / f"{run_id}.direct_sql.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.direct_sql.manifest.json"
    compared_output = output_dir / f"{run_id}.compared.manifest.json"

    metric_generation_code = run_local_metric_dsl_benchmark(
        model_name=model_name,
        adapter_path=metric_adapter_path,
        input_path=validated["metric_input_path"],
        output_path=metric_predictions,
        output_field="generated_metric_dsl",
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
    )
    if metric_generation_code != 0:
        return metric_generation_code
    direct_generation_code = run_local_metric_dsl_benchmark(
        model_name=model_name,
        adapter_path=direct_adapter_path,
        input_path=validated["direct_input_path"],
        output_path=direct_predictions,
        output_field="generated_sql",
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
    )
    if direct_generation_code != 0:
        return direct_generation_code

    metric_eval_code = run_metric_dsl_eval(
        input_path=metric_predictions,
        output_path=metric_results,
        manifest_output=metric_manifest_output,
        model_name=model_name,
        command=[
            "python",
            "-m",
            "eval.run_local_metric_dsl_comparison",
            "--run-id",
            run_id,
            "# metric_dsl",
        ],
    )
    if metric_eval_code != 0:
        return metric_eval_code
    direct_eval_code = run_direct_sql_eval(
        input_path=direct_predictions,
        output_path=direct_results,
        manifest_output=direct_manifest_output,
        model_name=model_name,
        fixtures_path=fixtures_path,
        working_dir=output_dir / ".scratch",
        command=[
            "python",
            "-m",
            "eval.run_local_metric_dsl_comparison",
            "--run-id",
            run_id,
            "# direct_sql_control",
        ],
    )
    if direct_eval_code != 0:
        return direct_eval_code
    compare_metric_dsl_direct_sql_manifest_files(
        metric_dsl_manifest_path=metric_manifest_output,
        direct_sql_manifest_path=direct_manifest_output,
        output_path=compared_output,
        repo_root=repo_root,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-training-manifest", type=Path, required=True)
    parser.add_argument("--direct-training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--metric-adapter-path", type=Path, default=None)
    parser.add_argument("--direct-adapter-path", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--fixtures", type=Path, default=None)
    args = parser.parse_args()
    return run_local_metric_dsl_comparison(
        metric_training_manifest=args.metric_training_manifest,
        direct_training_manifest=args.direct_training_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        metric_adapter_path=args.metric_adapter_path,
        direct_adapter_path=args.direct_adapter_path,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
        repo_root=args.repo_root,
        fixtures_path=args.fixtures,
    )


if __name__ == "__main__":
    raise SystemExit(main())
