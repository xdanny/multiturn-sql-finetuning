"""Run generated-history recovery rollout and teacher-forced comparison."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from openai import OpenAI

from eval.behavior_recovery_teacher_forced import run_behavior_recovery_teacher_forced
from eval.compare_rollout_history import compare_rollout_manifest_files
from eval.local_generation import local_adapter_generate_fn
from eval.result_manifest import build_result_manifest, write_result_manifest
from eval.rollout_eval import (
    evaluate_rollout_records,
    load_rollout_prepared_records,
)
from eval.run_eval import generate_sql, summarize_eval_metrics, write_results

GenerateFn = Callable[[list[dict[str, str]]], tuple[str, float]]


def _endpoint_generate_fn(
    *,
    endpoint: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> GenerateFn:
    client = OpenAI(base_url=endpoint, api_key=api_key)

    def generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        return generate_sql(
            client,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return generate


def _write_rollout_eval(
    *,
    input_path: Path,
    output_path: Path,
    manifest_output: Path,
    model_name: str,
    endpoint: str,
    database_root: Path | None,
    generate_fn: GenerateFn,
    command: Sequence[str],
) -> None:
    records = load_rollout_prepared_records(input_path)
    results = evaluate_rollout_records(
        records,
        generate_fn=generate_fn,
        model_name=model_name,
        database_root=database_root,
    )
    written = write_results(results, output_path)
    metrics = summarize_eval_metrics(results)
    metrics["history_policy"] = "model_generated_sql_rollout"
    metrics["rollout_dialog_count"] = len(records)
    evaluation_modes = metrics.get("evaluation_modes", {})
    evaluation_mode = (
        next(iter(evaluation_modes))
        if len(evaluation_modes) == 1
        else ",".join(sorted(evaluation_modes)) or "unknown"
    )
    manifest = build_result_manifest(
        run_id=output_path.stem,
        benchmark="prepared_rollout",
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        endpoint=endpoint,
        evaluation_mode=evaluation_mode,
        oracle_allowed=False,
        prompt_variant=None,
        database_root=database_root,
        command=list(command),
        row_count=written,
        metrics=metrics,
    )
    write_result_manifest(manifest, manifest_output)


def run_behavior_recovery_comparison(
    *,
    input_path: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    generate_fn: GenerateFn,
    database_root: Path | None = None,
    endpoint: str = "offline",
    command: Sequence[str] | None = None,
) -> dict:
    """Write rollout, teacher-forced, and comparison manifests for one recovery run."""

    command = list(command or sys.argv)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_output = output_dir / f"{run_id}.rollout.jsonl"
    rollout_manifest = output_dir / f"{run_id}.rollout.manifest.json"
    teacher_output = output_dir / f"{run_id}.teacher_forced.jsonl"
    teacher_manifest = output_dir / f"{run_id}.teacher_forced.manifest.json"
    comparison_manifest = output_dir / f"{run_id}.comparison.manifest.json"

    _write_rollout_eval(
        input_path=input_path,
        output_path=rollout_output,
        manifest_output=rollout_manifest,
        model_name=model_name,
        endpoint=endpoint,
        database_root=database_root,
        generate_fn=generate_fn,
        command=command,
    )
    run_behavior_recovery_teacher_forced(
        input_path=input_path,
        output_path=teacher_output,
        manifest_output=teacher_manifest,
        model_name=model_name,
        command=command,
    )
    return compare_rollout_manifest_files(
        rollout_manifest_path=rollout_manifest,
        teacher_forced_manifest_path=teacher_manifest,
        output_path=comparison_manifest,
        repo_root=Path("."),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--result-model-name", default=None)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--backend", choices=["endpoint", "local"], default="endpoint")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    args = parser.parse_args()
    if args.backend == "local":
        generate_fn = local_adapter_generate_fn(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            max_tokens=args.max_tokens,
            max_memory_gb=args.max_memory_gb,
        )
        endpoint = "local"
    else:
        generate_fn = _endpoint_generate_fn(
            endpoint=args.endpoint,
            api_key=args.api_key,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        endpoint = args.endpoint

    compared = run_behavior_recovery_comparison(
        input_path=args.input,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.result_model_name or args.model_name,
        database_root=args.database_root,
        endpoint=endpoint,
        generate_fn=generate_fn,
        command=sys.argv,
    )
    metrics = compared["metrics"]
    print(f"Wrote behavior recovery comparison under {args.output_dir}")
    print(
        "Value delta vs teacher-forced: "
        f"{metrics['rollout_value_delta_vs_teacher_forced']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
