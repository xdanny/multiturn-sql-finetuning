"""Run generated-history rollout locally against prepared dialog records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from eval.local_benchmark import generate_local_sql, load_model_and_tokenizer
from eval.result_manifest import build_result_manifest, write_result_manifest
from eval.rollout_eval import (
    MODEL_GENERATED_SQL_ROLLOUT,
    evaluate_rollout_records,
    load_rollout_prepared_records,
)
from eval.run_eval import summarize_eval_metrics, write_results


def run_local_rollout_benchmark(
    *,
    model_name: str,
    adapter_path: Path | None,
    input_path: Path,
    output_path: Path,
    manifest_output: Path | None,
    database_root: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    allow_oracle_plan: bool,
    prompt_variant: str | None = None,
    command: list[str] | None = None,
) -> int:
    records = load_rollout_prepared_records(
        input_path,
        limit_dialogs=None,
        allow_oracle_plan=allow_oracle_plan,
    )
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        adapter_path=adapter_path,
        max_memory_gb=max_memory_gb,
    )

    def local_generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        return generate_local_sql(
            model,
            tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )

    results = evaluate_rollout_records(
        records,
        generate_fn=local_generate,
        model_name=model_name if adapter_path is None else f"{model_name}+{adapter_path}",
        database_root=database_root,
    )
    written = write_results(results, output_path)
    metrics = summarize_eval_metrics(results)
    metrics["history_policy"] = MODEL_GENERATED_SQL_ROLLOUT
    metrics["rollout_dialog_count"] = len(records)
    if manifest_output is None:
        manifest_output = output_path.with_suffix(".manifest.json")
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
        endpoint="local",
        evaluation_mode=evaluation_mode,
        oracle_allowed=allow_oracle_plan,
        prompt_variant=prompt_variant,
        database_root=database_root,
        command=list(command or ["python", "-m", "eval.local_rollout_benchmark"]),
        row_count=written,
        metrics=metrics,
    )
    write_result_manifest(manifest, manifest_output)
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--prompt-variant", default=None)
    parser.add_argument(
        "--allow-oracle-plan",
        action="store_true",
        help="Allow prepared inputs containing gold SQL-derived planning hints.",
    )
    args = parser.parse_args()

    return run_local_rollout_benchmark(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        input_path=args.input,
        output_path=args.output,
        manifest_output=args.manifest_output,
        database_root=args.database_root,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
        allow_oracle_plan=args.allow_oracle_plan,
        prompt_variant=args.prompt_variant,
        command=sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
