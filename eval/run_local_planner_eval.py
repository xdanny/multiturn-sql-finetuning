"""Run local planner prediction and planner-quality scoring from a training manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.local_planner_benchmark import run_local_planner_benchmark
from eval.planner_eval import (
    JSON_PLANNER_PREDICTIONS_SOURCE,
    annotate_prepared_records_with_plans,
    run_planner_eval,
)

PLANNER_STAGE = "planner_supervision"
PREPARED_BENCHMARK = "prepared"
NON_ORACLE_GENERATION = "non_oracle_generation"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_local_planner_training_manifest(*, training_manifest: Path) -> dict[str, Any]:
    manifest = _load_json(training_manifest)
    if manifest.get("stage") != PLANNER_STAGE:
        raise ValueError(f"planner training manifest must use stage={PLANNER_STAGE}")
    if manifest.get("benchmark") != PREPARED_BENCHMARK:
        raise ValueError(f"planner training manifest must use benchmark={PREPARED_BENCHMARK}")
    if manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError(
            f"planner training manifest must use evaluation_mode={NON_ORACLE_GENERATION}"
        )
    train_data_path = manifest.get("train_data_path")
    if not train_data_path:
        raise ValueError("planner training manifest is missing train_data_path")
    manifest["train_data_path"] = str(train_data_path)
    return manifest


def run_local_planner_eval(
    *,
    training_manifest: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    adapter_path: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    predicted_prepared_output: Path | None = None,
) -> int:
    manifest = validate_local_planner_training_manifest(training_manifest=training_manifest)
    input_path = Path(str(manifest["train_data_path"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_output = output_dir / f"{run_id}.predictions.jsonl"
    planner_output = output_dir / f"{run_id}.planner_eval.jsonl"
    summary_output = output_dir / f"{run_id}.planner_eval_summary.json"

    predict_code = run_local_planner_benchmark(
        model_name=model_name,
        adapter_path=adapter_path,
        input_path=input_path,
        output_path=predictions_output,
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
    )
    if predict_code != 0:
        return predict_code

    eval_code = run_planner_eval(
        input_path=input_path,
        output=planner_output,
        summary_output=summary_output,
        limit=None,
        allow_oracle_plan=False,
        planner_source=JSON_PLANNER_PREDICTIONS_SOURCE,
        planner_predictions_path=predictions_output,
    )
    if eval_code != 0:
        return eval_code
    if predicted_prepared_output is not None:
        annotate_prepared_records_with_plans(
            input_path,
            predicted_prepared_output,
            limit=None,
            planner_source=JSON_PLANNER_PREDICTIONS_SOURCE,
            planner_predictions_path=predictions_output,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    parser.add_argument("--predicted-prepared-output", type=Path, default=None)
    args = parser.parse_args()
    return run_local_planner_eval(
        training_manifest=args.training_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
        predicted_prepared_output=args.predicted_prepared_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
