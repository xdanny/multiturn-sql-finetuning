"""Run local Stage 5 generation, scoring, and comparison for paired adapters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_behavior_recovery_direct_sql import (
    compare_behavior_recovery_direct_sql_manifest_files,
)
from eval.direct_sql_eval import run_direct_sql_eval
from eval.local_text_benchmark import run_local_text_benchmark


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_training_manifest(
    manifest: dict[str, Any],
    *,
    expected_stage: str,
    expected_benchmark: str,
    label: str,
) -> Path:
    if manifest.get("stage") != expected_stage:
        raise ValueError(f"{label} training manifest must use stage={expected_stage}")
    if manifest.get("benchmark") != expected_benchmark:
        raise ValueError(f"{label} training manifest must use benchmark={expected_benchmark}")
    if manifest.get("evaluation_mode") != "non_oracle_generation":
        raise ValueError(f"{label} training manifest must use evaluation_mode=non_oracle_generation")
    train_data_path = manifest.get("train_data_path")
    if not train_data_path:
        raise ValueError(f"{label} training manifest is missing train_data_path")
    return Path(str(train_data_path))


def run_local_behavior_recovery_comparison(
    *,
    behavior_recovery_training_manifest: Path,
    direct_training_manifest: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    behavior_recovery_adapter_path: Path | None,
    direct_adapter_path: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    repo_root: Path = Path("."),
    fixtures_path: Path | None = None,
) -> int:
    recovery_input_path = _validate_training_manifest(
        _load_json(behavior_recovery_training_manifest),
        expected_stage="behavior_recovery",
        expected_benchmark="synthetic_behavior_recovery",
        label="behavior-recovery",
    )
    direct_input_path = _validate_training_manifest(
        _load_json(direct_training_manifest),
        expected_stage="direct_sql_control",
        expected_benchmark="behavior_recovery_direct_sql",
        label="direct SQL",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    recovery_predictions = output_dir / f"{run_id}.behavior_recovery.predictions.jsonl"
    recovery_results = output_dir / f"{run_id}.behavior_recovery.jsonl"
    recovery_manifest_output = output_dir / f"{run_id}.behavior_recovery.manifest.json"
    direct_predictions = output_dir / f"{run_id}.direct_sql.predictions.jsonl"
    direct_results = output_dir / f"{run_id}.direct_sql.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.direct_sql.manifest.json"
    compared_output = output_dir / f"{run_id}.compared.manifest.json"

    recovery_generation_code = run_local_text_benchmark(
        model_name=model_name,
        adapter_path=behavior_recovery_adapter_path,
        input_path=recovery_input_path,
        output_path=recovery_predictions,
        output_field="generated_sql",
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
    )
    if recovery_generation_code != 0:
        return recovery_generation_code
    direct_generation_code = run_local_text_benchmark(
        model_name=model_name,
        adapter_path=direct_adapter_path,
        input_path=direct_input_path,
        output_path=direct_predictions,
        output_field="generated_sql",
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
    )
    if direct_generation_code != 0:
        return direct_generation_code

    recovery_eval_code = run_direct_sql_eval(
        input_path=recovery_predictions,
        output_path=recovery_results,
        manifest_output=recovery_manifest_output,
        model_name=model_name,
        fixtures_path=fixtures_path,
        working_dir=output_dir / ".scratch",
        benchmark="behavior_recovery",
        prompt_variant="behavior_recovery",
        command=[
            "python",
            "-m",
            "eval.run_local_behavior_recovery_comparison",
            "--run-id",
            run_id,
            "# behavior_recovery",
        ],
    )
    if recovery_eval_code != 0:
        return recovery_eval_code
    direct_eval_code = run_direct_sql_eval(
        input_path=direct_predictions,
        output_path=direct_results,
        manifest_output=direct_manifest_output,
        model_name=model_name,
        fixtures_path=fixtures_path,
        working_dir=output_dir / ".scratch",
        benchmark="behavior_recovery_direct_sql",
        prompt_variant="direct_sql_control",
        command=[
            "python",
            "-m",
            "eval.run_local_behavior_recovery_comparison",
            "--run-id",
            run_id,
            "# direct_sql_control",
        ],
    )
    if direct_eval_code != 0:
        return direct_eval_code
    compare_behavior_recovery_direct_sql_manifest_files(
        behavior_recovery_manifest_path=recovery_manifest_output,
        direct_sql_manifest_path=direct_manifest_output,
        output_path=compared_output,
        repo_root=repo_root,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-recovery-training-manifest", type=Path, required=True)
    parser.add_argument("--direct-training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--behavior-recovery-adapter-path", type=Path, default=None)
    parser.add_argument("--direct-adapter-path", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--fixtures", type=Path, default=None)
    args = parser.parse_args()
    return run_local_behavior_recovery_comparison(
        behavior_recovery_training_manifest=args.behavior_recovery_training_manifest,
        direct_training_manifest=args.direct_training_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        behavior_recovery_adapter_path=args.behavior_recovery_adapter_path,
        direct_adapter_path=args.direct_adapter_path,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
        repo_root=args.repo_root,
        fixtures_path=args.fixtures,
    )


if __name__ == "__main__":
    raise SystemExit(main())
