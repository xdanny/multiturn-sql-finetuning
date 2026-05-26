"""Run local teacher-forced and generated-history prepared eval for one checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_rollout_history import compare_rollout_manifest_files
from eval.local_benchmark import run_local_benchmark
from eval.local_rollout_benchmark import run_local_rollout_benchmark


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_local_rollout_training_manifest(training_manifest: Path) -> dict[str, Any]:
    manifest = _load_json(training_manifest)
    if manifest.get("benchmark") != "prepared":
        raise ValueError("training manifest must use benchmark=prepared")
    if manifest.get("evaluation_mode") != "non_oracle_generation":
        raise ValueError(
            "training manifest must use evaluation_mode=non_oracle_generation"
        )
    input_path_value = manifest.get("eval_data_path") or manifest.get("train_data_path")
    if not input_path_value:
        raise ValueError("training manifest must include eval_data_path or train_data_path")
    return {
        "manifest": manifest,
        "input_path": Path(str(input_path_value)),
    }


def run_local_rollout_comparison(
    *,
    training_manifest: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    adapter_path: Path | None,
    database_root: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    repo_root: Path = Path("."),
) -> int:
    validated = validate_local_rollout_training_manifest(training_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    teacher_output = output_dir / f"{run_id}.teacher_forced.jsonl"
    teacher_manifest_output = output_dir / f"{run_id}.teacher_forced.manifest.json"
    rollout_output = output_dir / f"{run_id}.rollout.jsonl"
    rollout_manifest_output = output_dir / f"{run_id}.rollout.manifest.json"
    compared_output = output_dir / f"{run_id}.compared.manifest.json"

    teacher_code = run_local_benchmark(
        model_name=model_name,
        adapter_path=adapter_path,
        benchmark="prepared",
        input_path=validated["input_path"],
        output=teacher_output,
        limit=None,
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
        database_root=database_root,
        allow_oracle_plan=False,
        manifest_output=teacher_manifest_output,
        prompt_variant="teacher_forced_history",
        command=[
            "python",
            "-m",
            "eval.run_local_rollout_comparison",
            "--run-id",
            run_id,
            "# teacher_forced",
        ],
    )
    if teacher_code != 0:
        return teacher_code

    rollout_code = run_local_rollout_benchmark(
        model_name=model_name,
        adapter_path=adapter_path,
        input_path=validated["input_path"],
        output_path=rollout_output,
        manifest_output=rollout_manifest_output,
        database_root=database_root,
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
        allow_oracle_plan=False,
        prompt_variant="behavior_recovery_rollout",
        command=[
            "python",
            "-m",
            "eval.run_local_rollout_comparison",
            "--run-id",
            run_id,
            "# rollout",
        ],
    )
    if rollout_code != 0:
        return rollout_code

    compare_rollout_manifest_files(
        rollout_manifest_path=rollout_manifest_output,
        teacher_forced_manifest_path=teacher_manifest_output,
        output_path=compared_output,
        repo_root=repo_root,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    return run_local_rollout_comparison(
        training_manifest=args.training_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        database_root=args.database_root,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
