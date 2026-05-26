"""Run local teacher-forced and generated-history prepared eval for one checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_rollout_history import compare_rollout_manifest_files
from eval.local_benchmark import run_local_benchmark
from eval.local_rollout_benchmark import run_local_rollout_benchmark
from eval.result_manifest import sha256_file


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


def write_rollout_comparison_preflight(
    *,
    training_manifest: Path,
    output_path: Path,
) -> dict[str, object]:
    validated = validate_local_rollout_training_manifest(training_manifest)
    input_path = Path(str(validated["input_path"]))
    manifest = validated["manifest"]
    payload = {
        "schema_version": 1,
        "artifact_type": "rollout_comparison_preflight",
        "status": "ready_for_local_rollout_pair",
        "claim_boundary": "preflight only; no rollout execution claim",
        "training_manifest_path": str(training_manifest),
        "training_manifest_sha256": sha256_file(training_manifest),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "stage": manifest["stage"],
        "benchmark": manifest["benchmark"],
        "evaluation_mode": manifest["evaluation_mode"],
        "row_count": sum(1 for _ in input_path.open()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


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
    preflight_output: Path | None = None,
) -> int:
    validated = validate_local_rollout_training_manifest(training_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    if preflight_output is not None:
        write_rollout_comparison_preflight(
            training_manifest=training_manifest,
            output_path=preflight_output,
        )

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
    parser.add_argument("--preflight-output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        if args.preflight_output is None:
            raise SystemExit("--preflight-output is required with --preflight-only")
        write_rollout_comparison_preflight(
            training_manifest=args.training_manifest,
            output_path=args.preflight_output,
        )
        return 0

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
        preflight_output=args.preflight_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
