"""Run local direct-SQL and predicted-planner SQL eval as one comparable pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_predicted_planner import compare_predicted_planner_manifest_files
from eval.local_benchmark import run_local_benchmark
from eval.run_predicted_planner_comparison import validate_comparison_inputs

DIRECT_STAGE = "direct_sql_control"
PREDICTED_STAGE = "predicted_planner"
PREPARED_BENCHMARK = "prepared"
NON_ORACLE_GENERATION = "non_oracle_generation"
PREDICTED_PLANNER = "predicted_planner"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_training_manifest(
    manifest: dict[str, Any],
    *,
    expected_stage: str,
    expected_benchmark: str,
    expected_mode: str,
    label: str,
) -> Path:
    if manifest.get("stage") != expected_stage:
        raise ValueError(f"{label} training manifest must use stage={expected_stage}")
    if manifest.get("benchmark") != expected_benchmark:
        raise ValueError(f"{label} training manifest must use benchmark={expected_benchmark}")
    if manifest.get("evaluation_mode") != expected_mode:
        raise ValueError(f"{label} training manifest must use evaluation_mode={expected_mode}")
    train_data_path = manifest.get("train_data_path")
    if not train_data_path:
        raise ValueError(f"{label} training manifest is missing train_data_path")
    return Path(str(train_data_path))


def validate_local_predicted_planner_training_manifests(
    *,
    direct_training_manifest: Path,
    predicted_training_manifest: Path,
) -> dict[str, Any]:
    direct_manifest = _load_json(direct_training_manifest)
    predicted_manifest = _load_json(predicted_training_manifest)
    direct_input = _validate_training_manifest(
        direct_manifest,
        expected_stage=DIRECT_STAGE,
        expected_benchmark=PREPARED_BENCHMARK,
        expected_mode=NON_ORACLE_GENERATION,
        label="direct SQL",
    )
    predicted_input = _validate_training_manifest(
        predicted_manifest,
        expected_stage=PREDICTED_STAGE,
        expected_benchmark=PREPARED_BENCHMARK,
        expected_mode=PREDICTED_PLANNER,
        label="predicted planner",
    )
    summary = validate_comparison_inputs(
        direct_input=direct_input,
        predicted_input=predicted_input,
        limit=None,
    )
    return {
        "direct_manifest": direct_manifest,
        "predicted_manifest": predicted_manifest,
        "direct_input_path": direct_input,
        "predicted_input_path": predicted_input,
        **summary,
    }


def run_local_predicted_planner_comparison(
    *,
    direct_training_manifest: Path,
    predicted_training_manifest: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    direct_adapter_path: Path | None,
    predicted_adapter_path: Path | None,
    database_root: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    repo_root: Path = Path("."),
) -> int:
    validated = validate_local_predicted_planner_training_manifests(
        direct_training_manifest=direct_training_manifest,
        predicted_training_manifest=predicted_training_manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_output = output_dir / f"{run_id}.direct.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.direct.manifest.json"
    predicted_output = output_dir / f"{run_id}.predicted_planner.jsonl"
    predicted_manifest_output = output_dir / f"{run_id}.predicted_planner.manifest.json"
    compared_manifest = output_dir / f"{run_id}.compared.manifest.json"

    direct_code = run_local_benchmark(
        model_name=model_name,
        adapter_path=direct_adapter_path,
        benchmark="prepared",
        input_path=validated["direct_input_path"],
        output=direct_output,
        limit=None,
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
        database_root=database_root,
        allow_oracle_plan=False,
        manifest_output=direct_manifest_output,
        prompt_variant="direct_sql_control",
        command=[
            "python",
            "-m",
            "eval.run_local_predicted_planner_comparison",
            "--run-id",
            run_id,
            "# direct_sql_control",
        ],
    )
    if direct_code != 0:
        return direct_code

    predicted_code = run_local_benchmark(
        model_name=model_name,
        adapter_path=predicted_adapter_path,
        benchmark="prepared",
        input_path=validated["predicted_input_path"],
        output=predicted_output,
        limit=None,
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
        database_root=database_root,
        allow_oracle_plan=False,
        manifest_output=predicted_manifest_output,
        prompt_variant="predicted_planner",
        command=[
            "python",
            "-m",
            "eval.run_local_predicted_planner_comparison",
            "--run-id",
            run_id,
            "# predicted_planner",
        ],
    )
    if predicted_code != 0:
        return predicted_code

    compare_predicted_planner_manifest_files(
        predicted_manifest_path=predicted_manifest_output,
        direct_manifest_path=direct_manifest_output,
        output_path=compared_manifest,
        repo_root=repo_root,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-training-manifest", type=Path, required=True)
    parser.add_argument("--predicted-training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--direct-adapter-path", type=Path, default=None)
    parser.add_argument("--predicted-adapter-path", type=Path, default=None)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    return run_local_predicted_planner_comparison(
        direct_training_manifest=args.direct_training_manifest,
        predicted_training_manifest=args.predicted_training_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        direct_adapter_path=args.direct_adapter_path,
        predicted_adapter_path=args.predicted_adapter_path,
        database_root=args.database_root,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
