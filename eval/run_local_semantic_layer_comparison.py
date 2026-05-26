"""Run local semantic-layer and direct-SQL control eval as one comparable pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_semantic_layer_direct_sql import (
    compare_semantic_layer_direct_sql_manifest_files,
)
from eval.direct_sql_eval import run_direct_sql_eval
from eval.local_text_benchmark import run_local_text_benchmark

SEMANTIC_STAGE = "semantic_layer"
DIRECT_STAGE = "direct_sql_control"
SEMANTIC_BENCHMARK = "synthetic_semantic_layer"
DIRECT_BENCHMARK = "synthetic_semantic_layer_direct_sql"
NON_ORACLE_GENERATION = "non_oracle_generation"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


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
    if manifest.get("evaluation_mode") != NON_ORACLE_GENERATION:
        raise ValueError(f"{label} training manifest must use evaluation_mode=non_oracle_generation")
    train_data_path = manifest.get("train_data_path")
    if not train_data_path:
        raise ValueError(f"{label} training manifest is missing train_data_path")
    return Path(str(train_data_path))


def _row_identity(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("fixture_id")), str(row.get("reference_sql")))


def _validate_non_leak_rows(rows: list[dict[str, Any]], *, label: str) -> None:
    if not rows:
        raise ValueError(f"{label} training input must be non-empty")
    for row in rows:
        if row.get("oracle_policy") != "non_oracle_inputs_only":
            raise ValueError(f"{label} training row must declare non_oracle_inputs_only")
        for message in row.get("messages", []):
            content = str(message.get("content", ""))
            if "reference_sql" in content or "expected_rows" in content or "gold_metric_dsl" in content:
                raise ValueError(f"{label} training row leaked scorer-only content")


def validate_local_semantic_layer_training_manifests(
    *,
    semantic_training_manifest: Path,
    direct_training_manifest: Path,
) -> dict[str, Any]:
    semantic_manifest = _load_json(semantic_training_manifest)
    direct_manifest = _load_json(direct_training_manifest)
    semantic_input = _validate_training_manifest(
        semantic_manifest,
        expected_stage=SEMANTIC_STAGE,
        expected_benchmark=SEMANTIC_BENCHMARK,
        label="semantic-layer",
    )
    direct_input = _validate_training_manifest(
        direct_manifest,
        expected_stage=DIRECT_STAGE,
        expected_benchmark=DIRECT_BENCHMARK,
        label="direct SQL",
    )
    semantic_rows = _load_jsonl(semantic_input)
    direct_rows = _load_jsonl(direct_input)
    _validate_non_leak_rows(semantic_rows, label="semantic-layer")
    _validate_non_leak_rows(direct_rows, label="direct SQL")
    if [_row_identity(row) for row in semantic_rows] != [_row_identity(row) for row in direct_rows]:
        raise ValueError("semantic-layer and direct SQL training inputs must share the same row identity")
    return {
        "semantic_manifest": semantic_manifest,
        "direct_manifest": direct_manifest,
        "semantic_input_path": semantic_input,
        "direct_input_path": direct_input,
        "row_count": len(semantic_rows),
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
    validated = validate_local_semantic_layer_training_manifests(
        semantic_training_manifest=semantic_training_manifest,
        direct_training_manifest=direct_training_manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    semantic_predictions = output_dir / f"{run_id}.semantic_layer.predictions.jsonl"
    semantic_results = output_dir / f"{run_id}.semantic_layer.jsonl"
    semantic_manifest_output = output_dir / f"{run_id}.semantic_layer.manifest.json"
    direct_predictions = output_dir / f"{run_id}.direct_sql.predictions.jsonl"
    direct_results = output_dir / f"{run_id}.direct_sql.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.direct_sql.manifest.json"
    compared_output = output_dir / f"{run_id}.compared.manifest.json"

    semantic_generation_code = run_local_text_benchmark(
        model_name=model_name,
        adapter_path=semantic_adapter_path,
        input_path=validated["semantic_input_path"],
        output_path=semantic_predictions,
        output_field="generated_sql",
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
    )
    if semantic_generation_code != 0:
        return semantic_generation_code
    direct_generation_code = run_local_text_benchmark(
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

    semantic_eval_code = run_direct_sql_eval(
        input_path=semantic_predictions,
        output_path=semantic_results,
        manifest_output=semantic_manifest_output,
        model_name=model_name,
        fixtures_path=fixtures_path,
        working_dir=output_dir / ".scratch" / "semantic_layer",
        benchmark=SEMANTIC_BENCHMARK,
        prompt_variant="semantic_layer",
        command=[
            "python",
            "-m",
            "eval.run_local_semantic_layer_comparison",
            "--run-id",
            run_id,
            "# semantic_layer",
        ],
    )
    if semantic_eval_code != 0:
        return semantic_eval_code
    direct_eval_code = run_direct_sql_eval(
        input_path=direct_predictions,
        output_path=direct_results,
        manifest_output=direct_manifest_output,
        model_name=model_name,
        fixtures_path=fixtures_path,
        working_dir=output_dir / ".scratch" / "direct_sql",
        benchmark=DIRECT_BENCHMARK,
        prompt_variant="direct_sql_control",
        command=[
            "python",
            "-m",
            "eval.run_local_semantic_layer_comparison",
            "--run-id",
            run_id,
            "# direct_sql_control",
        ],
    )
    if direct_eval_code != 0:
        return direct_eval_code
    compare_semantic_layer_direct_sql_manifest_files(
        semantic_manifest_path=semantic_manifest_output,
        direct_manifest_path=direct_manifest_output,
        output_path=compared_output,
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
