"""Run local semantic-proxy and direct-SQL control eval as one comparable pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_semantic_proxy_direct_sql import (
    compare_semantic_proxy_direct_sql_manifest_files,
)
from eval.local_benchmark import run_local_benchmark

SEMANTIC_STAGE = "semantic_layer"
DIRECT_STAGE = "direct_sql_control"
SEMANTIC_BENCHMARK = "cosql_semantic_proxy"
DIRECT_BENCHMARK = "cosql_semantic_proxy_direct_sql"


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
    eval_data_path = manifest.get("eval_data_path") or manifest.get("train_data_path")
    if not eval_data_path:
        raise ValueError(f"{label} training manifest is missing eval_data_path or train_data_path")
    return Path(str(eval_data_path))


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row.get("dialog_id")), str(row.get("database_id")), str(row.get("reference_sql")))


def validate_local_semantic_proxy_training_manifests(
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
    if [_row_identity(row) for row in semantic_rows] != [_row_identity(row) for row in direct_rows]:
        raise ValueError("semantic proxy and direct SQL eval inputs must share the same row identity")
    return {
        "semantic_manifest": semantic_manifest,
        "direct_manifest": direct_manifest,
        "semantic_input_path": semantic_input,
        "direct_input_path": direct_input,
    }


def run_local_semantic_proxy_comparison(
    *,
    semantic_training_manifest: Path,
    direct_training_manifest: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    semantic_adapter_path: Path | None,
    direct_adapter_path: Path | None,
    database_root: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    repo_root: Path = Path("."),
) -> int:
    validated = validate_local_semantic_proxy_training_manifests(
        semantic_training_manifest=semantic_training_manifest,
        direct_training_manifest=direct_training_manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    semantic_output = output_dir / f"{run_id}.semantic_proxy.jsonl"
    semantic_manifest_output = output_dir / f"{run_id}.semantic_proxy.manifest.json"
    direct_output = output_dir / f"{run_id}.direct.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.direct.manifest.json"
    compared_output = output_dir / f"{run_id}.compared.manifest.json"

    semantic_code = run_local_benchmark(
        model_name=model_name,
        adapter_path=semantic_adapter_path,
        benchmark="prepared",
        input_path=validated["semantic_input_path"],
        output=semantic_output,
        limit=None,
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
        database_root=database_root,
        allow_oracle_plan=False,
        manifest_output=semantic_manifest_output,
        prompt_variant="semantic_proxy",
        command=["python", "-m", "eval.run_local_semantic_proxy_comparison"],
    )
    if semantic_code != 0:
        return semantic_code
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
        command=["python", "-m", "eval.run_local_semantic_proxy_comparison"],
    )
    if direct_code != 0:
        return direct_code
    compare_semantic_proxy_direct_sql_manifest_files(
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
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()
    return run_local_semantic_proxy_comparison(
        semantic_training_manifest=args.semantic_training_manifest,
        direct_training_manifest=args.direct_training_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        semantic_adapter_path=args.semantic_adapter_path,
        direct_adapter_path=args.direct_adapter_path,
        database_root=args.database_root,
        max_new_tokens=args.max_new_tokens,
        max_memory_gb=args.max_memory_gb,
        repo_root=args.repo_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
