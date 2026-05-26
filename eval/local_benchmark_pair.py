"""Shared local paired-benchmark orchestration for SQL-vs-SQL method comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from eval.local_benchmark import run_local_benchmark


@dataclass(frozen=True)
class LocalBenchmarkPairSpec:
    method_output_stem: str
    method_prompt_variant: str
    benchmark: str = "prepared"
    direct_output_stem: str = "direct"
    direct_prompt_variant: str = "direct_sql_control"
    allow_oracle_plan: bool = False


def run_local_benchmark_pair(
    *,
    spec: LocalBenchmarkPairSpec,
    output_dir: Path,
    run_id: str,
    model_name: str,
    method_input_path: Path,
    direct_input_path: Path,
    method_adapter_path: Path | None,
    direct_adapter_path: Path | None,
    database_root: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    method_command: list[str],
    direct_command: list[str],
    benchmark_runner: Callable[..., int] = run_local_benchmark,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    method_output = output_dir / f"{run_id}.{spec.method_output_stem}.jsonl"
    method_manifest_output = output_dir / f"{run_id}.{spec.method_output_stem}.manifest.json"
    direct_output = output_dir / f"{run_id}.{spec.direct_output_stem}.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.{spec.direct_output_stem}.manifest.json"
    compared_output = output_dir / f"{run_id}.compared.manifest.json"

    method_code = benchmark_runner(
        model_name=model_name,
        adapter_path=method_adapter_path,
        benchmark=spec.benchmark,
        input_path=method_input_path,
        output=method_output,
        limit=None,
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
        database_root=database_root,
        allow_oracle_plan=spec.allow_oracle_plan,
        manifest_output=method_manifest_output,
        prompt_variant=spec.method_prompt_variant,
        command=method_command,
    )
    if method_code != 0:
        raise RuntimeError(f"method benchmark failed with exit code {method_code}")

    direct_code = benchmark_runner(
        model_name=model_name,
        adapter_path=direct_adapter_path,
        benchmark=spec.benchmark,
        input_path=direct_input_path,
        output=direct_output,
        limit=None,
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
        database_root=database_root,
        allow_oracle_plan=spec.allow_oracle_plan,
        manifest_output=direct_manifest_output,
        prompt_variant=spec.direct_prompt_variant,
        command=direct_command,
    )
    if direct_code != 0:
        raise RuntimeError(f"direct benchmark failed with exit code {direct_code}")

    return {
        "method_output": method_output,
        "method_manifest_output": method_manifest_output,
        "direct_output": direct_output,
        "direct_manifest_output": direct_manifest_output,
        "compared_output": compared_output,
    }
