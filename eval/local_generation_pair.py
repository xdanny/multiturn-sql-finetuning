"""Shared local paired-generation orchestration for mixed-output method comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalGenerationPairSpec:
    method_output_stem: str
    direct_output_stem: str = "direct_sql"


def run_local_generation_pair(
    *,
    spec: LocalGenerationPairSpec,
    output_dir: Path,
    run_id: str,
    method_generate: Callable[[Path], int],
    direct_generate: Callable[[Path], int],
    method_eval: Callable[[Path, Path, Path], int],
    direct_eval: Callable[[Path, Path, Path], int],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    method_predictions = output_dir / f"{run_id}.{spec.method_output_stem}.predictions.jsonl"
    method_results = output_dir / f"{run_id}.{spec.method_output_stem}.jsonl"
    method_manifest_output = output_dir / f"{run_id}.{spec.method_output_stem}.manifest.json"
    direct_predictions = output_dir / f"{run_id}.{spec.direct_output_stem}.predictions.jsonl"
    direct_results = output_dir / f"{run_id}.{spec.direct_output_stem}.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.{spec.direct_output_stem}.manifest.json"
    compared_output = output_dir / f"{run_id}.compared.manifest.json"

    method_generation_code = method_generate(method_predictions)
    if method_generation_code != 0:
        raise RuntimeError(f"method generation failed with exit code {method_generation_code}")
    direct_generation_code = direct_generate(direct_predictions)
    if direct_generation_code != 0:
        raise RuntimeError(f"direct generation failed with exit code {direct_generation_code}")

    method_eval_code = method_eval(method_predictions, method_results, method_manifest_output)
    if method_eval_code != 0:
        raise RuntimeError(f"method eval failed with exit code {method_eval_code}")
    direct_eval_code = direct_eval(direct_predictions, direct_results, direct_manifest_output)
    if direct_eval_code != 0:
        raise RuntimeError(f"direct eval failed with exit code {direct_eval_code}")

    return {
        "method_predictions": method_predictions,
        "method_results": method_results,
        "method_manifest_output": method_manifest_output,
        "direct_predictions": direct_predictions,
        "direct_results": direct_results,
        "direct_manifest_output": direct_manifest_output,
        "compared_output": compared_output,
    }
