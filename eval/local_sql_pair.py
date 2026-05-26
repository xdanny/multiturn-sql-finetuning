"""Shared local comparison flow for SQL-generating method-vs-control pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.direct_sql_eval import run_direct_sql_eval
from eval.local_text_benchmark import run_local_text_benchmark
from eval.pair_input_contract import PairInputSpec, validate_pair_input_paths
from eval.training_manifest_pair import (
    TrainingManifestSpec,
    validate_training_manifest_path,
)

NON_ORACLE_GENERATION = "non_oracle_generation"
DEFAULT_FORBIDDEN_PROMPT_MARKERS = ("reference_sql", "expected_rows", "gold_metric_dsl")


@dataclass(frozen=True)
class SqlPairSpec:
    method_name: str
    method_stage: str
    method_benchmark: str
    method_prompt_variant: str
    direct_benchmark: str
    comparison_label: str
    direct_stage: str = "direct_sql_control"
    evaluation_mode: str = NON_ORACLE_GENERATION
    direct_prompt_variant: str = "direct_sql_control"
    method_output_stem: str | None = None
    row_identity_fields: tuple[str, ...] = ("fixture_id", "reference_sql")
    forbidden_prompt_markers: tuple[str, ...] = DEFAULT_FORBIDDEN_PROMPT_MARKERS

    @property
    def output_stem(self) -> str:
        return self.method_output_stem or self.method_name

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _validate_non_leak_rows(
    rows: list[dict[str, Any]],
    *,
    label: str,
    forbidden_prompt_markers: tuple[str, ...],
) -> None:
    if not rows:
        raise ValueError(f"{label} training input must be non-empty")
    for row in rows:
        if row.get("oracle_policy") != "non_oracle_inputs_only":
            raise ValueError(f"{label} training row must declare non_oracle_inputs_only")
        for message in row.get("messages", []):
            content = str(message.get("content", ""))
            for marker in forbidden_prompt_markers:
                if marker in content:
                    raise ValueError(f"{label} training row leaked scorer-only content")

def validate_sql_pair_training_manifests(
    *,
    method_training_manifest: Path,
    direct_training_manifest: Path,
    spec: SqlPairSpec,
) -> dict[str, Any]:
    method_manifest, method_input = validate_training_manifest_path(
        manifest_path=method_training_manifest,
        spec=TrainingManifestSpec(
            label=spec.method_name,
            stage=spec.method_stage,
            benchmark=spec.method_benchmark,
            evaluation_mode=spec.evaluation_mode,
        ),
    )
    direct_manifest, direct_input = validate_training_manifest_path(
        manifest_path=direct_training_manifest,
        spec=TrainingManifestSpec(
            label="direct SQL",
            stage=spec.direct_stage,
            benchmark=spec.direct_benchmark,
            evaluation_mode=spec.evaluation_mode,
        ),
    )
    method_rows = _load_jsonl(method_input)
    direct_rows = _load_jsonl(direct_input)
    _validate_non_leak_rows(
        method_rows,
        label=spec.method_name,
        forbidden_prompt_markers=spec.forbidden_prompt_markers,
    )
    _validate_non_leak_rows(
        direct_rows,
        label="direct SQL",
        forbidden_prompt_markers=spec.forbidden_prompt_markers,
    )
    validated_rows = validate_pair_input_paths(
        left_input_path=method_input,
        right_input_path=direct_input,
        spec=PairInputSpec(
            label=spec.comparison_label,
            left_name=spec.method_name,
            right_name="direct SQL",
            row_identity_fields=spec.row_identity_fields,
        ),
    )
    return {
        "method_manifest": method_manifest,
        "direct_manifest": direct_manifest,
        "method_input_path": method_input,
        "direct_input_path": direct_input,
        "row_count": validated_rows["row_count"],
    }


def run_local_sql_pair_generation_and_eval(
    *,
    validated: dict[str, Any],
    spec: SqlPairSpec,
    output_dir: Path,
    run_id: str,
    model_name: str,
    method_adapter_path: Path | None,
    direct_adapter_path: Path | None,
    max_new_tokens: int,
    max_memory_gb: int | None,
    fixtures_path: Path | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    method_predictions = output_dir / f"{run_id}.{spec.output_stem}.predictions.jsonl"
    method_results = output_dir / f"{run_id}.{spec.output_stem}.jsonl"
    method_manifest_output = output_dir / f"{run_id}.{spec.output_stem}.manifest.json"
    direct_predictions = output_dir / f"{run_id}.direct_sql.predictions.jsonl"
    direct_results = output_dir / f"{run_id}.direct_sql.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.direct_sql.manifest.json"

    method_generation_code = run_local_text_benchmark(
        model_name=model_name,
        adapter_path=method_adapter_path,
        input_path=validated["method_input_path"],
        output_path=method_predictions,
        output_field="generated_sql",
        max_new_tokens=max_new_tokens,
        max_memory_gb=max_memory_gb,
    )
    if method_generation_code != 0:
        raise RuntimeError(f"{spec.method_name} generation failed with exit code {method_generation_code}")

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
        raise RuntimeError(f"direct SQL generation failed with exit code {direct_generation_code}")

    method_eval_code = run_direct_sql_eval(
        input_path=method_predictions,
        output_path=method_results,
        manifest_output=method_manifest_output,
        model_name=model_name,
        fixtures_path=fixtures_path,
        working_dir=output_dir / ".scratch" / spec.output_stem,
        benchmark=spec.method_benchmark,
        prompt_variant=spec.method_prompt_variant,
        command=[
            "python",
            f"-m eval.run_local_{spec.comparison_label}_comparison",
            "--run-id",
            run_id,
            f"# {spec.method_name}",
        ],
    )
    if method_eval_code != 0:
        raise RuntimeError(f"{spec.method_name} eval failed with exit code {method_eval_code}")

    direct_eval_code = run_direct_sql_eval(
        input_path=direct_predictions,
        output_path=direct_results,
        manifest_output=direct_manifest_output,
        model_name=model_name,
        fixtures_path=fixtures_path,
        working_dir=output_dir / ".scratch" / "direct_sql",
        benchmark=spec.direct_benchmark,
        prompt_variant=spec.direct_prompt_variant,
        command=[
            "python",
            f"-m eval.run_local_{spec.comparison_label}_comparison",
            "--run-id",
            run_id,
            "# direct_sql_control",
        ],
    )
    if direct_eval_code != 0:
        raise RuntimeError(f"direct SQL eval failed with exit code {direct_eval_code}")

    return {
        "method_predictions": method_predictions,
        "method_results": method_results,
        "method_manifest_output": method_manifest_output,
        "direct_predictions": direct_predictions,
        "direct_results": direct_results,
        "direct_manifest_output": direct_manifest_output,
        "compared_output": output_dir / f"{run_id}.compared.manifest.json",
    }
