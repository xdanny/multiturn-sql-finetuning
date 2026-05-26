"""Run the offline Stage 4 metric-DSL comparison from paired training manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_metric_dsl_direct_sql import compare_metric_dsl_direct_sql_manifest_files
from eval.direct_sql_eval import run_direct_sql_eval
from eval.metric_dsl_eval import run_metric_dsl_eval
from eval.result_manifest import sha256_file

DIRECT_STAGE = "direct_sql_control"
DIRECT_BENCHMARK = "metric_dsl_direct_sql"
DIRECT_MODE = "non_oracle_generation"
METRIC_STAGE = "metric_dsl"
METRIC_BENCHMARK = "synthetic_metric_dsl_bootstrap"
METRIC_MODE = "metric_dsl"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_identity(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("fixture_id")), str(row.get("reference_sql")))


def _validate_manifest(
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


def validate_metric_dsl_comparison_inputs(
    *,
    metric_training_manifest: Path,
    direct_training_manifest: Path,
) -> dict[str, Any]:
    metric_manifest = _load_json(metric_training_manifest)
    direct_manifest = _load_json(direct_training_manifest)
    metric_input = _validate_manifest(
        metric_manifest,
        expected_stage=METRIC_STAGE,
        expected_benchmark=METRIC_BENCHMARK,
        expected_mode=METRIC_MODE,
        label="metric-DSL",
    )
    direct_input = _validate_manifest(
        direct_manifest,
        expected_stage=DIRECT_STAGE,
        expected_benchmark=DIRECT_BENCHMARK,
        expected_mode=DIRECT_MODE,
        label="direct SQL",
    )
    metric_rows = _load_jsonl(metric_input)
    direct_rows = _load_jsonl(direct_input)
    if not metric_rows or not direct_rows:
        raise ValueError("metric-DSL comparison requires non-empty paired inputs")
    if [_row_identity(row) for row in metric_rows] != [_row_identity(row) for row in direct_rows]:
        raise ValueError("metric-DSL and direct-SQL training inputs must share row identity")
    return {
        "metric_manifest": metric_manifest,
        "direct_manifest": direct_manifest,
        "metric_input_path": metric_input,
        "direct_input_path": direct_input,
        "row_count": len(metric_rows),
    }


def write_metric_dsl_comparison_preflight(
    *,
    metric_training_manifest: Path,
    direct_training_manifest: Path,
    output_path: Path,
) -> dict[str, Any]:
    validated = validate_metric_dsl_comparison_inputs(
        metric_training_manifest=metric_training_manifest,
        direct_training_manifest=direct_training_manifest,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "metric_dsl_comparison_preflight",
        "status": "ready_for_offline_pair",
        "claim_boundary": "preflight only; no SQL execution claim",
        "metric_training_manifest_path": str(metric_training_manifest),
        "metric_training_manifest_sha256": sha256_file(metric_training_manifest),
        "direct_training_manifest_path": str(direct_training_manifest),
        "direct_training_manifest_sha256": sha256_file(direct_training_manifest),
        "metric_input_path": str(validated["metric_input_path"]),
        "metric_input_sha256": sha256_file(validated["metric_input_path"]),
        "direct_input_path": str(validated["direct_input_path"]),
        "direct_input_sha256": sha256_file(validated["direct_input_path"]),
        "metric_stage": METRIC_STAGE,
        "direct_stage": DIRECT_STAGE,
        "row_count": validated["row_count"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def run_metric_dsl_comparison(
    *,
    metric_training_manifest: Path,
    direct_training_manifest: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    repo_root: Path = Path("."),
    fixtures_path: Path | None = None,
    preflight_output: Path | None = None,
) -> int:
    validated = validate_metric_dsl_comparison_inputs(
        metric_training_manifest=metric_training_manifest,
        direct_training_manifest=direct_training_manifest,
    )
    if preflight_output is not None:
        write_metric_dsl_comparison_preflight(
            metric_training_manifest=metric_training_manifest,
            direct_training_manifest=direct_training_manifest,
            output_path=preflight_output,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_output = output_dir / f"{run_id}.metric_dsl.jsonl"
    metric_manifest_output = output_dir / f"{run_id}.metric_dsl.manifest.json"
    direct_output = output_dir / f"{run_id}.direct_sql.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.direct_sql.manifest.json"
    compared_output = output_dir / f"{run_id}.compared.manifest.json"

    metric_code = run_metric_dsl_eval(
        input_path=validated["metric_input_path"],
        output_path=metric_output,
        manifest_output=metric_manifest_output,
        model_name=model_name,
        command=[
            "python",
            "-m",
            "eval.run_metric_dsl_comparison",
            "--run-id",
            run_id,
            "# metric_dsl",
        ],
    )
    if metric_code != 0:
        return metric_code
    direct_code = run_direct_sql_eval(
        input_path=validated["direct_input_path"],
        output_path=direct_output,
        manifest_output=direct_manifest_output,
        model_name=model_name,
        fixtures_path=fixtures_path,
        working_dir=output_dir / ".scratch",
        command=[
            "python",
            "-m",
            "eval.run_metric_dsl_comparison",
            "--run-id",
            run_id,
            "# direct_sql_control",
        ],
    )
    if direct_code != 0:
        return direct_code
    compare_metric_dsl_direct_sql_manifest_files(
        metric_dsl_manifest_path=metric_manifest_output,
        direct_sql_manifest_path=direct_manifest_output,
        output_path=compared_output,
        repo_root=repo_root,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-training-manifest", type=Path, required=True)
    parser.add_argument("--direct-training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--fixtures", type=Path, default=None)
    parser.add_argument("--preflight-output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        if args.preflight_output is None:
            raise SystemExit("--preflight-output is required with --preflight-only")
        write_metric_dsl_comparison_preflight(
            metric_training_manifest=args.metric_training_manifest,
            direct_training_manifest=args.direct_training_manifest,
            output_path=args.preflight_output,
        )
        return 0
    return run_metric_dsl_comparison(
        metric_training_manifest=args.metric_training_manifest,
        direct_training_manifest=args.direct_training_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        repo_root=args.repo_root,
        fixtures_path=args.fixtures,
        preflight_output=args.preflight_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
