"""Run the offline Stage 4 metric-DSL comparison from paired training manifests."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

from data.synthetic_method_fixtures import DEFAULT_OUTPUT as DEFAULT_FIXTURES_PATH
from data.synthetic_method_fixtures import build_synthetic_method_fixtures
from eval.compare_metric_dsl_direct_sql import compare_metric_dsl_direct_sql_manifest_files
from eval.direct_sql_eval import run_direct_sql_eval
from eval.metric_dsl_eval import run_metric_dsl_eval
from eval.pair_input_contract import PairInputSpec, validate_pair_input_paths
from eval.result_manifest import sha256_file
from eval.training_manifest_pair import (
    TrainingManifestSpec,
    validate_training_manifest_path,
)

DIRECT_STAGE = "direct_sql_control"
DIRECT_BENCHMARK = "metric_dsl_direct_sql"
DIRECT_MODE = "non_oracle_generation"
METRIC_STAGE = "metric_dsl"
METRIC_BENCHMARK = "synthetic_metric_dsl_bootstrap"
METRIC_MODE = "metric_dsl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _last_assistant_content(row: dict[str, Any]) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return str(message["content"])
    return None


def _row_id(row: dict[str, Any]) -> str:
    if row.get("id"):
        return str(row["id"])
    fixture_id = row.get("fixture_id")
    if fixture_id:
        return str(fixture_id)
    reference_sql = str(row.get("reference_sql") or "")
    return reference_sql or "row"


def _database_identity(row: dict[str, Any]) -> str:
    return str(
        row.get("database_id")
        or row.get("fixture_id")
        or row.get("schema_id")
        or "synthetic_metric_dsl"
    )


def _fixture_map(fixtures_path: Path | None) -> dict[str, dict[str, Any]]:
    if fixtures_path is not None and fixtures_path.exists():
        fixtures = _load_jsonl(fixtures_path)
    else:
        fixtures = build_synthetic_method_fixtures()
    return {str(fixture["fixture_id"]): fixture for fixture in fixtures}


def _database_path_for_fixture(
    fixture_id: str | None,
    *,
    fixtures: dict[str, dict[str, Any]],
    working_dir: Path,
) -> str | None:
    if not fixture_id:
        return None
    fixture = fixtures.get(str(fixture_id))
    if fixture is None:
        raise ValueError(f"unknown synthetic fixture: {fixture_id}")
    database_path = working_dir / f"{fixture_id}.sqlite"
    if not database_path.exists():
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database_path) as conn:
            conn.executescript(str(fixture["schema_sql"]))
            conn.executescript(str(fixture["seed_data_sql"]))
    return str(database_path)


def _materialize_metric_dsl_bootstrap_rows(
    rows: list[dict[str, Any]],
    *,
    fixtures: dict[str, dict[str, Any]],
    working_dir: Path,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        predicted_dsl = (
            row.get("predicted_dsl")
            or row.get("generated_metric_dsl")
            or row.get("raw_generation")
            or row.get("gold_dsl")
            or _last_assistant_content(row)
        )
        if not predicted_dsl:
            raise ValueError("metric-DSL bootstrap rows require predicted_dsl or gold_dsl")
        prepared.append(
            {
                **row,
                "id": _row_id(row),
                "database_id": _database_identity(row),
                "predicted_dsl": str(predicted_dsl),
                "database_path": row.get("database_path")
                or _database_path_for_fixture(
                    row.get("fixture_id"),
                    fixtures=fixtures,
                    working_dir=working_dir,
                ),
                "bootstrap_prediction_source": (
                    "predicted_field"
                    if row.get("predicted_dsl") or row.get("generated_metric_dsl") or row.get("raw_generation")
                    else "gold_dsl_label"
                ),
            }
        )
    return prepared


def _materialize_direct_sql_bootstrap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        generated_sql = (
            row.get("generated_sql")
            or row.get("predicted_sql")
            or row.get("raw_generation")
            or row.get("reference_sql")
            or _last_assistant_content(row)
        )
        if not generated_sql:
            raise ValueError("direct SQL bootstrap rows require generated_sql or reference_sql")
        prepared.append(
            {
                **row,
                "id": _row_id(row),
                "database_id": _database_identity(row),
                "generated_sql": str(generated_sql),
                "bootstrap_prediction_source": (
                    "predicted_field"
                    if row.get("generated_sql") or row.get("predicted_sql") or row.get("raw_generation")
                    else "reference_sql_label"
                ),
            }
        )
    return prepared

def validate_metric_dsl_comparison_inputs(
    *,
    metric_training_manifest: Path,
    direct_training_manifest: Path,
) -> dict[str, Any]:
    metric_manifest, metric_input = validate_training_manifest_path(
        manifest_path=metric_training_manifest,
        spec=TrainingManifestSpec(
            label="metric-DSL",
            stage=METRIC_STAGE,
            benchmark=METRIC_BENCHMARK,
            evaluation_mode=METRIC_MODE,
        ),
    )
    direct_manifest, direct_input = validate_training_manifest_path(
        manifest_path=direct_training_manifest,
        spec=TrainingManifestSpec(
            label="direct SQL",
            stage=DIRECT_STAGE,
            benchmark=DIRECT_BENCHMARK,
            evaluation_mode=DIRECT_MODE,
        ),
    )
    validated_rows = validate_pair_input_paths(
        left_input_path=metric_input,
        right_input_path=direct_input,
        spec=PairInputSpec(
            label="metric-DSL",
            left_name="metric-DSL",
            right_name="direct SQL",
            row_identity_fields=("fixture_id", "reference_sql"),
        ),
    )
    return {
        "metric_manifest": metric_manifest,
        "direct_manifest": direct_manifest,
        "metric_input_path": metric_input,
        "direct_input_path": direct_input,
        "row_count": validated_rows["row_count"],
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
    prepared_dir = output_dir / ".prepared"
    fixture_working_dir = output_dir / ".fixtures"
    fixture_map = _fixture_map(fixtures_path or DEFAULT_FIXTURES_PATH)
    metric_eval_input = prepared_dir / f"{run_id}.metric_dsl.input.jsonl"
    direct_eval_input = prepared_dir / f"{run_id}.direct_sql.input.jsonl"
    _write_jsonl(
        _materialize_metric_dsl_bootstrap_rows(
            _load_jsonl(validated["metric_input_path"]),
            fixtures=fixture_map,
            working_dir=fixture_working_dir,
        ),
        metric_eval_input,
    )
    _write_jsonl(
        _materialize_direct_sql_bootstrap_rows(_load_jsonl(validated["direct_input_path"])),
        direct_eval_input,
    )
    metric_output = output_dir / f"{run_id}.metric_dsl.jsonl"
    metric_manifest_output = output_dir / f"{run_id}.metric_dsl.manifest.json"
    direct_output = output_dir / f"{run_id}.direct_sql.jsonl"
    direct_manifest_output = output_dir / f"{run_id}.direct_sql.manifest.json"
    compared_output = output_dir / f"{run_id}.compared.manifest.json"

    metric_code = run_metric_dsl_eval(
        input_path=metric_eval_input,
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
        input_path=direct_eval_input,
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
