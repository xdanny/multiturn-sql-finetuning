from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_metric_dsl_comparison import (
    run_metric_dsl_comparison,
    validate_metric_dsl_comparison_inputs,
    write_metric_dsl_comparison_preflight,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _training_manifest(*, stage: str, benchmark: str, mode: str, train_path: Path) -> dict:
    return {
        "schema_version": 1,
        "run_id": f"{stage}-run",
        "status": "validated",
        "stage": stage,
        "benchmark": benchmark,
        "training_target": stage,
        "evaluation_mode": mode,
        "train_data_path": str(train_path),
        "train_data_sha256": "train-hash",
        "output_dir": "outputs/unit",
        "final_dir": "outputs/unit/final",
        "command": ["train.finetune", "validate-data-only"],
    }


def test_validate_metric_dsl_comparison_inputs_requires_expected_stage_pair(tmp_path: Path) -> None:
    metric_manifest_path = tmp_path / "metric.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    metric_input = tmp_path / "metric.jsonl"
    direct_input = tmp_path / "direct.jsonl"
    _write_json(
        metric_manifest_path,
        _training_manifest(
            stage="metric_dsl",
            benchmark="synthetic_metric_dsl_bootstrap",
            mode="metric_dsl",
            train_path=metric_input,
        ),
    )
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="planner_first_sql",
            benchmark="metric_dsl_direct_sql",
            mode="non_oracle_generation",
            train_path=direct_input,
        ),
    )
    _write_jsonl(
        metric_input,
        [{"fixture_id": "measure_preservation_metric", "reference_sql": "SELECT 1"}],
    )
    _write_jsonl(
        direct_input,
        [{"fixture_id": "measure_preservation_metric", "reference_sql": "SELECT 1"}],
    )

    with pytest.raises(ValueError, match="direct_sql_control"):
        validate_metric_dsl_comparison_inputs(
            metric_training_manifest=metric_manifest_path,
            direct_training_manifest=direct_manifest_path,
        )


def test_write_metric_dsl_comparison_preflight_records_ready_pair(tmp_path: Path) -> None:
    metric_manifest_path = tmp_path / "metric.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    metric_input = tmp_path / "metric.jsonl"
    direct_input = tmp_path / "direct.jsonl"
    preflight_path = tmp_path / "preflight.json"
    _write_json(
        metric_manifest_path,
        _training_manifest(
            stage="metric_dsl",
            benchmark="synthetic_metric_dsl_bootstrap",
            mode="metric_dsl",
            train_path=metric_input,
        ),
    )
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="direct_sql_control",
            benchmark="metric_dsl_direct_sql",
            mode="non_oracle_generation",
            train_path=direct_input,
        ),
    )
    shared_rows = [{"fixture_id": "measure_preservation_metric", "reference_sql": "SELECT 1"}]
    _write_jsonl(metric_input, shared_rows)
    _write_jsonl(direct_input, shared_rows)

    preflight = write_metric_dsl_comparison_preflight(
        metric_training_manifest=metric_manifest_path,
        direct_training_manifest=direct_manifest_path,
        output_path=preflight_path,
    )

    written = json.loads(preflight_path.read_text())
    assert preflight["status"] == "ready_for_offline_pair"
    assert written["row_count"] == 1
    assert written["metric_stage"] == "metric_dsl"
    assert written["direct_stage"] == "direct_sql_control"


def test_run_metric_dsl_comparison_runs_both_eval_paths_then_compares(tmp_path, monkeypatch) -> None:
    metric_manifest_path = tmp_path / "metric.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    metric_input = tmp_path / "metric.jsonl"
    direct_input = tmp_path / "direct.jsonl"
    output_dir = tmp_path / "results"
    _write_json(
        metric_manifest_path,
        _training_manifest(
            stage="metric_dsl",
            benchmark="synthetic_metric_dsl_bootstrap",
            mode="metric_dsl",
            train_path=metric_input,
        ),
    )
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="direct_sql_control",
            benchmark="metric_dsl_direct_sql",
            mode="non_oracle_generation",
            train_path=direct_input,
        ),
    )
    shared_rows = [{"fixture_id": "measure_preservation_metric", "reference_sql": "SELECT 1"}]
    _write_jsonl(metric_input, shared_rows)
    _write_jsonl(direct_input, shared_rows)

    metric_calls: list[dict] = []
    direct_calls: list[dict] = []
    compare_calls: list[dict] = []

    def fake_run_metric_dsl_eval(**kwargs):
        metric_calls.append(kwargs)
        kwargs["manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["output_path"].stem,
                    "output_path": str(kwargs["output_path"]),
                    "row_count": 1,
                    "benchmark": "metric_dsl",
                    "evaluation_mode": "metric_dsl",
                    "metrics": {
                        "metric_dsl_parse_rate": 1.0,
                        "metric_dsl_compile_rate": 1.0,
                        "compiled_sql_execution_evaluated_rows": 1,
                        "measure_preservation": 1.0,
                        "value_execution_accuracy": 1.0,
                        "strict_execution_accuracy": 1.0,
                    },
                }
            )
        )
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "id": "metric-1",
                    "fixture_id": "measure_preservation_metric",
                    "evaluation_mode": "metric_dsl",
                    "reference_sql": "SELECT 1",
                    "database_path": "metric.sqlite",
                    "sql_execution_attempted": True,
                    "value_execution_score": 1.0,
                    "strict_execution_score": 1.0,
                }
            )
            + "\n"
        )
        return 0

    def fake_run_direct_sql_eval(**kwargs):
        direct_calls.append(kwargs)
        kwargs["manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["output_path"].stem,
                    "output_path": str(kwargs["output_path"]),
                    "row_count": 1,
                    "benchmark": "metric_dsl_direct_sql",
                    "evaluation_mode": "non_oracle_generation",
                    "metrics": {
                        "value_execution_accuracy": 0.0,
                        "strict_execution_accuracy": 0.0,
                    },
                }
            )
        )
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "id": "metric-1",
                    "fixture_id": "measure_preservation_metric",
                    "evaluation_mode": "non_oracle_generation",
                    "reference_sql": "SELECT 1",
                    "database_path": "metric.sqlite",
                    "value_execution_score": 0.0,
                    "strict_execution_score": 0.0,
                }
            )
            + "\n"
        )
        return 0

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {"compared": True}

    monkeypatch.setattr("eval.run_metric_dsl_comparison.run_metric_dsl_eval", fake_run_metric_dsl_eval)
    monkeypatch.setattr("eval.run_metric_dsl_comparison.run_direct_sql_eval", fake_run_direct_sql_eval)
    monkeypatch.setattr(
        "eval.run_metric_dsl_comparison.compare_metric_dsl_direct_sql_manifest_files",
        fake_compare,
    )

    exit_code = run_metric_dsl_comparison(
        metric_training_manifest=metric_manifest_path,
        direct_training_manifest=direct_manifest_path,
        output_dir=output_dir,
        run_id="metric-stage4",
        model_name="local-9b",
    )

    assert exit_code == 0
    assert metric_calls[0]["input_path"] == metric_input
    assert direct_calls[0]["input_path"] == direct_input
    assert compare_calls == [
        {
            "metric_dsl_manifest_path": output_dir / "metric-stage4.metric_dsl.manifest.json",
            "direct_sql_manifest_path": output_dir / "metric-stage4.direct_sql.manifest.json",
            "output_path": output_dir / "metric-stage4.compared.manifest.json",
            "repo_root": Path("."),
        }
    ]
