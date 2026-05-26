from __future__ import annotations

import json
from pathlib import Path

from eval.run_local_metric_dsl_comparison import run_local_metric_dsl_comparison


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


def test_run_local_metric_dsl_comparison_generates_scores_and_compares(tmp_path, monkeypatch) -> None:
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
    _write_jsonl(metric_input, [{"fixture_id": "measure_preservation_metric", "reference_sql": "SELECT 1"}])
    _write_jsonl(direct_input, [{"fixture_id": "measure_preservation_metric", "reference_sql": "SELECT 1"}])

    helper_calls: list[dict] = []
    compare_calls: list[dict] = []

    def fake_run_local_generation_pair(**kwargs):
        helper_calls.append(kwargs)
        return {
            "method_manifest_output": output_dir / "metric-local.metric_dsl.manifest.json",
            "direct_manifest_output": output_dir / "metric-local.direct_sql.manifest.json",
            "compared_output": output_dir / "metric-local.compared.manifest.json",
        }

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {"compared": True}

    monkeypatch.setattr(
        "eval.run_local_metric_dsl_comparison.run_local_generation_pair",
        fake_run_local_generation_pair,
    )
    monkeypatch.setattr(
        "eval.run_local_metric_dsl_comparison.compare_metric_dsl_direct_sql_manifest_files",
        fake_compare,
    )

    exit_code = run_local_metric_dsl_comparison(
        metric_training_manifest=metric_manifest_path,
        direct_training_manifest=direct_manifest_path,
        output_dir=output_dir,
        run_id="metric-local",
        model_name="local-9b",
        metric_adapter_path=tmp_path / "metric_adapter",
        direct_adapter_path=tmp_path / "direct_adapter",
        max_new_tokens=64,
        max_memory_gb=24,
    )

    assert exit_code == 0
    assert helper_calls[0]["spec"].method_output_stem == "metric_dsl"
    assert compare_calls == [
        {
            "metric_dsl_manifest_path": output_dir / "metric-local.metric_dsl.manifest.json",
            "direct_sql_manifest_path": output_dir / "metric-local.direct_sql.manifest.json",
            "output_path": output_dir / "metric-local.compared.manifest.json",
            "repo_root": Path("."),
        }
    ]
