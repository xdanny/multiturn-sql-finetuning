from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_bird_interact_comparison import (
    run_bird_interact_comparison,
    validate_bird_interact_result_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _result_manifest(*, run_id: str, endpoint: str) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "benchmark": "bird_interact_transfer",
        "model_name": run_id,
        "endpoint": endpoint,
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/bird_interact_rows.jsonl",
        "input_sha256": "input-hash",
        "output_path": f"results/{run_id}.jsonl",
        "output_sha256": f"{run_id}-hash",
        "row_count": 1,
        "metrics": {
            "value_execution_accuracy": 0.8,
            "strict_execution_accuracy": 0.7,
            "mean_latency_ms": 500.0,
            "total_cost_usd": 0.42,
        },
    }


def test_validate_bird_interact_result_manifest_requires_bird_benchmark(tmp_path) -> None:
    manifest_path = tmp_path / "result.manifest.json"
    payload = _result_manifest(run_id="local", endpoint="http://127.0.0.1:8000/v1")
    payload["benchmark"] = "prepared"
    _write_json(manifest_path, payload)

    with pytest.raises(ValueError, match="bird_interact"):
        validate_bird_interact_result_manifest(manifest_path)


def test_run_bird_interact_comparison_validates_and_compares(tmp_path, monkeypatch) -> None:
    local_result_manifest = tmp_path / "local.result.manifest.json"
    hosted_result_manifest = tmp_path / "hosted.result.manifest.json"
    output_path = tmp_path / "local.vs_hosted.manifest.json"

    _write_json(
        local_result_manifest,
        _result_manifest(run_id="local", endpoint="http://127.0.0.1:8000/v1"),
    )
    _write_json(
        hosted_result_manifest,
        _result_manifest(run_id="hosted", endpoint="https://api.example.test/v1"),
    )

    compare_calls: list[dict] = []

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {"compared": True}

    monkeypatch.setattr(
        "eval.run_bird_interact_comparison.compare_hosted_baseline_manifest_files",
        fake_compare,
    )

    exit_code = run_bird_interact_comparison(
        local_result_manifest=local_result_manifest,
        hosted_result_manifest=hosted_result_manifest,
        output_path=output_path,
    )

    assert exit_code == 0
    assert compare_calls == [
        {
            "local_manifest_path": local_result_manifest,
            "hosted_manifest_path": hosted_result_manifest,
            "output_path": output_path,
            "repo_root": Path("."),
        }
    ]
