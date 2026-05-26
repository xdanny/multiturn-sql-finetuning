from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.compare_hosted_baseline import (
    compare_hosted_baseline_manifest_files,
    compare_hosted_baseline_manifests,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _base_rows(*, mode: str = "non_oracle_generation") -> list[dict]:
    return [
        {
            "id": "turn-1",
            "dialog_id": "dialog-1",
            "turn_index": 0,
            "database_id": "db",
            "reference_sql": "SELECT 1",
            "evaluation_mode": mode,
            "value_execution_score": 1.0,
            "strict_execution_score": 1.0,
        }
    ]


def _manifest(
    *,
    run_id: str,
    endpoint: str,
    model_name: str,
    input_sha256: str = "input-hash",
    output_sha256: str = "output-hash",
    value_accuracy: float = 0.8,
    strict_accuracy: float = 0.7,
    extra_metrics: dict | None = None,
) -> dict:
    metrics = {
        "value_execution_accuracy": value_accuracy,
        "strict_execution_accuracy": strict_accuracy,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "benchmark": "prepared",
        "model_name": model_name,
        "endpoint": endpoint,
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "prompt_variant": None,
        "input_path": "data/eval.jsonl",
        "input_sha256": input_sha256,
        "output_path": f"results/{run_id}.jsonl",
        "output_sha256": output_sha256,
        "row_count": 1,
        "metrics": metrics,
        "command": ["run"],
    }


def test_compare_hosted_baseline_records_local_delta_and_cost() -> None:
    local = _manifest(
        run_id="local_9b",
        endpoint="http://127.0.0.1:8000/v1",
        model_name="local-9b",
        value_accuracy=0.8,
        strict_accuracy=0.7,
    )
    hosted = _manifest(
        run_id="hosted_frontier",
        endpoint="https://api.example.test/v1",
        model_name="frontier",
        output_sha256="hosted-output-hash",
        value_accuracy=0.6,
        strict_accuracy=0.5,
        extra_metrics={"mean_latency_ms": 500.0, "total_cost_usd": 0.42},
    )

    compared = compare_hosted_baseline_manifests(
        local_manifest=local,
        hosted_manifest=hosted,
        local_rows=_base_rows(),
        hosted_rows=_base_rows(),
    )

    metrics = compared["metrics"]
    assert metrics["hosted_comparison_run_id"] == "hosted_frontier"
    assert metrics["hosted_model_name"] == "frontier"
    assert metrics["hosted_output_sha256"] == "hosted-output-hash"
    assert metrics["hosted_value_execution_accuracy"] == 0.6
    assert metrics["hosted_strict_execution_accuracy"] == 0.5
    assert metrics["hosted_mean_latency_ms"] == 500.0
    assert metrics["hosted_total_cost_usd"] == 0.42
    assert metrics["local_value_delta_vs_hosted"] == pytest.approx(0.2)
    assert metrics["local_strict_delta_vs_hosted"] == pytest.approx(0.2)
    assert metrics["hosted_comparable_row_count"] == 1
    assert "# compared-with-hosted" in compared["command"]
    assert "hosted_frontier" in compared["command"]


def test_compare_hosted_baseline_rejects_row_identity_mismatch() -> None:
    local_rows = _base_rows()
    hosted_rows = _base_rows()
    hosted_rows[0]["turn_index"] = 1

    with pytest.raises(ValueError, match="row identity mismatch"):
        compare_hosted_baseline_manifests(
            local_manifest=_manifest(
                run_id="local",
                endpoint="http://127.0.0.1:8000/v1",
                model_name="local-9b",
            ),
            hosted_manifest=_manifest(
                run_id="hosted",
                endpoint="https://api.example.test/v1",
                model_name="frontier",
                output_sha256="hosted-output-hash",
                extra_metrics={"mean_latency_ms": 500.0, "total_cost_usd": 0.42},
            ),
            local_rows=local_rows,
            hosted_rows=hosted_rows,
        )


def test_compare_hosted_baseline_requires_hosted_cost_and_latency() -> None:
    with pytest.raises(ValueError, match="cost and latency"):
        compare_hosted_baseline_manifests(
            local_manifest=_manifest(
                run_id="local",
                endpoint="http://127.0.0.1:8000/v1",
                model_name="local-9b",
            ),
            hosted_manifest=_manifest(
                run_id="hosted",
                endpoint="https://api.example.test/v1",
                model_name="frontier",
                output_sha256="hosted-output-hash",
            ),
            local_rows=_base_rows(),
            hosted_rows=_base_rows(),
        )


def test_compare_hosted_baseline_manifest_files_writes_augmented_manifest(tmp_path) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    local_output = tmp_path / "results" / "local.jsonl"
    hosted_output = tmp_path / "results" / "hosted.jsonl"
    _write_jsonl(input_path, [{"id": "input"}])
    _write_jsonl(local_output, _base_rows())
    _write_jsonl(hosted_output, _base_rows())
    local_manifest = _manifest(
        run_id="local",
        endpoint="http://127.0.0.1:8000/v1",
        model_name="local-9b",
        input_sha256="input-hash",
        output_sha256="local-output-hash",
        value_accuracy=0.8,
        strict_accuracy=0.7,
    )
    hosted_manifest = _manifest(
        run_id="hosted",
        endpoint="https://api.example.test/v1",
        model_name="frontier",
        input_sha256="input-hash",
        output_sha256="hosted-output-hash",
        value_accuracy=0.6,
        strict_accuracy=0.5,
        extra_metrics={"mean_generation_latency_ms": 500.0, "estimated_cost_usd": 0.42},
    )
    local_manifest["output_path"] = str(local_output.relative_to(tmp_path))
    hosted_manifest["output_path"] = str(hosted_output.relative_to(tmp_path))
    local_manifest_path = tmp_path / "local.manifest.json"
    hosted_manifest_path = tmp_path / "hosted.manifest.json"
    output_path = tmp_path / "local.compared.manifest.json"
    local_manifest_path.write_text(json.dumps(local_manifest))
    hosted_manifest_path.write_text(json.dumps(hosted_manifest))

    compared = compare_hosted_baseline_manifest_files(
        local_manifest_path=local_manifest_path,
        hosted_manifest_path=hosted_manifest_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert json.loads(output_path.read_text()) == compared
    assert compared["metrics"]["hosted_mean_latency_ms"] == 500.0
    assert compared["metrics"]["hosted_total_cost_usd"] == 0.42
