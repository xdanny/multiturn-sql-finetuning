from __future__ import annotations

import json

from eval.result_manifest import build_result_manifest, sha256_file, write_result_manifest


def test_build_result_manifest_records_hashes_metrics_and_mode(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    input_path.write_text('{"id": 1}\n{"id": 2}\n')
    output_path.write_text('{"execution_score": 1.0}\n{"execution_score": 0.0}\n')

    manifest = build_result_manifest(
        run_id="unit-run",
        benchmark="prepared",
        input_path=input_path,
        output_path=output_path,
        model_name="unit-model",
        endpoint="http://localhost:8000/v1",
        evaluation_mode="non_oracle_generation",
        oracle_allowed=False,
        prompt_variant=None,
        database_root=None,
        command=["python", "-m", "eval.run_eval"],
        row_count=2,
        metrics={"execution_accuracy": 0.5},
        git_commit="abc123",
    )

    assert manifest["schema_version"] == 1
    assert manifest["run_id"] == "unit-run"
    assert manifest["benchmark"] == "prepared"
    assert manifest["model_name"] == "unit-model"
    assert manifest["evaluation_mode"] == "non_oracle_generation"
    assert manifest["oracle_allowed"] is False
    assert manifest["input_sha256"] == sha256_file(input_path)
    assert manifest["output_sha256"] == sha256_file(output_path)
    assert manifest["row_count"] == 2
    assert manifest["metrics"]["execution_accuracy"] == 0.5
    assert manifest["git_commit"] == "abc123"


def test_write_result_manifest_writes_stable_json(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = {"schema_version": 1, "run_id": "unit-run"}

    write_result_manifest(manifest, manifest_path)

    assert json.loads(manifest_path.read_text()) == manifest
