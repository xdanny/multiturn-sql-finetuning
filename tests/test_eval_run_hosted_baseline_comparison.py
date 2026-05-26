from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_hosted_baseline_comparison import (
    run_hosted_baseline_comparison,
    validate_local_candidate_manifest,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _training_manifest(*, stage: str, benchmark: str, mode: str, eval_path: Path) -> dict:
    return {
        "schema_version": 1,
        "run_id": f"{stage}-run",
        "status": "validated",
        "stage": stage,
        "benchmark": benchmark,
        "training_target": stage,
        "evaluation_mode": mode,
        "train_data_path": str(eval_path),
        "train_data_sha256": "train-hash",
        "eval_data_path": str(eval_path),
        "eval_data_sha256": "eval-hash",
        "output_dir": "outputs/unit",
        "final_dir": "outputs/unit/final",
        "command": ["train.finetune", "validate-data-only"],
    }


def test_validate_local_candidate_manifest_requires_prepared_non_oracle_eval(tmp_path) -> None:
    manifest_path = tmp_path / "training.manifest.json"
    eval_path = tmp_path / "eval.jsonl"
    _write_json(
        manifest_path,
        _training_manifest(
            stage="metric_dsl",
            benchmark="metric_dsl",
            mode="metric_dsl",
            eval_path=eval_path,
        ),
    )

    with pytest.raises(ValueError, match="benchmark=prepared"):
        validate_local_candidate_manifest(manifest_path)


def test_run_hosted_baseline_comparison_validates_and_compares(tmp_path, monkeypatch) -> None:
    local_training_manifest = tmp_path / "local.training.manifest.json"
    local_result_manifest = tmp_path / "local.result.manifest.json"
    hosted_result_manifest = tmp_path / "hosted.result.manifest.json"
    output_path = tmp_path / "local.compared.manifest.json"
    eval_path = tmp_path / "eval.jsonl"

    _write_json(
        local_training_manifest,
        _training_manifest(
            stage="behavior_recovery",
            benchmark="prepared",
            mode="non_oracle_generation",
            eval_path=eval_path,
        ),
    )
    _write_json(
        local_result_manifest,
        {
            "run_id": "local-result",
            "benchmark": "prepared",
            "evaluation_mode": "non_oracle_generation",
            "input_path": str(eval_path),
        },
    )
    _write_json(
        hosted_result_manifest,
        {
            "run_id": "hosted-result",
            "benchmark": "prepared",
            "evaluation_mode": "non_oracle_generation",
            "input_path": str(eval_path),
        },
    )

    compare_calls: list[dict] = []

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {"compared": True}

    monkeypatch.setattr(
        "eval.run_hosted_baseline_comparison.compare_hosted_baseline_manifest_files",
        fake_compare,
    )

    exit_code = run_hosted_baseline_comparison(
        local_training_manifest=local_training_manifest,
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
