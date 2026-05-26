from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_local_planner_eval import (
    run_local_planner_eval,
    validate_local_planner_training_manifest,
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


def test_validate_local_planner_training_manifest_requires_planner_supervision_stage(tmp_path) -> None:
    manifest_path = tmp_path / "planner.manifest.json"
    input_path = tmp_path / "planner.jsonl"
    _write_json(
        manifest_path,
        _training_manifest(
            stage="planner_first_sql",
            benchmark="prepared",
            mode="non_oracle_generation",
            train_path=input_path,
        ),
    )
    _write_jsonl(input_path, [])

    with pytest.raises(ValueError, match="planner_supervision"):
        validate_local_planner_training_manifest(training_manifest=manifest_path)


def test_run_local_planner_eval_generates_predictions_and_scores(tmp_path, monkeypatch) -> None:
    manifest_path = tmp_path / "planner.manifest.json"
    input_path = tmp_path / "planner.jsonl"
    output_dir = tmp_path / "results"
    _write_json(
        manifest_path,
        _training_manifest(
            stage="planner_supervision",
            benchmark="prepared",
            mode="non_oracle_generation",
            train_path=input_path,
        ),
    )
    _write_jsonl(
        input_path,
        [
            {
                "dialog_id": "dialog-a",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {
                        "role": "user",
                        "content": (
                            "Schema/context:\n"
                            "customers(id int, name text)\n\n"
                            "Question:\nShow customer names."
                        ),
                    },
                    {"role": "assistant", "content": "SELECT name FROM customers;"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{"relevant_tables": ["customers"]}],
            }
        ],
    )

    benchmark_calls: list[dict] = []
    planner_eval_calls: list[dict] = []

    def fake_run_local_planner_benchmark(**kwargs):
        benchmark_calls.append(kwargs)
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "id": "dialog-a:0",
                    "dialog_id": "dialog-a",
                    "turn_index": 0,
                    "database_id": "store",
                    "model_name": kwargs["model_name"],
                    "predicted_plan": {"relevant_tables": ["customers"], "projection_shape": {"selected_count": 1}},
                    "planner_latency_ms": 1.0,
                }
            )
            + "\n"
        )
        return 0

    def fake_run_planner_eval(**kwargs):
        planner_eval_calls.append(kwargs)
        kwargs["output"].write_text(json.dumps({"planner_scores": {"macro_planner_score": 1.0}}) + "\n")
        kwargs["summary_output"].write_text(json.dumps({"rows": 1, "macro_planner_score": 1.0}) + "\n")
        if kwargs.get("predicted_prepared_output") is not None:
            kwargs["predicted_prepared_output"].write_text(json.dumps({"evaluation_mode": "predicted_planner"}) + "\n")
        return 0

    monkeypatch.setattr("eval.run_local_planner_eval.run_local_planner_benchmark", fake_run_local_planner_benchmark)
    monkeypatch.setattr("eval.run_local_planner_eval.run_planner_eval", fake_run_planner_eval)

    exit_code = run_local_planner_eval(
        training_manifest=manifest_path,
        output_dir=output_dir,
        run_id="planner-local",
        model_name="local-9b",
        adapter_path=tmp_path / "planner_adapter",
        max_new_tokens=64,
        max_memory_gb=24,
        predicted_prepared_output=output_dir / "planner-local.predicted_prepared.jsonl",
    )

    assert exit_code == 0
    assert benchmark_calls[0]["input_path"] == input_path
    assert benchmark_calls[0]["adapter_path"] == tmp_path / "planner_adapter"
    assert planner_eval_calls[0]["input_path"] == input_path
    assert planner_eval_calls[0]["planner_source"] == "json_planner_predictions"
    assert planner_eval_calls[0]["planner_predictions_path"] == output_dir / "planner-local.predictions.jsonl"
