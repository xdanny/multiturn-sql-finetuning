from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_local_rollout_comparison import (
    run_local_rollout_comparison,
    validate_local_rollout_training_manifest,
    write_rollout_comparison_preflight,
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


def test_validate_local_rollout_training_manifest_requires_prepared_non_oracle_eval(
    tmp_path,
) -> None:
    manifest_path = tmp_path / "training.manifest.json"
    eval_path = tmp_path / "eval.jsonl"
    _write_json(
        manifest_path,
        _training_manifest(
            stage="behavior_recovery",
            benchmark="synthetic_behavior_recovery",
            mode="non_oracle_generation",
            eval_path=eval_path,
        ),
    )

    with pytest.raises(ValueError, match="benchmark=prepared"):
        validate_local_rollout_training_manifest(manifest_path)


def test_run_local_rollout_comparison_runs_teacher_forced_then_rollout_then_compares(
    tmp_path, monkeypatch
) -> None:
    manifest_path = tmp_path / "training.manifest.json"
    eval_path = tmp_path / "eval.jsonl"
    output_dir = tmp_path / "results"
    _write_json(
        manifest_path,
        _training_manifest(
            stage="behavior_recovery",
            benchmark="prepared",
            mode="non_oracle_generation",
            eval_path=eval_path,
        ),
    )
    eval_path.write_text(
        json.dumps(
            {
                "dialog_id": "dialog-a",
                "database_id": "music",
                "source": "unit",
                "history_policy": "gold_sql_teacher_forced",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "List singers."},
                    {"role": "assistant", "content": "SELECT name FROM singer;"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        )
        + "\n"
    )
    teacher_calls: list[dict] = []
    rollout_calls: list[dict] = []
    compare_calls: list[dict] = []

    def fake_teacher(**kwargs):
        teacher_calls.append(kwargs)
        kwargs["manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["output"].stem,
                    "output_path": str(kwargs["output"]),
                    "row_count": 1,
                    "benchmark": "prepared",
                    "evaluation_mode": "non_oracle_generation",
                    "model_name": kwargs["model_name"],
                    "input_sha256": "shared",
                    "metrics": {
                        "history_policy": "gold_sql_teacher_forced",
                        "value_execution_accuracy": 0.5,
                        "strict_execution_accuracy": 0.5,
                    },
                }
            )
        )
        kwargs["output"].write_text(
            json.dumps(
                {
                    "id": "dialog-a:0",
                    "dialog_id": "dialog-a",
                    "turn_index": 0,
                    "database_id": "music",
                    "reference_sql": "SELECT name FROM singer;",
                    "history_policy": "gold_sql_teacher_forced",
                }
            )
            + "\n"
        )
        return 0

    def fake_rollout(**kwargs):
        rollout_calls.append(kwargs)
        kwargs["manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["output_path"].stem,
                    "output_path": str(kwargs["output_path"]),
                    "row_count": 1,
                    "benchmark": "prepared_rollout",
                    "evaluation_mode": "non_oracle_generation",
                    "model_name": kwargs["model_name"],
                    "input_sha256": "shared",
                    "metrics": {
                        "history_policy": "model_generated_sql_rollout",
                        "value_execution_accuracy": 1.0,
                        "strict_execution_accuracy": 1.0,
                    },
                }
            )
        )
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "id": "dialog-a:0",
                    "dialog_id": "dialog-a",
                    "turn_index": 0,
                    "database_id": "music",
                    "reference_sql": "SELECT name FROM singer;",
                    "history_policy": "model_generated_sql_rollout",
                }
            )
            + "\n"
        )
        return 0

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {"compared": True}

    monkeypatch.setattr("eval.run_local_rollout_comparison.run_local_benchmark", fake_teacher)
    monkeypatch.setattr(
        "eval.run_local_rollout_comparison.run_local_rollout_benchmark",
        fake_rollout,
    )
    monkeypatch.setattr(
        "eval.run_local_rollout_comparison.compare_rollout_manifest_files",
        fake_compare,
    )

    exit_code = run_local_rollout_comparison(
        training_manifest=manifest_path,
        output_dir=output_dir,
        run_id="rollout-local",
        model_name="local-9b",
        adapter_path=tmp_path / "adapter",
        database_root=tmp_path / "db",
        max_new_tokens=64,
        max_memory_gb=24,
    )

    assert exit_code == 0
    assert teacher_calls[0]["input_path"] == eval_path
    assert rollout_calls[0]["input_path"] == eval_path
    assert teacher_calls[0]["adapter_path"] == tmp_path / "adapter"
    assert rollout_calls[0]["adapter_path"] == tmp_path / "adapter"
    assert compare_calls == [
        {
            "rollout_manifest_path": output_dir / "rollout-local.rollout.manifest.json",
            "teacher_forced_manifest_path": output_dir / "rollout-local.teacher_forced.manifest.json",
            "output_path": output_dir / "rollout-local.compared.manifest.json",
            "repo_root": Path("."),
        }
    ]


def test_write_rollout_comparison_preflight_records_ready_pair(tmp_path: Path) -> None:
    manifest_path = tmp_path / "training.manifest.json"
    eval_path = tmp_path / "eval.jsonl"
    output_path = tmp_path / "rollout_comparison_preflight.json"
    _write_json(
        manifest_path,
        _training_manifest(
            stage="behavior_recovery",
            benchmark="prepared",
            mode="non_oracle_generation",
            eval_path=eval_path,
        ),
    )
    eval_path.write_text(
        json.dumps(
            {
                "dialog_id": "dialog-a",
                "database_id": "music",
                "source": "unit",
                "history_policy": "gold_sql_teacher_forced",
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "List singers."},
                    {"role": "assistant", "content": "SELECT name FROM singer;"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        )
        + "\n"
    )

    payload = write_rollout_comparison_preflight(
        training_manifest=manifest_path,
        output_path=output_path,
    )

    assert payload["artifact_type"] == "rollout_comparison_preflight"
    assert payload["status"] == "ready_for_local_rollout_pair"
    assert payload["claim_boundary"] == "preflight only; no rollout execution claim"
    assert payload["training_manifest_path"] == str(manifest_path)
    assert payload["input_path"] == str(eval_path)
    assert payload["stage"] == "behavior_recovery"
    assert payload["benchmark"] == "prepared"
    assert payload["evaluation_mode"] == "non_oracle_generation"
    assert payload["row_count"] == 1
    assert json.loads(output_path.read_text()) == payload
