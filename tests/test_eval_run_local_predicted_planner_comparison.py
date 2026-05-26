from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_local_predicted_planner_comparison import (
    run_local_predicted_planner_comparison,
    validate_local_predicted_planner_training_manifests,
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


def _record(*, mode: str, sql: str = "SELECT name FROM singer;") -> dict:
    record = {
        "id": "dialog-a",
        "database_id": "music",
        "source": "unit",
        "evaluation_mode": mode,
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
        "gold_plans": [
            {
                "parseable": True,
                "relevant_tables": ["singer"],
                "relevant_columns": ["singer.name"],
                "join_path": [],
                "query_skeleton": {"select": True},
                "projection_shape": {
                    "selected_count": 1,
                    "selected_expressions": ["singer.name"],
                    "preserve_duplicates": True,
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "sql"},
            {"role": "user", "content": "List singers."},
            {"role": "assistant", "content": sql},
        ],
    }
    if mode == "predicted_planner":
        record["predicted_plans"] = [
            {
                "parseable": True,
                "prediction_source": "unit_test",
                "relevant_tables": ["singer"],
                "relevant_columns": ["singer.name"],
                "join_path": [],
                "query_skeleton": {"select": True},
                "projection_shape": {
                    "selected_count": 1,
                    "selected_expressions": ["singer.name"],
                    "preserve_duplicates": True,
                },
            }
        ]
    return record


def test_validate_local_predicted_planner_training_manifests_requires_expected_pair(
    tmp_path,
) -> None:
    direct_manifest_path = tmp_path / "direct.manifest.json"
    predicted_manifest_path = tmp_path / "predicted.manifest.json"
    direct_input = tmp_path / "direct.jsonl"
    predicted_input = tmp_path / "predicted.jsonl"
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="direct_sql_control",
            benchmark="prepared",
            mode="non_oracle_generation",
            train_path=direct_input,
        ),
    )
    _write_json(
        predicted_manifest_path,
        _training_manifest(
            stage="planner_first_sql",
            benchmark="prepared",
            mode="predicted_planner",
            train_path=predicted_input,
        ),
    )
    _write_jsonl(direct_input, [_record(mode="non_oracle_generation")])
    _write_jsonl(predicted_input, [_record(mode="predicted_planner")])

    with pytest.raises(ValueError, match="predicted_planner"):
        validate_local_predicted_planner_training_manifests(
            direct_training_manifest=direct_manifest_path,
            predicted_training_manifest=predicted_manifest_path,
        )


def test_run_local_predicted_planner_comparison_runs_local_eval_then_compares(
    tmp_path, monkeypatch
) -> None:
    direct_manifest_path = tmp_path / "direct.manifest.json"
    predicted_manifest_path = tmp_path / "predicted.manifest.json"
    direct_input = tmp_path / "direct.jsonl"
    predicted_input = tmp_path / "predicted.jsonl"
    output_dir = tmp_path / "results"
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="direct_sql_control",
            benchmark="prepared",
            mode="non_oracle_generation",
            train_path=direct_input,
        ),
    )
    _write_json(
        predicted_manifest_path,
        _training_manifest(
            stage="predicted_planner",
            benchmark="prepared",
            mode="predicted_planner",
            train_path=predicted_input,
        ),
    )
    _write_jsonl(direct_input, [_record(mode="non_oracle_generation")])
    _write_jsonl(predicted_input, [_record(mode="predicted_planner")])

    pair_calls: list[dict] = []
    compare_calls: list[dict] = []

    def fake_run_local_benchmark_pair(**kwargs):
        pair_calls.append(kwargs)
        outputs = {
            "method_output": output_dir / "planner-local.predicted_planner.jsonl",
            "method_manifest_output": output_dir / "planner-local.predicted_planner.manifest.json",
            "direct_output": output_dir / "planner-local.direct.jsonl",
            "direct_manifest_output": output_dir / "planner-local.direct.manifest.json",
            "compared_output": output_dir / "planner-local.compared.manifest.json",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs["method_manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": "planner-local.predicted_planner",
                    "output_path": str(outputs["method_output"]),
                    "row_count": 1,
                    "benchmark": "prepared",
                    "evaluation_mode": "predicted_planner",
                    "model_name": kwargs["model_name"],
                    "metrics": {
                        "value_execution_accuracy": 1.0,
                        "strict_execution_accuracy": 1.0,
                    },
                }
            )
        )
        outputs["direct_manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": "planner-local.direct",
                    "output_path": str(outputs["direct_output"]),
                    "row_count": 1,
                    "benchmark": "prepared",
                    "evaluation_mode": "non_oracle_generation",
                    "model_name": kwargs["model_name"],
                    "metrics": {
                        "value_execution_accuracy": 0.5,
                        "strict_execution_accuracy": 0.5,
                    },
                }
            )
        )
        return outputs

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {"compared": True}

    monkeypatch.setattr(
        "eval.run_local_predicted_planner_comparison.run_local_benchmark_pair",
        fake_run_local_benchmark_pair,
    )
    monkeypatch.setattr(
        "eval.run_local_predicted_planner_comparison.compare_predicted_planner_manifest_files",
        fake_compare,
    )

    exit_code = run_local_predicted_planner_comparison(
        direct_training_manifest=direct_manifest_path,
        predicted_training_manifest=predicted_manifest_path,
        output_dir=output_dir,
        run_id="planner-local",
        model_name="local-9b",
        direct_adapter_path=tmp_path / "direct_adapter",
        predicted_adapter_path=tmp_path / "predicted_adapter",
        database_root=tmp_path / "db",
        max_new_tokens=64,
        max_memory_gb=24,
    )

    assert exit_code == 0
    assert len(pair_calls) == 1
    assert pair_calls[0]["direct_input_path"] == direct_input
    assert pair_calls[0]["method_input_path"] == predicted_input
    assert pair_calls[0]["direct_adapter_path"] == tmp_path / "direct_adapter"
    assert pair_calls[0]["method_adapter_path"] == tmp_path / "predicted_adapter"
    assert compare_calls == [
        {
            "predicted_manifest_path": output_dir / "planner-local.predicted_planner.manifest.json",
            "direct_manifest_path": output_dir / "planner-local.direct.manifest.json",
            "output_path": output_dir / "planner-local.compared.manifest.json",
            "repo_root": Path("."),
        }
    ]
