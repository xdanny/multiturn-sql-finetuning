from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.local_sql_pair import (
    SqlPairSpec,
    run_local_sql_pair_generation_and_eval,
    validate_sql_pair_training_manifests,
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


def _record(*, fixture_id: str) -> dict:
    return {
        "id": fixture_id,
        "fixture_id": fixture_id,
        "reference_sql": "SELECT name FROM singer;",
        "evaluation_mode": "non_oracle_generation",
        "oracle_policy": "non_oracle_inputs_only",
        "messages": [
            {"role": "system", "content": "sql"},
            {"role": "user", "content": "List singers."},
            {"role": "assistant", "content": "SELECT name FROM singer;"},
        ],
    }


def _spec() -> SqlPairSpec:
    return SqlPairSpec(
        method_name="semantic_layer",
        method_stage="semantic_layer",
        method_benchmark="synthetic_semantic_layer",
        method_prompt_variant="semantic_layer",
        direct_benchmark="synthetic_semantic_layer_direct_sql",
        comparison_label="semantic_layer",
    )


def test_validate_sql_pair_training_manifests_requires_expected_pair(tmp_path: Path) -> None:
    method_manifest_path = tmp_path / "method.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    method_input = tmp_path / "method.jsonl"
    direct_input = tmp_path / "direct.jsonl"
    _write_json(
        method_manifest_path,
        _training_manifest(
            stage="semantic_prompt",
            benchmark="synthetic_semantic_layer",
            mode="non_oracle_generation",
            train_path=method_input,
        ),
    )
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="direct_sql_control",
            benchmark="synthetic_semantic_layer_direct_sql",
            mode="non_oracle_generation",
            train_path=direct_input,
        ),
    )
    _write_jsonl(method_input, [_record(fixture_id="value_normalization_france")])
    _write_jsonl(direct_input, [_record(fixture_id="value_normalization_france")])

    with pytest.raises(ValueError, match="semantic_layer"):
        validate_sql_pair_training_manifests(
            method_training_manifest=method_manifest_path,
            direct_training_manifest=direct_manifest_path,
            spec=_spec(),
        )


def test_run_local_sql_pair_generation_and_eval_runs_generation_and_scoring(
    tmp_path: Path, monkeypatch
) -> None:
    method_manifest_path = tmp_path / "method.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    method_input = tmp_path / "method.jsonl"
    direct_input = tmp_path / "direct.jsonl"
    output_dir = tmp_path / "results"
    _write_json(
        method_manifest_path,
        _training_manifest(
            stage="semantic_layer",
            benchmark="synthetic_semantic_layer",
            mode="non_oracle_generation",
            train_path=method_input,
        ),
    )
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="direct_sql_control",
            benchmark="synthetic_semantic_layer_direct_sql",
            mode="non_oracle_generation",
            train_path=direct_input,
        ),
    )
    _write_jsonl(method_input, [_record(fixture_id="value_normalization_france")])
    _write_jsonl(direct_input, [_record(fixture_id="value_normalization_france")])

    generation_calls: list[dict] = []
    eval_calls: list[dict] = []

    def fake_run_local_text_benchmark(**kwargs):
        generation_calls.append(kwargs)
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "id": "value_normalization_france",
                    "fixture_id": "value_normalization_france",
                    "reference_sql": "SELECT name FROM singer;",
                    "evaluation_mode": "non_oracle_generation",
                    "generated_sql": "SELECT name FROM singer;",
                }
            )
            + "\n"
        )
        return 0

    def fake_run_direct_sql_eval(**kwargs):
        eval_calls.append(kwargs)
        kwargs["manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["output_path"].stem,
                    "output_path": str(kwargs["output_path"]),
                    "row_count": 1,
                    "benchmark": kwargs["benchmark"],
                    "evaluation_mode": "non_oracle_generation",
                    "model_name": kwargs["model_name"],
                    "metrics": {
                        "value_execution_accuracy": 1.0,
                        "strict_execution_accuracy": 1.0,
                        "execution_evaluated_rows": 1,
                    },
                }
            )
        )
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "id": "value_normalization_france",
                    "fixture_id": "value_normalization_france",
                    "reference_sql": "SELECT name FROM singer;",
                    "evaluation_mode": "non_oracle_generation",
                    "value_execution_score": 1.0,
                    "strict_execution_score": 1.0,
                }
            )
            + "\n"
        )
        return 0

    monkeypatch.setattr("eval.local_sql_pair.run_local_text_benchmark", fake_run_local_text_benchmark)
    monkeypatch.setattr("eval.local_sql_pair.run_direct_sql_eval", fake_run_direct_sql_eval)

    validated = validate_sql_pair_training_manifests(
        method_training_manifest=method_manifest_path,
        direct_training_manifest=direct_manifest_path,
        spec=_spec(),
    )
    outputs = run_local_sql_pair_generation_and_eval(
        validated=validated,
        spec=_spec(),
        output_dir=output_dir,
        run_id="semantic-local",
        model_name="local-9b",
        method_adapter_path=tmp_path / "method_adapter",
        direct_adapter_path=tmp_path / "direct_adapter",
        max_new_tokens=64,
        max_memory_gb=24,
    )

    assert [call["input_path"] for call in generation_calls] == [method_input, direct_input]
    assert generation_calls[0]["adapter_path"] == tmp_path / "method_adapter"
    assert generation_calls[1]["adapter_path"] == tmp_path / "direct_adapter"
    assert [call["benchmark"] for call in eval_calls] == [
        "synthetic_semantic_layer",
        "synthetic_semantic_layer_direct_sql",
    ]
    assert outputs["method_manifest_output"] == output_dir / "semantic-local.semantic_layer.manifest.json"
    assert outputs["direct_manifest_output"] == output_dir / "semantic-local.direct_sql.manifest.json"
