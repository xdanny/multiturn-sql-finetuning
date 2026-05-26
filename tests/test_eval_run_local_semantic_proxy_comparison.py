from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_local_semantic_proxy_comparison import (
    run_local_semantic_proxy_comparison,
    validate_local_semantic_proxy_training_manifests,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _training_manifest(*, stage: str, benchmark: str, train_path: Path, eval_path: Path) -> dict:
    return {
        "schema_version": 1,
        "run_id": f"{stage}-run",
        "status": "validated",
        "stage": stage,
        "benchmark": benchmark,
        "training_target": stage,
        "evaluation_mode": "non_oracle_generation",
        "train_data_path": str(train_path),
        "eval_data_path": str(eval_path),
        "train_data_sha256": "train-hash",
        "eval_data_sha256": "eval-hash",
        "output_dir": "outputs/unit",
        "final_dir": "outputs/unit/final",
        "command": ["train.finetune", "validate-data-only"],
    }


def _prepared_row(*, user_content: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "SELECT name FROM singer;"},
        ],
        "dialog_id": "dialog-a",
        "database_id": "music",
        "evaluation_mode": "non_oracle_generation",
        "gold_plans": [{}],
        "schema_link_labels": [{}],
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
    }


def test_validate_local_semantic_proxy_training_manifests_requires_expected_pair(
    tmp_path: Path,
) -> None:
    semantic_manifest_path = tmp_path / "semantic.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    semantic_train = tmp_path / "semantic_train.jsonl"
    semantic_eval = tmp_path / "semantic_eval.jsonl"
    direct_train = tmp_path / "direct_train.jsonl"
    direct_eval = tmp_path / "direct_eval.jsonl"
    _write_json(
        semantic_manifest_path,
        _training_manifest(
            stage="semantic_prompt",
            benchmark="cosql_semantic_proxy",
            train_path=semantic_train,
            eval_path=semantic_eval,
        ),
    )
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="direct_sql_control",
            benchmark="cosql_semantic_proxy_direct_sql",
            train_path=direct_train,
            eval_path=direct_eval,
        ),
    )
    _write_jsonl(
        semantic_eval,
        [_prepared_row(user_content="Schema/context:\nfoo\n\nSemantic model:\n- Cube foo\n\nQuestion:\nQ")],
    )
    _write_jsonl(direct_eval, [_prepared_row(user_content="Schema/context:\nfoo\n\nQuestion:\nQ")])

    with pytest.raises(ValueError, match="semantic_layer"):
        validate_local_semantic_proxy_training_manifests(
            semantic_training_manifest=semantic_manifest_path,
            direct_training_manifest=direct_manifest_path,
        )


def test_run_local_semantic_proxy_comparison_uses_eval_rows_and_compares(tmp_path: Path, monkeypatch) -> None:
    semantic_manifest_path = tmp_path / "semantic.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    semantic_train = tmp_path / "semantic_train.jsonl"
    semantic_eval = tmp_path / "semantic_eval.jsonl"
    direct_train = tmp_path / "direct_train.jsonl"
    direct_eval = tmp_path / "direct_eval.jsonl"
    output_dir = tmp_path / "results"
    _write_json(
        semantic_manifest_path,
        _training_manifest(
            stage="semantic_layer",
            benchmark="cosql_semantic_proxy",
            train_path=semantic_train,
            eval_path=semantic_eval,
        ),
    )
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="direct_sql_control",
            benchmark="cosql_semantic_proxy_direct_sql",
            train_path=direct_train,
            eval_path=direct_eval,
        ),
    )
    _write_jsonl(
        semantic_eval,
        [_prepared_row(user_content="Schema/context:\nfoo\n\nSemantic model:\n- Cube foo\n\nQuestion:\nQ")],
    )
    _write_jsonl(direct_eval, [_prepared_row(user_content="Schema/context:\nfoo\n\nQuestion:\nQ")])

    pair_calls: list[dict] = []
    compare_calls: list[dict] = []

    def fake_run_local_benchmark_pair(**kwargs):
        pair_calls.append(kwargs)
        outputs = {
            "method_output": output_dir / "semantic-proxy-local.semantic_proxy.jsonl",
            "method_manifest_output": output_dir / "semantic-proxy-local.semantic_proxy.manifest.json",
            "direct_output": output_dir / "semantic-proxy-local.direct.jsonl",
            "direct_manifest_output": output_dir / "semantic-proxy-local.direct.manifest.json",
            "compared_output": output_dir / "semantic-proxy-local.compared.manifest.json",
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs["method_manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": "semantic-proxy-local.semantic_proxy",
                    "output_path": str(outputs["method_output"]),
                    "row_count": 1,
                    "benchmark": "prepared",
                    "evaluation_mode": "non_oracle_generation",
                    "prompt_variant": "semantic_proxy",
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
                    "run_id": "semantic-proxy-local.direct",
                    "output_path": str(outputs["direct_output"]),
                    "row_count": 1,
                    "benchmark": "prepared",
                    "evaluation_mode": "non_oracle_generation",
                    "prompt_variant": "direct_sql_control",
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
        "eval.run_local_semantic_proxy_comparison.run_local_benchmark_pair",
        fake_run_local_benchmark_pair,
    )
    monkeypatch.setattr(
        "eval.run_local_semantic_proxy_comparison.compare_semantic_proxy_direct_sql_manifest_files",
        fake_compare,
    )

    exit_code = run_local_semantic_proxy_comparison(
        semantic_training_manifest=semantic_manifest_path,
        direct_training_manifest=direct_manifest_path,
        output_dir=output_dir,
        run_id="semantic-proxy-local",
        model_name="local-9b",
        semantic_adapter_path=tmp_path / "semantic_adapter",
        direct_adapter_path=tmp_path / "direct_adapter",
        database_root=tmp_path / "db",
        max_new_tokens=64,
        max_memory_gb=24,
    )

    assert exit_code == 0
    assert len(pair_calls) == 1
    assert pair_calls[0]["method_input_path"] == semantic_eval
    assert pair_calls[0]["direct_input_path"] == direct_eval
    assert pair_calls[0]["method_adapter_path"] == tmp_path / "semantic_adapter"
    assert pair_calls[0]["direct_adapter_path"] == tmp_path / "direct_adapter"
    assert compare_calls == [
        {
            "semantic_manifest_path": output_dir / "semantic-proxy-local.semantic_proxy.manifest.json",
            "direct_manifest_path": output_dir / "semantic-proxy-local.direct.manifest.json",
            "output_path": output_dir / "semantic-proxy-local.compared.manifest.json",
            "repo_root": Path("."),
        }
    ]
