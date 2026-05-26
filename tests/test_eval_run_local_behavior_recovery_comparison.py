from __future__ import annotations

import json
from pathlib import Path

from eval.run_local_behavior_recovery_comparison import run_local_behavior_recovery_comparison


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _training_manifest(*, stage: str, benchmark: str, train_path: Path) -> dict:
    return {
        "schema_version": 1,
        "run_id": f"{stage}-run",
        "status": "validated",
        "stage": stage,
        "benchmark": benchmark,
        "training_target": stage,
        "evaluation_mode": "non_oracle_generation",
        "train_data_path": str(train_path),
        "train_data_sha256": "train-hash",
        "output_dir": "outputs/unit",
        "final_dir": "outputs/unit/final",
        "command": ["train.finetune", "validate-data-only"],
    }


def test_run_local_behavior_recovery_comparison_generates_scores_and_compares(
    tmp_path, monkeypatch
) -> None:
    recovery_manifest_path = tmp_path / "recovery.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    recovery_input = tmp_path / "recovery.jsonl"
    direct_input = tmp_path / "direct.jsonl"
    output_dir = tmp_path / "results"
    _write_json(
        recovery_manifest_path,
        _training_manifest(
            stage="behavior_recovery",
            benchmark="synthetic_behavior_recovery",
            train_path=recovery_input,
        ),
    )
    _write_json(
        direct_manifest_path,
        _training_manifest(
            stage="direct_sql_control",
            benchmark="behavior_recovery_direct_sql",
            train_path=direct_input,
        ),
    )
    row = {
        "id": "recovery-1",
        "fixture_id": "recovery_empty_result",
        "reference_sql": "SELECT 1",
        "messages": [{"role": "user", "content": "repair it"}],
    }
    _write_jsonl(recovery_input, [row])
    _write_jsonl(direct_input, [row])

    generation_calls: list[dict] = []
    eval_calls: list[dict] = []
    compare_calls: list[dict] = []

    def fake_run_local_text_benchmark(**kwargs):
        generation_calls.append(kwargs)
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "id": "recovery-1",
                    "fixture_id": "recovery_empty_result",
                    "reference_sql": "SELECT 1",
                    "evaluation_mode": "non_oracle_generation",
                    "generated_sql": "SELECT 1",
                }
            )
            + "\n"
        )
        return 0

    def fake_run_direct_sql_eval(**kwargs):
        eval_calls.append(kwargs)
        benchmark = kwargs["benchmark"]
        success = benchmark == "behavior_recovery"
        kwargs["manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["output_path"].stem,
                    "output_path": str(kwargs["output_path"]),
                    "row_count": 1,
                    "benchmark": benchmark,
                    "evaluation_mode": "non_oracle_generation",
                    "metrics": {
                        "value_execution_accuracy": 1.0 if success else 0.0,
                        "strict_execution_accuracy": 1.0 if success else 0.0,
                        "recovery_evaluated_rows": 1,
                        "recovery_success_rate": 1.0 if success else 0.0,
                    },
                }
            )
        )
        kwargs["output_path"].write_text(
            json.dumps(
                {
                    "id": "recovery-1",
                    "fixture_id": "recovery_empty_result",
                    "evaluation_mode": "non_oracle_generation",
                    "reference_sql": "SELECT 1",
                    "database_path": "recovery.sqlite",
                    "value_execution_score": 1.0 if success else 0.0,
                    "strict_execution_score": 1.0 if success else 0.0,
                    "recovery_required": True,
                    "recovery_success": success,
                }
            )
            + "\n"
        )
        return 0

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {"compared": True}

    monkeypatch.setattr(
        "eval.run_local_behavior_recovery_comparison.run_local_text_benchmark",
        fake_run_local_text_benchmark,
    )
    monkeypatch.setattr(
        "eval.run_local_behavior_recovery_comparison.run_direct_sql_eval",
        fake_run_direct_sql_eval,
    )
    monkeypatch.setattr(
        "eval.run_local_behavior_recovery_comparison.compare_behavior_recovery_direct_sql_manifest_files",
        fake_compare,
    )

    exit_code = run_local_behavior_recovery_comparison(
        behavior_recovery_training_manifest=recovery_manifest_path,
        direct_training_manifest=direct_manifest_path,
        output_dir=output_dir,
        run_id="behavior-local",
        model_name="local-9b",
        behavior_recovery_adapter_path=tmp_path / "recovery_adapter",
        direct_adapter_path=tmp_path / "direct_adapter",
        max_new_tokens=64,
        max_memory_gb=24,
    )

    assert exit_code == 0
    assert [call["adapter_path"] for call in generation_calls] == [
        tmp_path / "recovery_adapter",
        tmp_path / "direct_adapter",
    ]
    assert [call["benchmark"] for call in eval_calls] == [
        "behavior_recovery",
        "behavior_recovery_direct_sql",
    ]
    assert compare_calls == [
        {
            "behavior_recovery_manifest_path": output_dir
            / "behavior-local.behavior_recovery.manifest.json",
            "direct_sql_manifest_path": output_dir / "behavior-local.direct_sql.manifest.json",
            "output_path": output_dir / "behavior-local.compared.manifest.json",
            "repo_root": Path("."),
        }
    ]
