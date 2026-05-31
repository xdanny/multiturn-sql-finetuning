from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from eval.clean_holdout_failure_analysis import build_clean_holdout_failure_analysis


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _result_row(
    *,
    row_id: str,
    model_name: str,
    reference_sql: str,
    generated_sql: str,
    value_execution_score: float,
) -> dict:
    return {
        "id": row_id,
        "dialog_id": row_id.split(":")[0],
        "turn_index": int(row_id.split(":")[1]),
        "model_name": model_name,
        "messages": [{"role": "user", "content": "owners(id, name)\npets(id, name)"}],
        "reference_sql": reference_sql,
        "generated_sql": generated_sql,
        "value_execution_score": value_execution_score,
        "strict_execution_score": value_execution_score,
        "execution_score": value_execution_score,
        "syntax_valid": True,
    }


def _result_manifest(output_path: Path, *, model_name: str, row_hash: str = "rows") -> dict:
    return {
        "schema_version": 1,
        "run_id": output_path.stem,
        "benchmark": "prepared",
        "model_name": model_name,
        "prompt_variant": None,
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "output_path": str(output_path),
        "output_sha256": "hash",
        "row_count": 2,
        "metrics": {
            "split_row_ids_sha256": row_hash,
            "split_eval_turn_ids_sha256": "turns",
        },
    }


def _write_fixture(tmp_path: Path, *, lora_row_hash: str = "rows") -> Path:
    config_path = tmp_path / "configs" / "direct_sql_full_non_oracle.yaml"
    result_dir = tmp_path / "results" / "runs" / "direct_sql_full_non_oracle_control"
    base_output = result_dir / "base_clean_holdout.jsonl"
    lora_output = result_dir / "lora_clean_holdout.jsonl"
    base_manifest = result_dir / "base_clean_holdout.manifest.json"
    lora_manifest = result_dir / "lora_clean_holdout.manifest.json"
    config = {
        "checkpoint3_artifacts": {
            "result_manifests": {
                "base_clean_holdout": {
                    "model_role": "base",
                    "split_id": "cosql_dev_clean_holdout_v1",
                    "split_role": "clean_local_holdout",
                    "manifest": str(base_manifest.relative_to(tmp_path)),
                },
                "lora_clean_holdout": {
                    "model_role": "lora",
                    "split_id": "cosql_dev_clean_holdout_v1",
                    "split_role": "clean_local_holdout",
                    "manifest": str(lora_manifest.relative_to(tmp_path)),
                },
            }
        }
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    _write_jsonl(
        base_output,
        [
            _result_row(
                row_id="dialog-1:0",
                model_name="base-served",
                reference_sql="SELECT name FROM owners;",
                generated_sql="SELECT name FROM owners;",
                value_execution_score=1.0,
            ),
            _result_row(
                row_id="dialog-2:0",
                model_name="base-served",
                reference_sql="SELECT name FROM owners;",
                generated_sql="SELECT name FROM pets;",
                value_execution_score=0.0,
            ),
        ],
    )
    _write_jsonl(
        lora_output,
        [
            _result_row(
                row_id="dialog-1:0",
                model_name="lora-served",
                reference_sql="SELECT name FROM owners;",
                generated_sql="SELECT name FROM owners;",
                value_execution_score=1.0,
            ),
            _result_row(
                row_id="dialog-2:0",
                model_name="lora-served",
                reference_sql="SELECT name FROM owners;",
                generated_sql="SELECT name FROM owners;",
                value_execution_score=1.0,
            ),
        ],
    )
    _write_json(base_manifest, _result_manifest(base_output, model_name="base-served"))
    _write_json(
        lora_manifest,
        _result_manifest(lora_output, model_name="lora-served", row_hash=lora_row_hash),
    )
    return config_path


def test_clean_holdout_failure_analysis_writes_artifacts(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path)
    output_dir = tmp_path / "analysis"

    manifest = build_clean_holdout_failure_analysis(
        config_path=config_path,
        output_dir=output_dir,
        command=["test-command"],
    )

    assert manifest["artifact_type"] == "clean_holdout_failure_analysis"
    assert manifest["split_id"] == "cosql_dev_clean_holdout_v1"
    assert manifest["primary_error_counts"]["base"] == {"correct": 1, "schema_link": 1}
    assert manifest["primary_error_counts"]["lora"] == {"correct": 2}
    assert manifest["roadmap_method_hint_counts"]["base"] == {"planner_schema_linking": 1}
    assert (output_dir / "base_clean_holdout.classified.jsonl").exists()
    assert (output_dir / "lora_clean_holdout.error_summary.csv").exists()
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8")) == manifest

    with (output_dir / "pairwise_vs_base.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["candidate"] == "lora-served"
    assert rows[0]["baseline"] == "base-served"
    assert rows[0]["fixed_turns"] == "1"
    assert rows[0]["net_fixed"] == "1"


def test_clean_holdout_failure_analysis_requires_same_rows(tmp_path: Path) -> None:
    config_path = _write_fixture(tmp_path, lora_row_hash="different-rows")

    with pytest.raises(ValueError, match="split_row_ids_sha256 mismatch"):
        build_clean_holdout_failure_analysis(config_path=config_path)
