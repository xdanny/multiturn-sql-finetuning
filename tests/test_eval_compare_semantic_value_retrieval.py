from __future__ import annotations

import json

import pytest

from eval.compare_semantic_value_retrieval import (
    compare_semantic_value_retrieval_manifest_files,
    compare_semantic_value_retrieval_manifests,
)
from eval.result_manifest import sha256_file


def _direct_manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "direct",
        "benchmark": "prepared",
        "model_name": "local-9b",
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/direct.jsonl",
        "input_sha256": "direct-input",
        "output_path": "results/direct.jsonl",
        "output_sha256": "direct-output",
        "row_count": 2,
        "metrics": {
            "value_execution_accuracy": 0.5,
            "strict_execution_accuracy": 0.25,
        },
        "command": ["run-direct"],
    }


def _semantic_manifest(value_index_sha: str = "value-index-sha") -> dict:
    return {
        "schema_version": 1,
        "run_id": "semantic",
        "benchmark": "prepared",
        "model_name": "local-9b",
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/semantic.jsonl",
        "input_sha256": "semantic-input",
        "output_path": "results/semantic.jsonl",
        "output_sha256": "semantic-output",
        "row_count": 2,
        "metrics": {
            "value_execution_accuracy": 0.75,
            "strict_execution_accuracy": 0.5,
            "value_index_manifest_sha256": value_index_sha,
            "value_index_index_source": "database_contents",
        },
        "command": ["run-semantic"],
    }


def _rows() -> list[dict]:
    return [
        {
            "id": "dialog-a:0",
            "dialog_id": "dialog-a",
            "turn_index": 0,
            "database_id": "store",
            "reference_sql": "SELECT COUNT(*) FROM orders;",
            "evaluation_mode": "non_oracle_generation",
            "value_execution_score": 1.0,
            "strict_execution_score": 1.0,
        },
        {
            "id": "dialog-a:1",
            "dialog_id": "dialog-a",
            "turn_index": 1,
            "database_id": "store",
            "reference_sql": "SELECT SUM(amount) FROM orders;",
            "evaluation_mode": "non_oracle_generation",
            "value_execution_score": 0.0,
            "strict_execution_score": 0.0,
        },
    ]


def test_compare_semantic_value_retrieval_manifests_adds_direct_sql_delta() -> None:
    compared = compare_semantic_value_retrieval_manifests(
        semantic_manifest=_semantic_manifest(),
        direct_manifest=_direct_manifest(),
        semantic_rows=_rows(),
        direct_rows=_rows(),
        value_index_manifest_sha256="value-index-sha",
    )

    metrics = compared["metrics"]
    assert compared["run_id"] == "semantic"
    assert metrics["direct_sql_comparison_run_id"] == "direct"
    assert metrics["direct_sql_model_name"] == "local-9b"
    assert metrics["direct_sql_value_execution_accuracy"] == 0.5
    assert metrics["semantic_value_retrieval_value_delta_vs_direct_sql"] == pytest.approx(
        0.25
    )
    assert metrics["semantic_value_retrieval_strict_delta_vs_direct_sql"] == pytest.approx(
        0.25
    )
    assert metrics["semantic_value_retrieval_comparable_row_count"] == 2
    assert metrics["semantic_value_retrieval_comparer"] == (
        "eval.compare_semantic_value_retrieval"
    )
    assert "# compared-with-direct-sql" in compared["command"]


def test_compare_semantic_value_retrieval_rejects_value_index_mismatch() -> None:
    with pytest.raises(ValueError, match="value-index manifest"):
        compare_semantic_value_retrieval_manifests(
            semantic_manifest=_semantic_manifest(value_index_sha="other"),
            direct_manifest=_direct_manifest(),
            semantic_rows=_rows(),
            direct_rows=_rows(),
            value_index_manifest_sha256="value-index-sha",
        )


def test_compare_semantic_value_retrieval_rejects_oracle_rows() -> None:
    semantic_rows = _rows()
    semantic_rows[0]["semantic_model_oracle_derived"] = True

    with pytest.raises(ValueError, match="oracle"):
        compare_semantic_value_retrieval_manifests(
            semantic_manifest=_semantic_manifest(),
            direct_manifest=_direct_manifest(),
            semantic_rows=semantic_rows,
            direct_rows=_rows(),
            value_index_manifest_sha256="value-index-sha",
        )


def test_compare_semantic_value_retrieval_rejects_row_mismatch() -> None:
    direct_rows = _rows()
    direct_rows[1]["turn_index"] = 7

    with pytest.raises(ValueError, match="row identity mismatch"):
        compare_semantic_value_retrieval_manifests(
            semantic_manifest=_semantic_manifest(),
            direct_manifest=_direct_manifest(),
            semantic_rows=_rows(),
            direct_rows=direct_rows,
            value_index_manifest_sha256="value-index-sha",
        )


def test_compare_semantic_value_retrieval_manifest_files_writes_augmented_manifest(
    tmp_path,
) -> None:
    value_index_manifest_path = tmp_path / "docs" / "value_index.manifest.json"
    value_index_manifest_path.parent.mkdir(parents=True)
    value_index_manifest_path.write_text(json.dumps({"artifact_type": "non_oracle_value_index_v1"}))
    value_index_sha = sha256_file(value_index_manifest_path)
    assert value_index_sha is not None

    semantic_output = tmp_path / "results" / "semantic.jsonl"
    direct_output = tmp_path / "results" / "direct.jsonl"
    semantic_output.parent.mkdir(parents=True)
    semantic_output.write_text("\n".join(json.dumps(row) for row in _rows()) + "\n")
    direct_output.write_text("\n".join(json.dumps(row) for row in _rows()) + "\n")

    semantic_manifest = _semantic_manifest(value_index_sha=value_index_sha)
    semantic_manifest["output_path"] = str(semantic_output.relative_to(tmp_path))
    direct_manifest = _direct_manifest()
    direct_manifest["output_path"] = str(direct_output.relative_to(tmp_path))
    semantic_manifest_path = tmp_path / "semantic.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    output_path = tmp_path / "semantic.compared.manifest.json"
    semantic_manifest_path.write_text(json.dumps(semantic_manifest))
    direct_manifest_path.write_text(json.dumps(direct_manifest))

    compared = compare_semantic_value_retrieval_manifest_files(
        semantic_manifest_path=semantic_manifest_path,
        direct_manifest_path=direct_manifest_path,
        value_index_manifest_path=value_index_manifest_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert json.loads(output_path.read_text()) == compared
    assert compared["metrics"]["value_index_manifest_sha256"] == value_index_sha
