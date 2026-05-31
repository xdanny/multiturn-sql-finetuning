from __future__ import annotations

import json

import pytest

from eval.compare_semantic_value_retrieval import (
    compare_semantic_value_retrieval_manifest_files,
    compare_semantic_value_retrieval_manifests,
)
from eval.result_manifest import sha256_file


def _direct_manifest(
    *,
    row_count: int = 2,
    value_accuracy: float = 0.5,
    strict_accuracy: float = 0.25,
    split_role: str | None = None,
) -> dict:
    metrics = {
        "value_execution_accuracy": value_accuracy,
        "strict_execution_accuracy": strict_accuracy,
    }
    if split_role is not None:
        metrics["split_roles"] = {split_role: row_count}
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
        "row_count": row_count,
        "metrics": metrics,
        "command": ["run-direct"],
    }


def _semantic_manifest(
    value_index_sha: str = "value-index-sha",
    *,
    row_count: int = 2,
    value_accuracy: float = 0.75,
    strict_accuracy: float = 0.5,
    split_role: str | None = None,
) -> dict:
    metrics = {
        "value_execution_accuracy": value_accuracy,
        "strict_execution_accuracy": strict_accuracy,
        "value_index_manifest_sha256": value_index_sha,
        "value_index_index_source": "database_contents",
    }
    if split_role is not None:
        metrics["split_roles"] = {split_role: row_count}
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
        "row_count": row_count,
        "metrics": metrics,
        "command": ["run-semantic"],
    }


def _rows(count: int = 2) -> list[dict]:
    return [
        {
            "id": f"dialog-a:{index}",
            "dialog_id": "dialog-a",
            "turn_index": index,
            "database_id": "store",
            "reference_sql": f"SELECT {index};",
            "evaluation_mode": "non_oracle_generation",
            "value_execution_score": 1.0 if index % 2 == 0 else 0.0,
            "strict_execution_score": 1.0 if index % 2 == 0 else 0.0,
        }
        for index in range(count)
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
    assert metrics["semantic_value_retrieval_promotion_ready"] is False
    assert "comparable row count below minimum 24" in metrics[
        "semantic_value_retrieval_promotion_blockers"
    ]
    assert "# compared-with-direct-sql" in compared["command"]


def test_compare_semantic_value_retrieval_marks_clean_holdout_win_promotable() -> None:
    compared = compare_semantic_value_retrieval_manifests(
        semantic_manifest=_semantic_manifest(
            row_count=24,
            value_accuracy=0.70,
            strict_accuracy=0.60,
            split_role="clean_local_holdout",
        ),
        direct_manifest=_direct_manifest(
            row_count=24,
            value_accuracy=0.60,
            strict_accuracy=0.60,
            split_role="clean_local_holdout",
        ),
        semantic_rows=_rows(24),
        direct_rows=_rows(24),
        value_index_manifest_sha256="value-index-sha",
    )

    metrics = compared["metrics"]
    assert metrics["semantic_value_retrieval_promotion_ready"] is True
    assert metrics["semantic_value_retrieval_promotion_blockers"] == []
    assert metrics["semantic_value_retrieval_promotion_policy"]["required_split_role"] == (
        "clean_local_holdout"
    )


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
    semantic_rows[0]["semantic_context_pruned_by_oracle_labels"] = True

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
