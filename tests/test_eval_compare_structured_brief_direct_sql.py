from __future__ import annotations

import json

import pytest

from eval.compare_structured_brief_direct_sql import (
    compare_structured_brief_manifest_files,
    compare_structured_brief_manifests,
)


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


def _structured_manifest(
    *,
    row_count: int = 2,
    value_accuracy: float = 0.75,
    strict_accuracy: float = 0.5,
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
        "run_id": "structured",
        "benchmark": "prepared",
        "model_name": "local-9b",
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "input_path": "data/structured.jsonl",
        "input_sha256": "structured-input",
        "output_path": "results/structured.jsonl",
        "output_sha256": "structured-output",
        "row_count": row_count,
        "metrics": metrics,
        "command": ["run-structured"],
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


def test_compare_structured_brief_manifests_adds_direct_sql_delta() -> None:
    compared = compare_structured_brief_manifests(
        structured_manifest=_structured_manifest(),
        direct_manifest=_direct_manifest(),
        structured_rows=_rows(),
        direct_rows=_rows(),
    )

    metrics = compared["metrics"]
    assert compared["run_id"] == "structured"
    assert metrics["direct_sql_comparison_run_id"] == "direct"
    assert metrics["direct_sql_model_name"] == "local-9b"
    assert metrics["direct_sql_value_execution_accuracy"] == 0.5
    assert metrics["structured_brief_value_delta_vs_direct_sql"] == pytest.approx(0.25)
    assert metrics["structured_brief_strict_delta_vs_direct_sql"] == pytest.approx(0.25)
    assert metrics["structured_brief_comparable_row_count"] == 2
    assert metrics["structured_brief_comparer"] == "eval.compare_structured_brief_direct_sql"
    assert metrics["structured_brief_promotion_ready"] is False
    assert "comparable row count below minimum 24" in metrics[
        "structured_brief_promotion_blockers"
    ]
    assert "# compared-with-direct-sql" in compared["command"]


def test_compare_structured_brief_marks_clean_holdout_win_promotable() -> None:
    compared = compare_structured_brief_manifests(
        structured_manifest=_structured_manifest(
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
        structured_rows=_rows(24),
        direct_rows=_rows(24),
    )

    metrics = compared["metrics"]
    assert metrics["structured_brief_promotion_ready"] is True
    assert metrics["structured_brief_promotion_blockers"] == []
    assert metrics["structured_brief_promotion_policy"]["required_split_role"] == (
        "clean_local_holdout"
    )


def test_compare_structured_brief_rejects_oracle_rows() -> None:
    structured_rows = _rows()
    structured_rows[0]["uses_oracle_planning_hints"] = True

    with pytest.raises(ValueError, match="oracle"):
        compare_structured_brief_manifests(
            structured_manifest=_structured_manifest(),
            direct_manifest=_direct_manifest(),
            structured_rows=structured_rows,
            direct_rows=_rows(),
        )


def test_compare_structured_brief_rejects_row_mismatch() -> None:
    direct_rows = _rows()
    direct_rows[1]["turn_index"] = 7

    with pytest.raises(ValueError, match="row identity mismatch"):
        compare_structured_brief_manifests(
            structured_manifest=_structured_manifest(),
            direct_manifest=_direct_manifest(),
            structured_rows=_rows(),
            direct_rows=direct_rows,
        )


def test_compare_structured_brief_manifest_files_writes_augmented_manifest(tmp_path) -> None:
    structured_output = tmp_path / "results" / "structured.jsonl"
    direct_output = tmp_path / "results" / "direct.jsonl"
    structured_output.parent.mkdir(parents=True)
    structured_output.write_text("\n".join(json.dumps(row) for row in _rows()) + "\n")
    direct_output.write_text("\n".join(json.dumps(row) for row in _rows()) + "\n")

    structured_manifest = _structured_manifest()
    structured_manifest["output_path"] = str(structured_output.relative_to(tmp_path))
    direct_manifest = _direct_manifest()
    direct_manifest["output_path"] = str(direct_output.relative_to(tmp_path))
    structured_manifest_path = tmp_path / "structured.manifest.json"
    direct_manifest_path = tmp_path / "direct.manifest.json"
    output_path = tmp_path / "structured.compared.manifest.json"
    structured_manifest_path.write_text(json.dumps(structured_manifest))
    direct_manifest_path.write_text(json.dumps(direct_manifest))

    compared = compare_structured_brief_manifest_files(
        structured_manifest_path=structured_manifest_path,
        direct_manifest_path=direct_manifest_path,
        output_path=output_path,
        repo_root=tmp_path,
    )

    assert json.loads(output_path.read_text()) == compared
