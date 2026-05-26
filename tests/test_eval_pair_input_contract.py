from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.pair_input_contract import PairInputSpec, validate_pair_input_paths


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_validate_pair_input_paths_accepts_matching_row_identity(tmp_path: Path) -> None:
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    rows = [{"fixture_id": "metric-1", "reference_sql": "SELECT 1"}]
    _write_jsonl(left_path, rows)
    _write_jsonl(right_path, rows)

    validated = validate_pair_input_paths(
        left_input_path=left_path,
        right_input_path=right_path,
        spec=PairInputSpec(
            label="metric-DSL",
            left_name="metric-DSL",
            right_name="direct SQL",
            row_identity_fields=("fixture_id", "reference_sql"),
        ),
    )

    assert validated["row_count"] == 1
    assert validated["left_rows"][0]["fixture_id"] == "metric-1"


def test_validate_pair_input_paths_rejects_empty_inputs(tmp_path: Path) -> None:
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    _write_jsonl(left_path, [])
    _write_jsonl(right_path, [])

    with pytest.raises(ValueError, match="non-empty"):
        validate_pair_input_paths(
            left_input_path=left_path,
            right_input_path=right_path,
            spec=PairInputSpec(
                label="metric-DSL",
                left_name="metric-DSL",
                right_name="direct SQL",
                row_identity_fields=("fixture_id", "reference_sql"),
            ),
        )


def test_validate_pair_input_paths_rejects_identity_mismatch(tmp_path: Path) -> None:
    left_path = tmp_path / "left.jsonl"
    right_path = tmp_path / "right.jsonl"
    _write_jsonl(left_path, [{"fixture_id": "metric-1", "reference_sql": "SELECT 1"}])
    _write_jsonl(right_path, [{"fixture_id": "metric-2", "reference_sql": "SELECT 1"}])

    with pytest.raises(ValueError, match="row identity"):
        validate_pair_input_paths(
            left_input_path=left_path,
            right_input_path=right_path,
            spec=PairInputSpec(
                label="metric-DSL",
                left_name="metric-DSL",
                right_name="direct SQL",
                row_identity_fields=("fixture_id", "reference_sql"),
            ),
        )
