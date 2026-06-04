from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.semantic_context_transfer_inputs import (
    select_semantic_context_transfer_rows,
    write_semantic_context_transfer_input_artifacts,
)
from eval.result_manifest import sha256_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _dialog(index: int, *, history_policy: str = "gold_sql_teacher_forced") -> dict:
    return {
        "id": f"dialog-{index}",
        "dialog_id": f"dialog-{index}",
        "database_id": "db",
        "source": "unit",
        "split_id": "cosql_dev_clean_holdout_v1",
        "split_role": "clean_local_holdout",
        "split_row_id": f"cosql_dev:{index:04d}:db",
        "history_policy": history_policy,
        "evaluation_mode": "non_oracle_generation",
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
        "messages": [
            {"role": "system", "content": "Return SQL."},
            {"role": "user", "content": f"Show Alice {index}."},
            {"role": "assistant", "content": f"SELECT {index};"},
            {"role": "user", "content": f"Show Bob {index}."},
            {"role": "assistant", "content": f"SELECT {index + 10};"},
        ],
    }


def _value_index_rows() -> list[dict]:
    return [
        {
            "artifact_type": "non_oracle_value_index_entry",
            "schema_version": 1,
            "index_source": "database_contents",
            "database_id": "db",
            "table": "person",
            "column": "name",
            "raw_value": "Alice",
            "normalized_value": "alice",
            "aliases": ["Alice"],
            "source_frequency": 1,
        },
        {
            "artifact_type": "non_oracle_value_index_entry",
            "schema_version": 1,
            "index_source": "database_contents",
            "database_id": "db",
            "table": "person",
            "column": "name",
            "raw_value": "Bob",
            "normalized_value": "bob",
            "aliases": ["Bob"],
            "source_frequency": 1,
        },
    ]


def _write_value_index(tmp_path: Path) -> tuple[Path, Path]:
    value_index_path = tmp_path / "value_index.jsonl"
    manifest_path = tmp_path / "value_index.manifest.json"
    _write_jsonl(value_index_path, _value_index_rows())
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "non_oracle_value_index_v1",
                "index_source": "database_contents",
                "output_path": str(value_index_path),
                "output_sha256": sha256_file(value_index_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return value_index_path, manifest_path


def test_select_semantic_context_transfer_rows_excludes_single_turn(tmp_path: Path) -> None:
    input_path = tmp_path / "clean.jsonl"
    _write_jsonl(
        input_path,
        [
            _dialog(0, history_policy="single_turn"),
            _dialog(1),
            _dialog(2),
        ],
    )

    rows = select_semantic_context_transfer_rows(input_path=input_path, limit_dialogs=2)

    assert [row["dialog_id"] for row in rows] == ["dialog-1", "dialog-2"]
    assert {row["checkpoint"] for row in rows} == {10}
    assert {row["semantic_context_policy"] for row in rows} == {"normal_schema_context"}
    assert {row["rollout_target_history_policy"] for row in rows} == {
        "model_generated_sql_rollout"
    }


def test_select_semantic_context_transfer_rows_requires_enough_compatible_dialogs(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "clean.jsonl"
    _write_jsonl(input_path, [_dialog(0, history_policy="single_turn")])

    with pytest.raises(ValueError, match="requested 1"):
        select_semantic_context_transfer_rows(input_path=input_path, limit_dialogs=1)


def test_write_semantic_context_transfer_input_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "clean.jsonl"
    value_index_path, value_index_manifest = _write_value_index(tmp_path)
    normal_output = tmp_path / "processed" / "normal.jsonl"
    semantic_output = tmp_path / "processed" / "semantic.jsonl"
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    preflight_path = tmp_path / "preflight.json"
    _write_jsonl(input_path, [_dialog(0), _dialog(1)])

    manifest = write_semantic_context_transfer_input_artifacts(
        input_path=input_path,
        value_index_path=value_index_path,
        value_index_manifest_path=value_index_manifest,
        normal_output_path=normal_output,
        semantic_output_path=semantic_output,
        summary_path=summary_path,
        manifest_path=manifest_path,
        preflight_path=preflight_path,
        limit_dialogs=2,
        command=["build-cp10"],
    )

    normal_rows = [json.loads(line) for line in normal_output.read_text().splitlines()]
    semantic_rows = [json.loads(line) for line in semantic_output.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    written_manifest = json.loads(manifest_path.read_text())

    assert manifest == written_manifest
    assert manifest["artifact_type"] == "semantic_context_transfer_rollout_inputs"
    assert manifest["dialog_count"] == 2
    assert manifest["preflight_status"] == "ready_for_semantic_context_rollout_pair"
    assert manifest["normal_output_sha256"] == sha256_file(normal_output)
    assert manifest["semantic_output_sha256"] == sha256_file(semantic_output)
    assert manifest["preflight_sha256"] == sha256_file(preflight_path)
    assert [row["dialog_id"] for row in normal_rows] == ["dialog-0", "dialog-1"]
    assert [row["dialog_id"] for row in semantic_rows] == ["dialog-0", "dialog-1"]
    assert {row["semantic_context_policy"] for row in semantic_rows} == {
        "schema_context_plus_database_value_retrieval"
    }
    assert semantic_rows[0]["semantic_value_retrieval"]["index_source"] == (
        "database_contents"
    )
    assert summary["semantic_matched_turn_count"] == 4
    assert summary["semantic_matched_value_count"] == 4
    assert summary["split_row_ids"] == ["cosql_dev:0000:db", "cosql_dev:0001:db"]
    assert preflight["row_count"] == 4
    assert preflight["value_index_index_source"] == "database_contents"


def test_write_semantic_context_transfer_input_artifacts_requires_index_coverage(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "clean.jsonl"
    value_index_path, value_index_manifest = _write_value_index(tmp_path)
    missing_db = _dialog(0)
    missing_db["database_id"] = "missing"
    _write_jsonl(input_path, [missing_db])

    with pytest.raises(ValueError, match="missing selected database"):
        write_semantic_context_transfer_input_artifacts(
            input_path=input_path,
            value_index_path=value_index_path,
            value_index_manifest_path=value_index_manifest,
            normal_output_path=tmp_path / "normal.jsonl",
            semantic_output_path=tmp_path / "semantic.jsonl",
            summary_path=tmp_path / "summary.json",
            manifest_path=tmp_path / "manifest.json",
            preflight_path=tmp_path / "preflight.json",
            limit_dialogs=1,
        )
