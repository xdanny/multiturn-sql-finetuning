from __future__ import annotations

import json
from pathlib import Path

from data.bird_interact_transfer_dataset import (
    DEFAULT_MANIFEST_OUTPUT,
    DEFAULT_SUMMARY_OUTPUT,
    build_bird_interact_transfer_rows,
    summarize_bird_interact_transfer_rows,
    write_bird_interact_transfer_artifacts,
)


def _prepared_row() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a SQL expert."},
            {
                "role": "user",
                "content": "Database: foo\n\nSchema/context:\nfoo(id int)\n\nQuestion:\nList rows.",
            },
            {"role": "assistant", "content": "SELECT * FROM foo;"},
            {"role": "user", "content": "Question:\nOnly active ones."},
            {"role": "assistant", "content": "SELECT * FROM foo WHERE active = 1;"},
        ],
        "source": "unit",
        "database_id": "db1",
        "dialog_id": "dialog-a",
        "assistant_turn_count": 2,
        "turn_format": "multi_turn_dialog",
        "history_policy": "gold_sql_teacher_forced",
        "evaluation_mode": "non_oracle_generation",
        "gold_plans": [{}, {}],
        "schema_link_labels": [{}, {}],
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
    }


def test_build_bird_interact_transfer_rows_preserve_non_oracle_prepared_contract() -> None:
    rows = build_bird_interact_transfer_rows([_prepared_row()])

    assert rows[0]["benchmark"] == "bird_interact_transfer"
    assert rows[0]["training_target"] == "bird_interact_candidate"
    assert rows[0]["evaluation_mode"] == "non_oracle_generation"
    assert rows[0]["history_policy"] == "gold_sql_teacher_forced"


def test_summarize_bird_interact_transfer_rows_reports_fixed_slice_counts() -> None:
    summary = summarize_bird_interact_transfer_rows(build_bird_interact_transfer_rows([_prepared_row()]))

    assert summary["artifact_type"] == "bird_interact_transfer_dataset"
    assert summary["benchmark"] == "bird_interact_transfer"
    assert summary["row_count"] == 1
    assert summary["dialog_count"] == 1
    assert summary["database_count"] == 1


def test_write_bird_interact_transfer_artifacts_writes_rows_summary_and_manifest(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "bird_interact_rows.jsonl"
    summary_path = tmp_path / "bird_interact_summary.json"
    manifest_path = tmp_path / "bird_interact.manifest.json"
    input_path.write_text(json.dumps(_prepared_row()) + "\n")

    manifest = write_bird_interact_transfer_artifacts(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        command=["uv", "run", "python", "-m", "data.bird_interact_transfer_dataset"],
    )

    written_summary = json.loads(summary_path.read_text())
    written_manifest = json.loads(manifest_path.read_text())

    assert written_summary["row_count"] == 1
    assert manifest["benchmark"] == "bird_interact_transfer"
    assert manifest["evaluation_mode"] == "non_oracle_generation"
    assert written_manifest == manifest


def test_checked_in_bird_interact_transfer_manifest_exists() -> None:
    payload = json.loads(DEFAULT_MANIFEST_OUTPUT.read_text())
    summary = json.loads(DEFAULT_SUMMARY_OUTPUT.read_text())
    assert payload["benchmark"] == "bird_interact_transfer"
    assert payload["evaluation_mode"] == "non_oracle_generation"
    assert summary["benchmark"] == "bird_interact_transfer"
