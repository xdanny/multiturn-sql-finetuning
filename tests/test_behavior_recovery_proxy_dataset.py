from __future__ import annotations

import json
from pathlib import Path

from data.behavior_recovery_proxy_dataset import (
    DEFAULT_MANIFEST_OUTPUT,
    DEFAULT_SUMMARY_OUTPUT,
    build_behavior_recovery_proxy_eval_rows,
    build_behavior_recovery_proxy_train_rows,
    summarize_behavior_recovery_proxy_artifacts,
    write_behavior_recovery_proxy_artifacts,
)


def _train_row() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Use semantic model hints."},
            {
                "role": "user",
                "content": "Schema/context:\nfoo(id int)\n\nSemantic model:\n- Cube foo\n\nQuestion:\nList rows.",
            },
            {"role": "assistant", "content": "SELECT * FROM foo;"},
            {"role": "user", "content": "Question:\nNow filter to active rows."},
            {"role": "assistant", "content": "SELECT * FROM foo WHERE active = 1;"},
        ],
        "source": "unit",
        "database_id": "db1",
    }


def _eval_row() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Use semantic model hints."},
            {
                "role": "user",
                "content": "Schema/context:\nfoo(id int)\n\nSemantic model:\n- Cube foo\n\nQuestion:\nList rows.",
            },
            {"role": "assistant", "content": "SELECT * FROM foo;"},
            {"role": "user", "content": "Question:\nNow filter to active rows."},
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


def test_build_behavior_recovery_proxy_rows_preserve_non_oracle_semantic_context() -> None:
    train_rows = build_behavior_recovery_proxy_train_rows([_train_row()])
    eval_rows = build_behavior_recovery_proxy_eval_rows([_eval_row()])

    assert train_rows[0]["training_target"] == "behavior_recovery"
    assert train_rows[0]["benchmark"] == "prepared"
    assert train_rows[0]["evaluation_mode"] == "non_oracle_generation"
    assert "Semantic model:" in train_rows[0]["messages"][1]["content"]
    assert eval_rows[0]["training_target"] == "behavior_recovery"
    assert eval_rows[0]["benchmark"] == "prepared"
    assert eval_rows[0]["history_policy"] == "gold_sql_teacher_forced"
    assert eval_rows[0]["uses_oracle_planning_hints"] is False


def test_summarize_behavior_recovery_proxy_artifacts_reports_prepared_proxy_counts() -> None:
    summary = summarize_behavior_recovery_proxy_artifacts(
        train_rows=build_behavior_recovery_proxy_train_rows([_train_row(), _train_row()]),
        eval_rows=build_behavior_recovery_proxy_eval_rows([_eval_row()]),
    )

    assert summary["artifact_type"] == "behavior_recovery_proxy_dataset"
    assert summary["training_target"] == "behavior_recovery"
    assert summary["benchmark"] == "prepared"
    assert summary["train_row_count"] == 2
    assert summary["eval_row_count"] == 1
    assert summary["eval_dialog_count"] == 1
    assert summary["eval_multi_turn_count"] == 1


def test_write_behavior_recovery_proxy_artifacts_writes_train_eval_summary_and_manifest(
    tmp_path: Path,
) -> None:
    train_input = tmp_path / "train_behavior_recovery_proxy.jsonl"
    eval_input = tmp_path / "eval_behavior_recovery_proxy.jsonl"
    train_output = tmp_path / "behavior_recovery_proxy_train.jsonl"
    eval_output = tmp_path / "behavior_recovery_proxy_eval.jsonl"
    summary_output = tmp_path / "behavior_recovery_proxy_summary.json"
    manifest_output = tmp_path / "behavior_recovery_proxy.manifest.json"
    train_input.write_text(json.dumps(_train_row()) + "\n")
    eval_input.write_text(json.dumps(_eval_row()) + "\n")

    manifest = write_behavior_recovery_proxy_artifacts(
        train_input_path=train_input,
        eval_input_path=eval_input,
        train_output_path=train_output,
        eval_output_path=eval_output,
        summary_path=summary_output,
        manifest_path=manifest_output,
        command=["uv", "run", "python", "-m", "data.behavior_recovery_proxy_dataset"],
    )

    written_summary = json.loads(summary_output.read_text())
    written_manifest = json.loads(manifest_output.read_text())

    assert written_summary["train_row_count"] == 1
    assert written_summary["eval_row_count"] == 1
    assert manifest["training_target"] == "behavior_recovery"
    assert manifest["benchmark"] == "prepared"
    assert written_manifest == manifest


def test_checked_in_behavior_recovery_proxy_manifest_exists() -> None:
    payload = json.loads(DEFAULT_MANIFEST_OUTPUT.read_text())
    summary = json.loads(DEFAULT_SUMMARY_OUTPUT.read_text())
    assert payload["training_target"] == "behavior_recovery"
    assert payload["benchmark"] == "prepared"
    assert summary["benchmark"] == "prepared"
