from __future__ import annotations

import json
from pathlib import Path

from data.semantic_proxy_direct_sql_dataset import (
    DEFAULT_MANIFEST_OUTPUT,
    DEFAULT_SUMMARY_OUTPUT,
    build_semantic_proxy_direct_eval_rows,
    build_semantic_proxy_direct_train_rows,
    summarize_semantic_proxy_direct_artifacts,
    write_semantic_proxy_direct_artifacts,
)


def _train_row() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Given database context."},
            {"role": "user", "content": "Schema/context:\nfoo(id int)\n\nQuestion:\nList rows."},
            {"role": "assistant", "content": "SELECT * FROM foo;"},
        ],
        "source": "unit",
        "database_id": "db1",
    }


def _semantic_eval_row() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Use semantic model hints."},
            {
                "role": "user",
                "content": "Schema/context:\nfoo(id int)\n\nSemantic model:\n- Cube foo\n\nQuestion:\nList rows.",
            },
            {"role": "assistant", "content": "SELECT * FROM foo;"},
        ],
        "source": "unit",
        "database_id": "db1",
        "dialog_id": "dialog-a",
        "evaluation_mode": "non_oracle_generation",
        "gold_plans": [{}],
        "schema_link_labels": [{}],
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
    }


def test_build_semantic_proxy_direct_rows_strip_semantic_context_from_eval_prompt() -> None:
    train_rows = build_semantic_proxy_direct_train_rows([_train_row()])
    eval_rows = build_semantic_proxy_direct_eval_rows([_semantic_eval_row()])

    assert train_rows[0]["training_target"] == "direct_sql_control"
    assert train_rows[0]["benchmark"] == "cosql_semantic_proxy_direct_sql"
    assert eval_rows[0]["training_target"] == "direct_sql_control"
    assert eval_rows[0]["benchmark"] == "cosql_semantic_proxy_direct_sql"
    assert "Semantic model:" not in eval_rows[0]["messages"][1]["content"]
    assert "Question:\nList rows." in eval_rows[0]["messages"][1]["content"]


def test_summarize_semantic_proxy_direct_artifacts_reports_stage3_control() -> None:
    summary = summarize_semantic_proxy_direct_artifacts(
        train_rows=build_semantic_proxy_direct_train_rows([_train_row(), _train_row()]),
        eval_rows=build_semantic_proxy_direct_eval_rows([_semantic_eval_row()]),
    )

    assert summary["artifact_type"] == "semantic_proxy_direct_sql_dataset"
    assert summary["training_target"] == "direct_sql_control"
    assert summary["benchmark"] == "cosql_semantic_proxy_direct_sql"
    assert summary["train_row_count"] == 2
    assert summary["eval_row_count"] == 1


def test_write_semantic_proxy_direct_artifacts_writes_train_eval_summary_and_manifest(
    tmp_path: Path,
) -> None:
    train_input = tmp_path / "train_direct.jsonl"
    eval_input = tmp_path / "eval_semantic.jsonl"
    train_output = tmp_path / "direct_proxy_train.jsonl"
    eval_output = tmp_path / "direct_proxy_eval.jsonl"
    summary_output = tmp_path / "direct_proxy_summary.json"
    manifest_output = tmp_path / "direct_proxy.manifest.json"
    train_input.write_text(json.dumps(_train_row()) + "\n")
    eval_input.write_text(json.dumps(_semantic_eval_row()) + "\n")

    manifest = write_semantic_proxy_direct_artifacts(
        train_input_path=train_input,
        eval_input_path=eval_input,
        train_output_path=train_output,
        eval_output_path=eval_output,
        summary_path=summary_output,
        manifest_path=manifest_output,
        command=["uv", "run", "python", "-m", "data.semantic_proxy_direct_sql_dataset"],
    )

    written_summary = json.loads(summary_output.read_text())
    written_manifest = json.loads(manifest_output.read_text())

    assert written_summary["train_row_count"] == 1
    assert written_summary["eval_row_count"] == 1
    assert manifest["training_target"] == "direct_sql_control"
    assert manifest["benchmark"] == "cosql_semantic_proxy_direct_sql"
    assert written_manifest == manifest


def test_checked_in_semantic_proxy_direct_manifest_exists() -> None:
    payload = json.loads(DEFAULT_MANIFEST_OUTPUT.read_text())
    summary = json.loads(DEFAULT_SUMMARY_OUTPUT.read_text())
    assert payload["training_target"] == "direct_sql_control"
    assert payload["benchmark"] == "cosql_semantic_proxy_direct_sql"
    assert summary["benchmark"] == "cosql_semantic_proxy_direct_sql"
