from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_alias_column_context_comparison import (
    validate_comparison_inputs,
    write_comparison_preflight,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _record(*, content_suffix: str = "") -> dict:
    return {
        "id": "dialog-a",
        "database_id": "store",
        "source": "unit",
        "evaluation_mode": "non_oracle_generation",
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
        "gold_plans": [{"parseable": True}],
        "messages": [
            {"role": "system", "content": "sql"},
            {"role": "user", "content": f"Show revenue.{content_suffix}"},
            {"role": "assistant", "content": "SELECT 1"},
        ],
    }


def test_validate_comparison_inputs_requires_same_row_identity(tmp_path) -> None:
    direct = tmp_path / "direct.jsonl"
    alias = tmp_path / "alias.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(direct, [_record()])
    _write_jsonl(alias, [_record(content_suffix="\nColumn-role constraints")])
    manifest.write_text(json.dumps({"artifact_type": "alias_column_context"}))

    summary = validate_comparison_inputs(
        direct_input=direct,
        alias_input=alias,
        alias_context_manifest=manifest,
        limit=None,
    )

    assert summary["row_count"] == 1
    assert summary["dialog_count"] == 1
    assert summary["direct_input_sha256"] != summary["alias_input_sha256"]


def test_validate_comparison_inputs_rejects_mismatch(tmp_path) -> None:
    direct = tmp_path / "direct.jsonl"
    alias = tmp_path / "alias.jsonl"
    manifest = tmp_path / "manifest.json"
    direct_record = _record()
    alias_record = _record(content_suffix="\nColumn-role constraints")
    alias_record["messages"][-1]["content"] = "SELECT 2"
    _write_jsonl(direct, [direct_record])
    _write_jsonl(alias, [alias_record])
    manifest.write_text(json.dumps({"artifact_type": "alias_column_context"}))

    with pytest.raises(ValueError, match="identity mismatch"):
        validate_comparison_inputs(
            direct_input=direct,
            alias_input=alias,
            alias_context_manifest=manifest,
            limit=None,
        )


def test_write_comparison_preflight(tmp_path) -> None:
    direct = tmp_path / "direct.jsonl"
    alias = tmp_path / "alias.jsonl"
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "preflight.json"
    _write_jsonl(direct, [_record()])
    _write_jsonl(alias, [_record(content_suffix="\nColumn-role constraints")])
    manifest.write_text(json.dumps({"artifact_type": "alias_column_context"}))

    payload = write_comparison_preflight(
        direct_input=direct,
        alias_input=alias,
        alias_context_manifest=manifest,
        output_path=output,
        limit=None,
    )

    assert payload["status"] == "ready_for_generation_pair"
    assert json.loads(output.read_text())["artifact_type"] == (
        "alias_column_context_comparison_preflight"
    )
