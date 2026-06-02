from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.semantic_value_retrieval_inputs import (
    add_semantic_value_retrieval_context,
    build_semantic_value_retrieval_inputs,
    retrieve_value_matches,
    write_semantic_value_retrieval_input_artifacts,
)
from eval.result_manifest import sha256_file


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _prepared_record() -> dict:
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
            {"role": "user", "content": "Show French customers."},
            {"role": "assistant", "content": "SELECT * FROM customers WHERE country = 'FR'"},
            {"role": "user", "content": "Only Alice this time."},
            {"role": "assistant", "content": "SELECT * FROM customers WHERE name = 'Alice'"},
        ],
    }


def _value_index_rows() -> list[dict]:
    return [
        {
            "artifact_type": "non_oracle_value_index_entry",
            "schema_version": 1,
            "index_source": "database_contents",
            "database_id": "store",
            "table": "customers",
            "column": "country_code",
            "raw_value": "FR",
            "normalized_value": "fr",
            "aliases": ["France", "French"],
            "source_frequency": 4,
        },
        {
            "artifact_type": "non_oracle_value_index_entry",
            "schema_version": 1,
            "index_source": "database_contents",
            "database_id": "store",
            "table": "customers",
            "column": "customer_name",
            "raw_value": "Alice Smith",
            "normalized_value": "alice smith",
            "aliases": ["Alice Smith", "Alice"],
            "source_frequency": 1,
        },
    ]


def test_retrieve_value_matches_prunes_short_ambiguous_aliases() -> None:
    rows = [
        {
            "artifact_type": "non_oracle_value_index_entry",
            "schema_version": 1,
            "index_source": "database_contents",
            "database_id": "store",
            "table": "countries",
            "column": "code2",
            "raw_value": "IS",
            "normalized_value": "is",
            "aliases": ["IS"],
            "source_frequency": 3,
        },
        *_value_index_rows(),
    ]

    matches = retrieve_value_matches(
        text="What country is Alice from?",
        value_index_rows=rows,
    )

    assert [match["raw_value"] for match in matches] == ["Alice Smith"]


def test_retrieve_value_matches_uses_alias_boundaries() -> None:
    matches = retrieve_value_matches(
        text="Only Alice this time, not malice.",
        value_index_rows=_value_index_rows(),
    )

    assert [match["raw_value"] for match in matches] == ["Alice Smith"]
    assert matches[0]["matched_alias"] == "Alice"


def test_add_semantic_value_retrieval_context_uses_user_text_not_assistant_sql() -> None:
    record = _prepared_record()
    # FR appears only in the assistant reference SQL, while French appears in user text.
    record["messages"][1]["content"] = "Show customers from Europe."

    updated, summary = add_semantic_value_retrieval_context(
        record,
        value_index_by_database={"store": _value_index_rows()},
    )

    first_user = updated["messages"][1]["content"]
    second_user = updated["messages"][3]["content"]
    assert "country_code" not in first_user
    assert "customer_name" in second_user
    assert summary["matched_turn_count"] == 1
    assert updated["semantic_value_retrieval"]["leakage_boundary"] == (
        "matches use user-authored text up to each turn only"
    )


def test_build_semantic_value_retrieval_inputs_summarizes_scope() -> None:
    rows, summary = build_semantic_value_retrieval_inputs(
        prepared_rows=[_prepared_record()],
        value_index_rows=_value_index_rows(),
        max_matches_per_turn=3,
    )

    assert rows[0]["evaluation_mode"] == "non_oracle_generation"
    assert "Database value retrieval" in rows[0]["messages"][1]["content"]
    assert summary["row_count"] == 1
    assert summary["assistant_turn_count"] == 2
    assert summary["matched_turn_count"] == 2
    assert summary["matched_value_count"] == 2
    assert summary["max_matches_per_turn"] == 3
    assert summary["retrieval_scope"] == "current_turn"
    assert summary["min_alias_chars"] == 3
    assert "reference SQL" in summary["leakage_boundary"]
    assert summary["evaluation_gate"] == "eval.run_semantic_value_retrieval_comparison"


def test_build_semantic_value_retrieval_inputs_can_use_history_scope() -> None:
    _rows, summary = build_semantic_value_retrieval_inputs(
        prepared_rows=[_prepared_record()],
        value_index_rows=_value_index_rows(),
        max_matches_per_turn=3,
        retrieval_scope="history",
    )

    assert summary["matched_value_count"] == 3
    assert summary["retrieval_scope"] == "history"


def test_build_semantic_value_retrieval_inputs_rejects_oracle_record() -> None:
    record = _prepared_record()
    record["semantic_context_pruned_by_oracle_labels"] = True

    with pytest.raises(ValueError, match="non-oracle"):
        build_semantic_value_retrieval_inputs(
            prepared_rows=[record],
            value_index_rows=_value_index_rows(),
        )


def test_write_semantic_value_retrieval_input_artifacts(tmp_path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    value_index_path = tmp_path / "value_index.jsonl"
    value_index_manifest_path = tmp_path / "value_index.manifest.json"
    output_path = tmp_path / "semantic.jsonl"
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "manifest.json"
    _write_jsonl(input_path, [_prepared_record()])
    _write_jsonl(value_index_path, _value_index_rows())
    value_index_manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "non_oracle_value_index_v1",
                "index_source": "database_contents",
                "output_path": str(value_index_path),
                "output_sha256": sha256_file(value_index_path),
            }
        )
    )

    manifest = write_semantic_value_retrieval_input_artifacts(
        input_path=input_path,
        value_index_path=value_index_path,
        value_index_manifest_path=value_index_manifest_path,
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        command=["unit"],
    )

    output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert output_rows[0]["semantic_value_retrieval"]["index_source"] == "database_contents"
    assert summary["oracle_policy"] == "non_oracle_database_value_index_matched_to_user_text_only"
    assert manifest["max_matches_per_turn"] == 4
    assert manifest["retrieval_scope"] == "current_turn"
    assert manifest["min_alias_chars"] == 3
    assert manifest["output_sha256"] == sha256_file(output_path)
    assert manifest["value_index_manifest_sha256"] == sha256_file(value_index_manifest_path)
