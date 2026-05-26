from __future__ import annotations

import json
from pathlib import Path

from data.value_artifacts import (
    build_value_grounding_artifacts,
    build_value_grounding_manifest,
    extract_sql_value_references,
    run_value_grounding_export,
    summarize_value_grounding_artifacts,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_extract_sql_value_references_keeps_column_operator_and_type() -> None:
    refs = extract_sql_value_references(
        "SELECT first_name FROM Students AS s WHERE s.country = 'Haiti' AND age >= 21"
    )

    assert {
        (
            ref["table"],
            ref["column"],
            ref["operator"],
            ref["literal_value"],
            ref["literal_type"],
        )
        for ref in refs
    } == {
        ("students", "s.country", "=", "Haiti", "string"),
        ("students", "age", ">=", "21", "number"),
    }


def test_value_grounding_artifacts_separate_current_history_and_normalization() -> None:
    records = [
        {
            "database_id": "customers",
            "messages": [
                {"role": "system", "content": "sql"},
                {
                    "role": "user",
                    "content": "Question:\nShow revenue for France.",
                },
                {
                    "role": "assistant",
                    "content": "SELECT SUM(amount) FROM orders WHERE country = 'FR'",
                },
                {
                    "role": "user",
                    "content": "Question:\nNow only include Alice.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "SELECT SUM(amount) FROM orders "
                        "WHERE country = 'FR' AND customer = 'Alice'"
                    ),
                },
            ],
        }
    ]

    artifacts = build_value_grounding_artifacts(records)

    first_turn = [row for row in artifacts if row["turn_index"] == 0]
    second_turn = [row for row in artifacts if row["turn_index"] == 1]
    assert first_turn[0]["artifact_type"] == "value_grounding_label"
    assert first_turn[0]["schema_version"] == 1
    assert first_turn[0]["label_source"] == "gold_reference_sql"
    assert first_turn[0]["turn_id"] == "customers:0:0"
    assert first_turn[0]["resolved_table"] == "orders"
    assert first_turn[0]["resolved_column"] == "country"
    assert first_turn[0]["resolved_value"] == "FR"
    assert first_turn[0]["mention_text"] is None
    assert first_turn[0]["literal_value"] == "FR"
    assert first_turn[0]["mention_status"] == "missing_from_user_text"
    assert first_turn[0]["requires_value_normalization"] is True

    by_value = {row["literal_value"]: row for row in second_turn}
    assert by_value["FR"]["mention_status"] == "carried_from_prior_sql"
    assert by_value["FR"]["prior_turn_reference"] == "assistant_sql"
    assert by_value["FR"]["mention_text"] == "FR"
    assert by_value["FR"]["requires_context_carryover"] is True
    assert by_value["FR"]["requires_value_normalization"] is True
    assert by_value["Alice"]["mention_status"] == "exact_in_current_turn"
    assert by_value["Alice"]["prior_turn_reference"] is None
    assert by_value["Alice"]["mention_text"] == "Alice"
    assert by_value["Alice"]["requires_context_carryover"] is False


def test_summarize_value_grounding_artifacts_counts_failure_modes() -> None:
    summary = summarize_value_grounding_artifacts(
        [
            {"database_id": "db1", "literal_value": "FR", "mention_status": "missing_from_user_text"},
            {"database_id": "db1", "literal_value": "Alice", "mention_status": "exact_in_current_turn"},
            {"database_id": "db2", "literal_value": "FR", "mention_status": "carried_from_prior_sql"},
        ]
    )

    assert summary["value_reference_count"] == 3
    assert summary["database_count"] == 2
    assert summary["missing_from_user_text_count"] == 1
    assert summary["exact_in_current_turn_count"] == 1
    assert summary["carried_from_prior_sql_count"] == 1


def test_build_value_grounding_manifest_records_hashes_and_contract(tmp_path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "value_labels.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "database_id": "concert_singer",
                "messages": [
                    {"role": "system", "content": "sql"},
                    {"role": "user", "content": "Question:\nwhat is the capacity of Balmoor?"},
                    {"role": "assistant", "content": "select Capacity from stadium where name = 'Balmoor'"},
                ],
            }
        ],
    )
    rows = build_value_grounding_artifacts(
        [json.loads(line) for line in input_path.read_text().splitlines()]
    )
    _write_jsonl(output_path, rows)
    summary = summarize_value_grounding_artifacts(rows)

    manifest = build_value_grounding_manifest(
        input_path=input_path,
        output_path=output_path,
        summary=summary,
        rows=rows,
    )

    assert manifest["artifact_type"] == "value_entity_artifact_v1"
    assert manifest["schema_version"] == 1
    assert manifest["label_source"] == "gold_reference_sql"
    assert manifest["input_path"] == str(input_path)
    assert manifest["output_path"] == str(output_path)
    assert len(manifest["input_sha256"]) == 64
    assert len(manifest["output_sha256"]) == 64
    assert manifest["row_count"] == 1
    assert manifest["database_count"] == 1
    assert manifest["metrics"]["value_reference_count"] == 1


def test_run_value_grounding_export_writes_jsonl_summary_and_manifest(tmp_path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "value_labels.jsonl"
    summary_path = tmp_path / "value_labels_summary.json"
    manifest_path = tmp_path / "value_labels.manifest.json"
    _write_jsonl(
        input_path,
        [
            {
                "database_id": "concert_singer",
                "messages": [
                    {"role": "system", "content": "sql"},
                    {"role": "user", "content": "Question:\nwhat is the capacity of Balmoor?"},
                    {"role": "assistant", "content": "select Capacity from stadium where name = 'Balmoor'"},
                ],
            }
        ],
    )

    exit_code = run_value_grounding_export(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )

    assert exit_code == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert rows[0]["database_id"] == "concert_singer"
    assert rows[0]["literal_value"] == "Balmoor"
    assert rows[0]["mention_status"] == "exact_in_current_turn"
    assert summary["value_reference_count"] == 1
    assert manifest["row_count"] == 1
    assert manifest["metrics"]["exact_in_current_turn_count"] == 1
