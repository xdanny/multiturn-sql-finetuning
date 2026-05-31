from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.prepare_split import prepare_records_from_split, write_prepared_split
from data.split_manifest import build_split_manifests


def _write_cosql(path: Path, *, count: int, turns_per_dialog: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "database_id": f"db_{index % 3}",
            "interaction": [
                {
                    "utterance": f"question {index}-{turn}",
                    "query": f"SELECT {index + turn}",
                }
                for turn in range(turns_per_dialog)
            ],
        }
        for index in range(count)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_fixture_pack(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"fixture_id": "fixture_a", "failure_modes": ["schema"], "training_targets": ["planner"]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _write_spider_tables(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "db_id": "db_0",
            "table_names_original": ["orders", "customers"],
            "column_names_original": [
                [-1, "*"],
                [0, "id"],
                [0, "customer_id"],
                [0, "amount"],
                [1, "id"],
                [1, "name"],
            ],
            "column_types": ["text", "number", "number", "number", "number", "text"],
            "primary_keys": [1, 4],
            "foreign_keys": [[2, 4]],
        }
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_split_manifests(tmp_path: Path) -> dict[str, Path]:
    cosql_root = tmp_path / "data" / "raw" / "cosql_dataset" / "sql_state_tracking"
    _write_cosql(cosql_root / "cosql_train.json", count=4, turns_per_dialog=2)
    _write_cosql(cosql_root / "cosql_dev.json", count=105, turns_per_dialog=2)
    fixtures = tmp_path / "docs" / "data_artifacts" / "synthetic_method_fixtures.jsonl"
    _write_fixture_pack(fixtures)

    split_dir = tmp_path / "data" / "splits"
    paths = {}
    for manifest in build_split_manifests(cosql_root=cosql_root, synthetic_fixtures=fixtures):
        path = split_dir / f"{manifest['split_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths[manifest["split_id"]] = path
    return paths


def test_prepare_records_from_split_selects_clean_holdout_range(tmp_path) -> None:
    split_paths = _write_split_manifests(tmp_path)

    records, metadata = prepare_records_from_split(
        split_paths["cosql_dev_clean_holdout_v1"],
        source_roots=[tmp_path],
    )

    assert len(records) == 5
    assert records[0]["split_id"] == "cosql_dev_clean_holdout_v1"
    assert records[0]["split_role"] == "clean_local_holdout"
    assert records[0]["split_row_id"].startswith("cosql_dev:0100:")
    assert "question 100-0" in records[0]["messages"][1]["content"]
    assert metadata["selected_row_ids"][0].startswith("cosql_dev:0100:")
    assert {record["evaluation_mode"] for record in records} == {"non_oracle_generation"}
    assert not any(record["uses_oracle_planning_hints"] for record in records)
    assert not any(record["semantic_context_pruned_by_oracle_labels"] for record in records)
    user_text = "\n".join(
        message["content"]
        for record in records
        for message in record["messages"]
        if message["role"] == "user"
    )
    assert "Oracle SQL planning hints" not in user_text
    assert "derived from reference SQL" not in user_text


def test_prepare_records_from_split_resolves_tables_path_from_source_root(tmp_path) -> None:
    split_paths = _write_split_manifests(tmp_path)
    relative_tables = Path("fixtures/source_root/tables.json")
    _write_spider_tables(tmp_path / relative_tables)

    records, _ = prepare_records_from_split(
        split_paths["cosql_train_v1"],
        tables_path=relative_tables,
        source_roots=[tmp_path],
        limit=1,
    )

    first_user_message = records[0]["messages"][1]["content"]
    assert "Schema/context:" in first_user_message
    assert "orders(id number, customer_id number, amount number)" in first_user_message
    assert "Semantic model:" in first_user_message
    assert "orders.customer_id -> customers.id" in first_user_message
    assert records[0]["schema_link_labels"][0]["relevant_tables"] == []
    assert not records[0]["uses_oracle_planning_hints"]


def test_prepare_records_from_proxy_and_holdout_are_disjoint(tmp_path) -> None:
    split_paths = _write_split_manifests(tmp_path)

    proxy_records, _ = prepare_records_from_split(
        split_paths["cosql_dev_100_proxy_seen_v1"],
        source_roots=[tmp_path],
        limit=3,
    )
    holdout_records, _ = prepare_records_from_split(
        split_paths["cosql_dev_clean_holdout_v1"],
        source_roots=[tmp_path],
    )

    proxy_ids = {record["split_row_id"] for record in proxy_records}
    holdout_ids = {record["split_row_id"] for record in holdout_records}
    assert proxy_ids.isdisjoint(holdout_ids)


def test_write_prepared_split_records_manifest_provenance(tmp_path) -> None:
    split_paths = _write_split_manifests(tmp_path)
    output = tmp_path / "data" / "processed" / "cosql_train_v1.jsonl"
    manifest_output = tmp_path / "data" / "processed" / "cosql_train_v1.manifest.json"

    manifest = write_prepared_split(
        split_manifest_path=split_paths["cosql_train_v1"],
        output_path=output,
        manifest_output_path=manifest_output,
        source_roots=[tmp_path],
        command=["uv", "run", "--active", "--no-sync", "python", "-m", "data.prepare_split"],
    )

    written = json.loads(manifest_output.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert manifest == written
    assert manifest["artifact_type"] == "prepared_split_dataset"
    assert manifest["split_id"] == "cosql_train_v1"
    assert manifest["split_role"] == "train"
    assert manifest["row_count"] == 4
    assert manifest["assistant_turn_count"] == 8
    assert manifest["evaluation_modes"] == {"non_oracle_generation": 4}
    assert manifest["uses_oracle_planning_hints"] is False
    assert manifest["semantic_context_pruned_by_oracle_labels"] is False
    assert manifest["output_sha256"]
    assert records[0]["split_source_path"].startswith("data/raw/")


def test_prepare_records_from_split_rejects_pending_or_non_cosql_manifest(tmp_path) -> None:
    split_paths = _write_split_manifests(tmp_path)

    with pytest.raises(ValueError, match="only ready split manifests"):
        prepare_records_from_split(split_paths["bird_mini_dev_pending_v1"], source_roots=[tmp_path])

    with pytest.raises(ValueError, match="only CoSQL split preparation"):
        prepare_records_from_split(split_paths["synthetic_method_fixtures_v1"], source_roots=[tmp_path])
