from __future__ import annotations

import json
from pathlib import Path

from data.split_manifest import build_split_manifests, load_split_manifest, write_split_manifests
from eval.experiment_registry import load_experiment_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_cosql(path: Path, *, count: int, turns_per_dialog: int) -> None:
    rows = [
        {
            "database_id": f"db_{index % 3}",
            "interaction": [
                {"utterance": f"question {index}-{turn}", "query": "SELECT 1"}
                for turn in range(turns_per_dialog)
            ],
        }
        for index in range(count)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")


def _write_fixture_pack(path: Path) -> None:
    rows = [
        {
            "fixture_id": "fixture_a",
            "failure_modes": ["value_normalization"],
            "training_targets": ["semantic_layer"],
        },
        {
            "fixture_id": "fixture_b",
            "failure_modes": ["recovery"],
            "training_targets": ["behavior_recovery"],
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_split_manifest_builder_separates_proxy_and_holdout(tmp_path) -> None:
    cosql_root = tmp_path / "cosql"
    cosql_root.mkdir()
    _write_cosql(cosql_root / "cosql_train.json", count=3, turns_per_dialog=2)
    _write_cosql(cosql_root / "cosql_dev.json", count=105, turns_per_dialog=3)
    fixtures = tmp_path / "fixtures.jsonl"
    _write_fixture_pack(fixtures)

    manifests = {
        manifest["split_id"]: manifest
        for manifest in build_split_manifests(cosql_root=cosql_root, synthetic_fixtures=fixtures)
    }

    proxy = manifests["cosql_dev_100_proxy_seen_v1"]
    holdout = manifests["cosql_dev_clean_holdout_v1"]

    assert proxy["role"] == "proxy_dev_seen"
    assert holdout["role"] == "clean_local_holdout"
    assert len(proxy["row_ids"]) == 100
    assert len(holdout["row_ids"]) == 5
    assert set(proxy["row_ids"]).isdisjoint(holdout["row_ids"])
    assert proxy["turn_count"] == 300
    assert holdout["turn_count"] == 15


def test_checked_in_split_manifests_have_valid_roles_and_hashes() -> None:
    split_dir = REPO_ROOT / "data" / "splits"
    manifests = [load_split_manifest(path) for path in sorted(split_dir.glob("*.json"))]

    split_ids = {manifest["split_id"] for manifest in manifests}
    assert {
        "cosql_train_v1",
        "cosql_dev_100_proxy_seen_v1",
        "cosql_dev_clean_holdout_v1",
        "synthetic_method_fixtures_v1",
        "sparc_context_transfer_pending_v1",
        "bird_mini_dev_pending_v1",
        "bird_interact_lite_pending_v1",
    } <= split_ids

    proxy = next(row for row in manifests if row["split_id"] == "cosql_dev_100_proxy_seen_v1")
    holdout = next(row for row in manifests if row["split_id"] == "cosql_dev_clean_holdout_v1")
    assert set(proxy["row_ids"]).isdisjoint(holdout["row_ids"])
    assert holdout["status"] == "ready"
    assert "Reserved for final local method comparisons" in holdout["leakage_boundary"]


def test_experiment_registry_split_ids_exist() -> None:
    split_ids = {
        load_split_manifest(path)["split_id"]
        for path in sorted((REPO_ROOT / "data" / "splits").glob("*.json"))
    }
    experiments = load_experiment_registry(REPO_ROOT / "configs" / "experiments.yaml")

    for experiment in experiments:
        assert experiment["train_split_id"] in split_ids
        assert experiment["validation_split_id"] in split_ids
        assert experiment["test_split_id"] in split_ids


def test_split_manifest_writer_round_trips(tmp_path) -> None:
    cosql_root = tmp_path / "cosql"
    cosql_root.mkdir()
    _write_cosql(cosql_root / "cosql_train.json", count=2, turns_per_dialog=1)
    _write_cosql(cosql_root / "cosql_dev.json", count=101, turns_per_dialog=1)
    fixtures = tmp_path / "fixtures.jsonl"
    _write_fixture_pack(fixtures)

    written = write_split_manifests(
        output_dir=tmp_path / "splits",
        cosql_root=cosql_root,
        synthetic_fixtures=fixtures,
    )

    assert len(written) == 7
    for path in written:
        assert load_split_manifest(path)["split_id"] == path.stem
