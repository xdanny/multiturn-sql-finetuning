"""Build and validate dataset split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_ROLES = {
    "train",
    "validation",
    "proxy_dev_seen",
    "clean_local_holdout",
    "external_target",
}

DEFAULT_COSQL_ROOT = Path("data/raw/cosql_dataset/sql_state_tracking")
DEFAULT_SYNTHETIC_FIXTURES = Path("docs/data_artifacts/synthetic_method_fixtures.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/splits")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(encoded)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _portable_source_path(path: Path) -> str:
    parts = path.parts
    for index in range(len(parts) - 1):
        if parts[index : index + 2] in (("data", "raw"), ("docs", "data_artifacts")):
            return str(Path(*parts[index:]))
    return str(path)


def _cosql_dialog_id(split_name: str, index: int, database_id: str) -> str:
    return f"cosql_{split_name}:{index:04d}:{database_id}"


def _cosql_turn_id(split_name: str, dialog_index: int, turn_index: int, database_id: str) -> str:
    return f"cosql_{split_name}:{dialog_index:04d}:{turn_index:02d}:{database_id}"


def _cosql_manifest(
    *,
    split_id: str,
    role: str,
    source_path: Path,
    source_split: str,
    selection_start: int,
    selection_end: int,
    status: str = "ready",
    leakage_boundary: str,
    notes: str,
) -> dict[str, Any]:
    records = _read_json(source_path)
    selected = records[selection_start:selection_end]
    dialog_ids = [
        _cosql_dialog_id(source_split, selection_start + offset, str(row["database_id"]))
        for offset, row in enumerate(selected)
    ]
    turn_ids = [
        _cosql_turn_id(source_split, selection_start + dialog_offset, turn_index, str(row["database_id"]))
        for dialog_offset, row in enumerate(selected)
        for turn_index, _turn in enumerate(row.get("interaction") or ())
    ]
    database_counts = Counter(str(row["database_id"]) for row in selected)
    return {
        "schema_version": 1,
        "split_id": split_id,
        "dataset": "CoSQL",
        "role": role,
        "status": status,
        "source_path": _portable_source_path(source_path),
        "source_sha256": _sha256_bytes(source_path.read_bytes()),
        "source_split": source_split,
        "selection": {
            "type": "dialog_index_range",
            "start_inclusive": selection_start,
            "end_exclusive": selection_end,
        },
        "row_id_policy": "cosql_<split>:<dialog_index>:<database_id>",
        "turn_id_policy": "cosql_<split>:<dialog_index>:<turn_index>:<database_id>",
        "row_ids": dialog_ids,
        "row_ids_sha256": _sha256_json(dialog_ids),
        "turn_count": len(turn_ids),
        "turn_ids_sha256": _sha256_json(turn_ids),
        "database_counts": dict(sorted(database_counts.items())),
        "leakage_boundary": leakage_boundary,
        "notes": notes,
    }


def _synthetic_manifest(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    fixture_ids = [str(row["fixture_id"]) for row in rows]
    failure_modes = sorted({mode for row in rows for mode in row.get("failure_modes", [])})
    training_targets = sorted({target for row in rows for target in row.get("training_targets", [])})
    return {
        "schema_version": 1,
        "split_id": "synthetic_method_fixtures_v1",
        "dataset": "synthetic_method_fixtures",
        "role": "validation",
        "status": "ready",
        "source_path": _portable_source_path(path),
        "source_sha256": _sha256_bytes(path.read_bytes()),
        "row_id_policy": "fixture_id",
        "row_ids": fixture_ids,
        "row_ids_sha256": _sha256_json(fixture_ids),
        "row_count": len(fixture_ids),
        "failure_modes": failure_modes,
        "training_targets": training_targets,
        "leakage_boundary": (
            "Reference SQL, expected rows, gold metric DSL, and repair labels are scorer-side "
            "unless a run is explicitly diagnostic."
        ),
        "notes": "Small synthetic validation surface for isolated method failures; not a benchmark.",
    }


def _pending_manifest(
    *,
    split_id: str,
    dataset: str,
    role: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "split_id": split_id,
        "dataset": dataset,
        "role": role,
        "status": "pending_external_data",
        "row_id_policy": "pending",
        "row_ids": [],
        "row_ids_sha256": _sha256_json([]),
        "leakage_boundary": (
            "Do not train on evaluation gold SQL, hidden answers, expected rows, or future turns; "
            "keep scorer-only fields outside production prompts."
        ),
        "notes": notes,
    }


def build_split_manifests(
    *,
    cosql_root: Path = DEFAULT_COSQL_ROOT,
    synthetic_fixtures: Path = DEFAULT_SYNTHETIC_FIXTURES,
) -> tuple[dict[str, Any], ...]:
    """Return the current roadmap split manifests."""

    cosql_train = cosql_root / "cosql_train.json"
    cosql_dev = cosql_root / "cosql_dev.json"
    dev_records = _read_json(cosql_dev)
    train_records = _read_json(cosql_train)
    return (
        _cosql_manifest(
            split_id="cosql_train_v1",
            role="train",
            source_path=cosql_train,
            source_split="train",
            selection_start=0,
            selection_end=len(train_records),
            leakage_boundary="Allowed for supervised training and label extraction.",
            notes="Full local CoSQL train split from the raw archive.",
        ),
        _cosql_manifest(
            split_id="cosql_dev_100_proxy_seen_v1",
            role="proxy_dev_seen",
            source_path=cosql_dev,
            source_split="dev",
            selection_start=0,
            selection_end=100,
            leakage_boundary=(
                "Inspected continuity proxy. Do not use for clean benchmark claims or final "
                "method selection."
            ),
            notes="The historical fixed 100-dialog CoSQL dev proxy slice.",
        ),
        _cosql_manifest(
            split_id="cosql_dev_clean_holdout_v1",
            role="clean_local_holdout",
            source_path=cosql_dev,
            source_split="dev",
            selection_start=100,
            selection_end=len(dev_records),
            leakage_boundary=(
                "Reserved for final local method comparisons. Do not use for prompt search, "
                "manual diagnostics, training, or evidence iteration."
            ),
            notes="Held-out remainder of CoSQL dev after the inspected 100-dialog proxy slice.",
        ),
        _synthetic_manifest(synthetic_fixtures),
        _pending_manifest(
            split_id="sparc_context_transfer_pending_v1",
            dataset="SParC",
            role="validation",
            notes="Pending a full-dialog SParC source with frozen dialog and turn ids.",
        ),
        _pending_manifest(
            split_id="bird_mini_dev_pending_v1",
            dataset="BIRD mini-dev",
            role="external_target",
            notes="Pending allowed BIRD-style evaluation rows and scorer-side gold policy.",
        ),
        _pending_manifest(
            split_id="bird_interact_lite_pending_v1",
            dataset="BIRD-Interact Lite or LiveSQLBench",
            role="external_target",
            notes="Pending target interactive benchmark rows after the local loop is stable.",
        ),
    )


def load_split_manifest(path: Path) -> dict[str, Any]:
    """Load and validate one split manifest."""

    payload = _read_json(path)
    split_id = payload.get("split_id") or "<unknown split>"
    if payload.get("schema_version") != 1:
        raise ValueError(f"{split_id}: split manifest must use schema_version=1")
    for field in ("split_id", "dataset", "role", "status", "row_id_policy", "leakage_boundary"):
        if not str(payload.get(field) or "").strip():
            raise ValueError(f"{split_id}: missing {field}")
    if payload["role"] not in ALLOWED_ROLES:
        raise ValueError(f"{split_id}: invalid role {payload['role']!r}")
    row_ids = payload.get("row_ids")
    if not isinstance(row_ids, list):
        raise ValueError(f"{split_id}: row_ids must be a list")
    if payload.get("row_ids_sha256") != _sha256_json(row_ids):
        raise ValueError(f"{split_id}: row_ids_sha256 mismatch")
    return payload


def split_manifest_map(split_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, dict[str, Any]]:
    """Return split manifests keyed by split id."""

    return {
        manifest["split_id"]: manifest
        for manifest in (load_split_manifest(path) for path in sorted(split_dir.glob("*.json")))
    }


def write_split_manifests(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cosql_root: Path = DEFAULT_COSQL_ROOT,
    synthetic_fixtures: Path = DEFAULT_SYNTHETIC_FIXTURES,
) -> tuple[Path, ...]:
    """Write split manifests and return their paths."""

    paths = []
    for manifest in build_split_manifests(
        cosql_root=cosql_root,
        synthetic_fixtures=synthetic_fixtures,
    ):
        path = output_dir / f"{manifest['split_id']}.json"
        _write_json(path, manifest)
        paths.append(path)
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cosql-root", type=Path, default=DEFAULT_COSQL_ROOT)
    parser.add_argument("--synthetic-fixtures", type=Path, default=DEFAULT_SYNTHETIC_FIXTURES)
    args = parser.parse_args()

    for path in write_split_manifests(
        output_dir=args.output_dir,
        cosql_root=args.cosql_root,
        synthetic_fixtures=args.synthetic_fixtures,
    ):
        print(path)


if __name__ == "__main__":
    main()
