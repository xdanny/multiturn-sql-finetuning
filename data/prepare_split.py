"""Prepare non-oracle training/eval rows from frozen split manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from data.prepare import DatasetSpec, iter_formatted_records, write_jsonl
from data.split_manifest import DEFAULT_OUTPUT_DIR, load_split_manifest, split_manifest_map


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_source_path(manifest: dict[str, Any], *, source_roots: Iterable[Path]) -> Path:
    source_path = manifest.get("source_path")
    if not source_path:
        raise ValueError(f"{manifest['split_id']}: split manifest has no source_path")

    path = Path(str(source_path))
    candidates = [path] if path.is_absolute() else [*(root / path for root in source_roots), Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"{manifest['split_id']}: source_path not found; searched {searched}")


def _resolve_tables_path(tables_path: Path | None, *, source_roots: Iterable[Path]) -> Path | None:
    if tables_path is None:
        return None
    candidates = (
        [tables_path]
        if tables_path.is_absolute()
        else [Path.cwd() / tables_path, *(root / tables_path for root in source_roots)]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return tables_path


def _selected_rows(raw_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    selection = manifest.get("selection") or {}
    if selection.get("type") != "dialog_index_range":
        raise ValueError(f"{manifest['split_id']}: unsupported selection type {selection.get('type')!r}")
    start = int(selection.get("start_inclusive", 0))
    end = int(selection.get("end_exclusive", 0))
    if start < 0 or end < start or end > len(raw_rows):
        raise ValueError(f"{manifest['split_id']}: selection range {start}:{end} outside source rows")
    rows = raw_rows[start:end]
    expected_row_ids = manifest.get("row_ids") or []
    if expected_row_ids and len(rows) != len(expected_row_ids):
        raise ValueError(
            f"{manifest['split_id']}: selected {len(rows)} rows but manifest names "
            f"{len(expected_row_ids)} row_ids"
        )
    return rows


def prepare_records_from_split(
    split_manifest_path: Path,
    *,
    tables_path: Path | None = None,
    source_roots: Iterable[Path] = (),
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return prepared records and source metadata for one ready CoSQL split."""

    manifest = load_split_manifest(split_manifest_path)
    if manifest.get("status") != "ready":
        raise ValueError(f"{manifest['split_id']}: only ready split manifests can be prepared")
    if manifest.get("dataset") != "CoSQL":
        raise ValueError(f"{manifest['split_id']}: only CoSQL split preparation is supported")

    source_roots = tuple(source_roots)
    source_path = _resolve_source_path(manifest, source_roots=source_roots)
    source_sha256 = _sha256_file(source_path)
    if manifest.get("source_sha256") and source_sha256 != manifest["source_sha256"]:
        raise ValueError(f"{manifest['split_id']}: source_sha256 mismatch for {source_path}")

    raw_rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError(f"{manifest['split_id']}: source data must be a JSON list")
    selected = _selected_rows(raw_rows, manifest)
    if limit is not None:
        selected = selected[:limit]

    resolved_tables_path = _resolve_tables_path(tables_path, source_roots=source_roots)
    spec = DatasetSpec(
        name=manifest["split_id"],
        split=str(manifest.get("source_split") or manifest["role"]),
        formatter="cosql",
        tables_path=str(resolved_tables_path) if resolved_tables_path else None,
    )
    records = list(iter_formatted_records(selected, spec=spec, limit=None))
    row_ids = list(manifest.get("row_ids") or [])[: len(records)]
    for index, record in enumerate(records):
        record["split_id"] = manifest["split_id"]
        record["split_role"] = manifest["role"]
        record["split_row_id"] = row_ids[index] if index < len(row_ids) else None
        record["split_source_path"] = manifest["source_path"]
        record["split_source_sha256"] = source_sha256
        record["split_row_ids_sha256"] = manifest.get("row_ids_sha256")

    return records, {
        "split_manifest": manifest,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "selected_row_ids": row_ids,
    }


def build_prepared_split_manifest(
    *,
    records: list[dict[str, Any]],
    split_manifest_path: Path,
    output_path: Path,
    source_metadata: dict[str, Any],
    command: list[str] | None = None,
) -> dict[str, Any]:
    split_manifest = source_metadata["split_manifest"]
    return {
        "schema_version": 1,
        "artifact_type": "prepared_split_dataset",
        "split_id": split_manifest["split_id"],
        "split_role": split_manifest["role"],
        "source_path": split_manifest.get("source_path"),
        "source_sha256": source_metadata["source_sha256"],
        "split_manifest_path": str(split_manifest_path),
        "split_manifest_sha256": _sha256_file(split_manifest_path),
        "selection": split_manifest.get("selection"),
        "row_count": len(records),
        "assistant_turn_count": sum(int(record.get("assistant_turn_count") or 0) for record in records),
        "row_ids": source_metadata["selected_row_ids"],
        "row_ids_sha256": _sha256_json(source_metadata["selected_row_ids"]),
        "evaluation_modes": dict(Counter(str(record.get("evaluation_mode", "unknown")) for record in records)),
        "turn_formats": dict(Counter(str(record.get("turn_format", "unknown")) for record in records)),
        "history_policies": dict(Counter(str(record.get("history_policy", "unknown")) for record in records)),
        "oracle_policy": "non_oracle_generation",
        "uses_oracle_planning_hints": any(bool(record.get("uses_oracle_planning_hints")) for record in records),
        "semantic_context_pruned_by_oracle_labels": any(
            bool(record.get("semantic_context_pruned_by_oracle_labels")) for record in records
        ),
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path) if output_path.exists() else None,
        "command": command or [],
    }


def write_prepared_split(
    *,
    split_manifest_path: Path,
    output_path: Path,
    manifest_output_path: Path,
    tables_path: Path | None = None,
    source_roots: Iterable[Path] = (),
    limit: int | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    records, metadata = prepare_records_from_split(
        split_manifest_path,
        tables_path=tables_path,
        source_roots=source_roots,
        limit=limit,
    )
    write_jsonl(records, output_path)
    manifest = build_prepared_split_manifest(
        records=records,
        split_manifest_path=split_manifest_path,
        output_path=output_path,
        source_metadata=metadata,
        command=command,
    )
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _split_path_from_args(args: argparse.Namespace) -> Path:
    if args.split_manifest:
        return args.split_manifest
    if not args.split_id:
        raise ValueError("pass either --split-manifest or --split-id")
    manifests = split_manifest_map(args.split_dir)
    try:
        manifest = manifests[args.split_id]
    except KeyError as exc:
        raise ValueError(f"unknown split id {args.split_id!r}") from exc
    return args.split_dir / f"{manifest['split_id']}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-id", default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--tables-path", type=Path, default=Path("data/raw/cosql_dataset/tables.json"))
    parser.add_argument(
        "--source-root",
        type=Path,
        action="append",
        default=[],
        help="Additional root used to resolve repo-relative split source_path values.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    split_manifest_path = _split_path_from_args(args)
    command = [
        "uv",
        "run",
        "--active",
        "--no-sync",
        "python",
        "-m",
        "data.prepare_split",
    ]
    if args.split_id:
        command.extend(["--split-id", args.split_id])
    if args.split_manifest:
        command.extend(["--split-manifest", str(args.split_manifest)])
    command.extend(["--output", str(args.output), "--manifest-output", str(args.manifest_output)])
    if args.tables_path:
        command.extend(["--tables-path", str(args.tables_path)])
    for source_root in args.source_root:
        command.extend(["--source-root", str(source_root)])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])

    manifest = write_prepared_split(
        split_manifest_path=split_manifest_path,
        output_path=args.output,
        manifest_output_path=args.manifest_output,
        tables_path=args.tables_path,
        source_roots=args.source_root,
        limit=args.limit,
        command=command,
    )
    print(
        f"Wrote {args.output} and {args.manifest_output} "
        f"for {manifest['split_id']} ({manifest['row_count']} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
