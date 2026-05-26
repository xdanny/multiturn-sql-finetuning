"""Shared validation for paired prepared-input files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PairInputSpec:
    label: str
    left_name: str
    right_name: str
    row_identity_fields: tuple[str, ...]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _row_identity(row: dict[str, Any], *, fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(field)) for field in fields)


def validate_pair_input_paths(
    *,
    left_input_path: Path,
    right_input_path: Path,
    spec: PairInputSpec,
) -> dict[str, Any]:
    left_rows = _load_jsonl(left_input_path)
    right_rows = _load_jsonl(right_input_path)
    if not left_rows or not right_rows:
        raise ValueError(f"{spec.label} comparison requires non-empty paired inputs")
    if [_row_identity(row, fields=spec.row_identity_fields) for row in left_rows] != [
        _row_identity(row, fields=spec.row_identity_fields) for row in right_rows
    ]:
        raise ValueError(
            f"{spec.left_name} and {spec.right_name} training inputs must share row identity"
        )
    return {
        "left_rows": left_rows,
        "right_rows": right_rows,
        "row_count": len(left_rows),
    }
