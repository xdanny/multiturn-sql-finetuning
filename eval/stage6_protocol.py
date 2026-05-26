"""Shared Stage 6 validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def validate_result_manifest_against_contract(
    *,
    result_manifest_path: Path,
    contract_manifest_path: Path,
    expected_benchmark: str,
    label: str,
) -> dict[str, Any]:
    result = _load_json(result_manifest_path)
    contract = _load_json(contract_manifest_path)
    if result.get("benchmark") != expected_benchmark:
        raise ValueError(f"{label} result manifest must use benchmark={expected_benchmark}")
    if contract.get("benchmark") != expected_benchmark:
        raise ValueError(f"{label} contract manifest must use benchmark={expected_benchmark}")
    expected_input_sha = contract.get("output_sha256")
    if not expected_input_sha:
        raise ValueError(f"{label} contract manifest is missing output_sha256")
    if result.get("input_sha256") != expected_input_sha:
        raise ValueError(f"{label} result manifest input sha does not match contract output sha")
    return result
