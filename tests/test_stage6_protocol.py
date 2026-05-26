from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.stage6_protocol import validate_result_manifest_against_contract


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_validate_result_manifest_against_contract_requires_matching_input_sha(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "contract.manifest.json"
    result_path = tmp_path / "result.manifest.json"
    _write_json(
        contract_path,
        {
            "benchmark": "prepared",
            "output_path": "docs/data_artifacts/hosted_baseline_rows.jsonl",
            "output_sha256": "expected-sha",
        },
    )
    _write_json(
        result_path,
        {
            "benchmark": "prepared",
            "input_path": "docs/data_artifacts/hosted_baseline_rows.jsonl",
            "input_sha256": "wrong-sha",
        },
    )

    with pytest.raises(ValueError, match="input sha"):
        validate_result_manifest_against_contract(
            result_manifest_path=result_path,
            contract_manifest_path=contract_path,
            expected_benchmark="prepared",
            label="hosted baseline",
        )


def test_validate_result_manifest_against_contract_accepts_matching_sha(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "contract.manifest.json"
    result_path = tmp_path / "result.manifest.json"
    _write_json(
        contract_path,
        {
            "benchmark": "prepared",
            "output_path": "docs/data_artifacts/hosted_baseline_rows.jsonl",
            "output_sha256": "expected-sha",
        },
    )
    _write_json(
        result_path,
        {
            "benchmark": "prepared",
            "input_path": "docs/data_artifacts/hosted_baseline_rows.jsonl",
            "input_sha256": "expected-sha",
        },
    )

    payload = validate_result_manifest_against_contract(
        result_manifest_path=result_path,
        contract_manifest_path=contract_path,
        expected_benchmark="prepared",
        label="hosted baseline",
    )

    assert payload["input_sha256"] == "expected-sha"
