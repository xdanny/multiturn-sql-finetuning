from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from eval.compare_alias_column_context import compare_alias_column_context_manifest_files


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(generated_sql: str, score: float = 1.0) -> dict:
    return {
        "id": "d1:0",
        "dialog_id": "d1",
        "turn_index": 0,
        "database_id": "store",
        "reference_sql": "SELECT 1",
        "evaluation_mode": "non_oracle_generation",
        "generated_sql": generated_sql,
        "value_execution_score": score,
        "strict_execution_score": score,
    }


def _manifest(path: Path, output_path: Path, *, run_id: str, score: float = 1.0) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "benchmark": "prepared",
                "model_name": "local-9b",
                "endpoint": "local",
                "evaluation_mode": "non_oracle_generation",
                "oracle_allowed": False,
                "prompt_variant": run_id,
                "input_path": "input.jsonl",
                "input_sha256": "input",
                "output_path": str(output_path.relative_to(path.parents[1])),
                "output_sha256": _sha256(output_path),
                "row_count": 1,
                "metrics": {
                    "value_execution_accuracy": score,
                    "strict_execution_accuracy": score,
                },
                "command": ["unit"],
            }
        )
    )


def test_compare_alias_column_context_manifest_files_writes_deltas(tmp_path) -> None:
    direct_rows = tmp_path / "results" / "direct.jsonl"
    alias_rows = tmp_path / "results" / "alias.jsonl"
    direct_manifest = tmp_path / "results" / "direct.manifest.json"
    alias_manifest = tmp_path / "results" / "alias.manifest.json"
    context_manifest = tmp_path / "docs" / "alias.manifest.json"
    compared = tmp_path / "results" / "compared.manifest.json"
    _write_jsonl(direct_rows, [_row("SELECT 0", score=0.0)])
    _write_jsonl(alias_rows, [_row("SELECT 1", score=1.0)])
    context_manifest.parent.mkdir(parents=True)
    context_manifest.write_text(json.dumps({"artifact_type": "alias_column_context"}))
    _manifest(direct_manifest, direct_rows, run_id="direct", score=0.0)
    _manifest(alias_manifest, alias_rows, run_id="alias", score=1.0)

    payload = compare_alias_column_context_manifest_files(
        alias_manifest_path=alias_manifest,
        direct_manifest_path=direct_manifest,
        alias_context_manifest_path=context_manifest,
        output_path=compared,
        repo_root=tmp_path,
    )

    metrics = payload["metrics"]
    assert metrics["alias_column_context_value_delta_vs_direct_sql"] == 1.0
    assert metrics["alias_column_context_comparable_row_count"] == 1
    assert json.loads(compared.read_text())["run_id"] == "alias"


def test_compare_alias_column_context_rejects_row_identity_mismatch(tmp_path) -> None:
    direct_rows = tmp_path / "results" / "direct.jsonl"
    alias_rows = tmp_path / "results" / "alias.jsonl"
    direct_manifest = tmp_path / "results" / "direct.manifest.json"
    alias_manifest = tmp_path / "results" / "alias.manifest.json"
    context_manifest = tmp_path / "docs" / "alias.manifest.json"
    _write_jsonl(direct_rows, [_row("SELECT 0", score=0.0)])
    mismatched = _row("SELECT 1", score=1.0)
    mismatched["turn_index"] = 1
    _write_jsonl(alias_rows, [mismatched])
    context_manifest.parent.mkdir(parents=True)
    context_manifest.write_text(json.dumps({"artifact_type": "alias_column_context"}))
    _manifest(direct_manifest, direct_rows, run_id="direct", score=0.0)
    _manifest(alias_manifest, alias_rows, run_id="alias", score=1.0)

    with pytest.raises(ValueError, match="identity mismatch"):
        compare_alias_column_context_manifest_files(
            alias_manifest_path=alias_manifest,
            direct_manifest_path=direct_manifest,
            alias_context_manifest_path=context_manifest,
            output_path=tmp_path / "results" / "compared.manifest.json",
            repo_root=tmp_path,
        )
