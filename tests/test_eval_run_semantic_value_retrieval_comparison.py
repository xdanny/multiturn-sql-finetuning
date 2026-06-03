from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.result_manifest import sha256_file
from eval.run_semantic_value_retrieval_comparison import (
    annotate_semantic_manifest_with_value_index,
    run_semantic_value_retrieval_comparison,
    validate_comparison_inputs,
    write_comparison_preflight,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _record(*, semantic: bool = False, sql: str = "SELECT name FROM singer;") -> dict:
    question = "List singers."
    if semantic:
        question += "\n\nRetrieved database values:\n- singer.name: Alice, Bob"
    return {
        "id": "dialog-a",
        "database_id": "music",
        "source": "unit",
        "evaluation_mode": "non_oracle_generation",
        "messages": [
            {"role": "system", "content": "sql"},
            {"role": "user", "content": question},
            {"role": "assistant", "content": sql},
        ],
    }


def _value_index_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "non_oracle_value_index_v1",
                "index_source": "database_contents",
                "output_path": "docs/value_index.jsonl",
                "output_sha256": "value-index-output",
            }
        )
    )
    return path


def test_validate_comparison_inputs_requires_matching_rows(tmp_path) -> None:
    direct_path = tmp_path / "direct.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    _write_jsonl(direct_path, [_record()])
    _write_jsonl(semantic_path, [_record(semantic=True, sql="SELECT id FROM singer;")])

    with pytest.raises(ValueError, match="row identity mismatch"):
        validate_comparison_inputs(
            direct_input=direct_path,
            semantic_input=semantic_path,
            value_index_manifest=value_index_manifest,
            limit=None,
        )


def test_validate_comparison_inputs_rejects_identical_inputs(tmp_path) -> None:
    direct_path = tmp_path / "direct.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    _write_jsonl(direct_path, [_record()])

    with pytest.raises(ValueError, match="must differ"):
        validate_comparison_inputs(
            direct_input=direct_path,
            semantic_input=direct_path,
            value_index_manifest=value_index_manifest,
            limit=None,
        )


def test_write_comparison_preflight_records_ready_input_pair(tmp_path) -> None:
    direct_path = tmp_path / "direct.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    preflight_path = tmp_path / "preflight.json"
    _write_jsonl(direct_path, [_record()])
    _write_jsonl(semantic_path, [_record(semantic=True)])

    preflight = write_comparison_preflight(
        direct_input=direct_path,
        semantic_input=semantic_path,
        value_index_manifest=value_index_manifest,
        output_path=preflight_path,
        limit=None,
    )

    written = json.loads(preflight_path.read_text())
    assert preflight["status"] == "ready_for_endpoint_pair"
    assert written["artifact_type"] == "semantic_value_retrieval_comparison_preflight"
    assert written["row_count"] == 1
    assert written["direct_evaluation_mode"] == "non_oracle_generation"
    assert written["semantic_evaluation_mode"] == "non_oracle_generation"
    assert written["value_index_manifest_sha256"] == sha256_file(value_index_manifest)
    assert written["claim_boundary"] == "preflight only; no SQL execution claim"


def test_annotate_semantic_manifest_adds_value_index_provenance(tmp_path) -> None:
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    semantic_manifest = tmp_path / "semantic.manifest.json"
    semantic_manifest.write_text(
        json.dumps(
            {
                "run_id": "semantic",
                "metrics": {"value_execution_accuracy": 0.5},
                "command": ["run-semantic"],
            }
        )
    )

    annotated = annotate_semantic_manifest_with_value_index(
        semantic_manifest_path=semantic_manifest,
        value_index_manifest_path=value_index_manifest,
    )

    written = json.loads(semantic_manifest.read_text())
    assert annotated == written
    assert written["metrics"]["value_index_manifest_sha256"] == sha256_file(
        value_index_manifest
    )
    assert written["metrics"]["value_index_index_source"] == "database_contents"
    assert "# value-index-manifest" in written["command"]


def test_run_semantic_value_retrieval_comparison_runs_both_eval_paths_then_compares(
    tmp_path, monkeypatch
) -> None:
    direct_path = tmp_path / "direct.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    output_dir = tmp_path / "results"
    _write_jsonl(direct_path, [_record()])
    _write_jsonl(semantic_path, [_record(semantic=True)])

    run_calls: list[dict] = []

    def fake_run_eval(**kwargs):
        run_calls.append(kwargs)
        kwargs["manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["output"].stem,
                    "output_path": str(kwargs["output"]),
                    "row_count": 1,
                    "evaluation_mode": "non_oracle_generation",
                    "metrics": {
                        "value_execution_accuracy": 0.5,
                        "strict_execution_accuracy": 0.5,
                    },
                    "command": ["fake-run"],
                }
            )
        )
        return 0

    compare_calls: list[dict] = []

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        semantic_manifest = json.loads(kwargs["semantic_manifest_path"].read_text())
        assert semantic_manifest["metrics"]["value_index_manifest_sha256"] == sha256_file(
            value_index_manifest
        )
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {"compared": True}

    monkeypatch.setattr("eval.run_semantic_value_retrieval_comparison.run_eval", fake_run_eval)
    monkeypatch.setattr(
        "eval.run_semantic_value_retrieval_comparison.compare_semantic_value_retrieval_manifest_files",
        fake_compare,
    )

    exit_code = run_semantic_value_retrieval_comparison(
        direct_input=direct_path,
        semantic_input=semantic_path,
        value_index_manifest=value_index_manifest,
        output_dir=output_dir,
        run_id="semantic_probe",
        model_name="local-9b",
        endpoint="http://localhost:8000/v1",
        database_root=tmp_path / "db",
        api_key="EMPTY",
        temperature=0.0,
        max_tokens=256,
        limit=1,
    )

    assert exit_code == 0
    assert [call["input_path"] for call in run_calls] == [direct_path, semantic_path]
    assert {call["model_name"] for call in run_calls} == {"local-9b"}
    assert [call["prompt_variant"] for call in run_calls] == [
        "direct_sql_control",
        "semantic_value_retrieval",
    ]
    assert compare_calls == [
        {
            "semantic_manifest_path": output_dir
            / "semantic_probe.semantic_value_retrieval.manifest.json",
            "direct_manifest_path": output_dir / "semantic_probe.direct.manifest.json",
            "value_index_manifest_path": value_index_manifest,
            "output_path": output_dir / "semantic_probe.compared.manifest.json",
            "repo_root": Path("."),
        }
    ]
