from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.result_manifest import sha256_file
from eval.semantic_context_transfer import (
    build_semantic_context_transfer_preflight,
    compare_semantic_context_transfer_manifest_files,
    compare_semantic_context_transfer_manifests,
    write_semantic_context_transfer_preflight,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


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
        + "\n"
    )
    return path


def _prepared_dialog(*, semantic: bool = False, second_sql: str = "SELECT 2;") -> dict:
    first_question = "Show one."
    second_question = "Show two."
    record = {
        "id": "dialog-a",
        "dialog_id": "dialog-a",
        "database_id": "db",
        "source": "unit",
        "evaluation_mode": "non_oracle_generation",
        "history_policy": "gold_sql_teacher_forced",
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
        "messages": [
            {"role": "system", "content": "Return SQL."},
            {"role": "user", "content": first_question},
            {"role": "assistant", "content": "SELECT 1;"},
            {"role": "user", "content": second_question},
            {"role": "assistant", "content": second_sql},
        ],
    }
    if semantic:
        record["messages"][1]["content"] += (
            "\n\nDatabase value retrieval (non-oracle; matched only against user text so far):"
            "\n- t.c = \"one\" (matched: \"one\")"
        )
        record["semantic_value_retrieval"] = {
            "artifact_type": "semantic_value_retrieval_context",
            "index_source": "database_contents",
            "matched_turn_count": 1,
            "matched_value_count": 1,
        }
    return record


def _rollout_manifest(
    run_id: str,
    *,
    input_sha: str,
    output_path: str = "results/out.jsonl",
    value: float = 0.5,
    strict: float = 0.5,
    syntax: float = 1.0,
    model_name: str = "local-9b",
    endpoint: str = "http://127.0.0.1:8000/v1",
) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "benchmark": "prepared_rollout",
        "model_name": model_name,
        "endpoint": endpoint,
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "prompt_variant": None,
        "input_path": f"data/{run_id}.jsonl",
        "input_sha256": input_sha,
        "output_path": output_path,
        "output_sha256": f"{run_id}-output-sha",
        "row_count": 2,
        "metrics": {
            "history_policy": "model_generated_sql_rollout",
            "value_execution_accuracy": value,
            "strict_execution_accuracy": strict,
            "syntax_accuracy": syntax,
            "interaction_match_rate": 0.0,
        },
        "command": ["run-rollout"],
    }


def _result_rows() -> list[dict]:
    return [
        {
            "id": "dialog-a:0",
            "dialog_id": "dialog-a",
            "turn_index": 0,
            "database_id": "db",
            "reference_sql": "SELECT 1;",
            "history_policy": "model_generated_sql_rollout",
            "evaluation_mode": "non_oracle_generation",
            "value_execution_score": 1.0,
            "strict_execution_score": 1.0,
            "syntax_valid": True,
        },
        {
            "id": "dialog-a:1",
            "dialog_id": "dialog-a",
            "turn_index": 1,
            "database_id": "db",
            "reference_sql": "SELECT 2;",
            "history_policy": "model_generated_sql_rollout",
            "evaluation_mode": "non_oracle_generation",
            "value_execution_score": 0.0,
            "strict_execution_score": 0.0,
            "syntax_valid": True,
        },
    ]


def test_build_semantic_context_transfer_preflight_records_ready_pair(tmp_path: Path) -> None:
    normal_path = tmp_path / "normal.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    _write_jsonl(normal_path, [_prepared_dialog()])
    _write_jsonl(semantic_path, [_prepared_dialog(semantic=True)])

    preflight = build_semantic_context_transfer_preflight(
        normal_input_path=normal_path,
        semantic_input_path=semantic_path,
        value_index_manifest_path=value_index_manifest,
    )

    assert preflight["artifact_type"] == "semantic_context_transfer_preflight"
    assert preflight["status"] == "ready_for_semantic_context_rollout_pair"
    assert preflight["row_count"] == 2
    assert preflight["dialog_count"] == 1
    assert preflight["database_count"] == 1
    assert preflight["source_history_policies"] == {"gold_sql_teacher_forced": 2}
    assert preflight["rollout_target_history_policy"] == "model_generated_sql_rollout"
    assert preflight["normal_input_sha256"] == sha256_file(normal_path)
    assert preflight["semantic_input_sha256"] == sha256_file(semantic_path)
    assert preflight["value_index_manifest_sha256"] == sha256_file(value_index_manifest)
    assert preflight["claim_boundary"] == "preflight only; no SQL execution claim"


def test_build_semantic_context_transfer_preflight_rejects_identical_inputs(
    tmp_path: Path,
) -> None:
    normal_path = tmp_path / "normal.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    _write_jsonl(normal_path, [_prepared_dialog()])

    with pytest.raises(ValueError, match="must differ"):
        build_semantic_context_transfer_preflight(
            normal_input_path=normal_path,
            semantic_input_path=normal_path,
            value_index_manifest_path=value_index_manifest,
        )


def test_build_semantic_context_transfer_preflight_rejects_row_mismatch(
    tmp_path: Path,
) -> None:
    normal_path = tmp_path / "normal.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    _write_jsonl(normal_path, [_prepared_dialog()])
    _write_jsonl(semantic_path, [_prepared_dialog(semantic=True, second_sql="SELECT 3;")])

    with pytest.raises(ValueError, match="row identity mismatch"):
        build_semantic_context_transfer_preflight(
            normal_input_path=normal_path,
            semantic_input_path=semantic_path,
            value_index_manifest_path=value_index_manifest,
        )


def test_build_semantic_context_transfer_preflight_rejects_oracle_input(
    tmp_path: Path,
) -> None:
    normal_path = tmp_path / "normal.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    normal = _prepared_dialog()
    normal["messages"][1]["content"] += "\n\nOracle SQL planning hints: singer.name"
    _write_jsonl(normal_path, [normal])
    _write_jsonl(semantic_path, [_prepared_dialog(semantic=True)])

    with pytest.raises(ValueError, match="non-oracle"):
        build_semantic_context_transfer_preflight(
            normal_input_path=normal_path,
            semantic_input_path=semantic_path,
            value_index_manifest_path=value_index_manifest,
        )


def test_build_semantic_context_transfer_preflight_rejects_single_turn_source_history(
    tmp_path: Path,
) -> None:
    normal_path = tmp_path / "normal.jsonl"
    semantic_path = tmp_path / "semantic.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    normal = _prepared_dialog()
    semantic = _prepared_dialog(semantic=True)
    normal["history_policy"] = "single_turn"
    semantic["history_policy"] = "single_turn"
    _write_jsonl(normal_path, [normal])
    _write_jsonl(semantic_path, [semantic])

    with pytest.raises(ValueError, match="rollout-compatible source history"):
        build_semantic_context_transfer_preflight(
            normal_input_path=normal_path,
            semantic_input_path=semantic_path,
            value_index_manifest_path=value_index_manifest,
        )


def test_compare_semantic_context_transfer_manifests_records_context_delta() -> None:
    compared = compare_semantic_context_transfer_manifests(
        normal_manifest=_rollout_manifest("normal", input_sha="normal-input", value=0.5),
        semantic_manifest=_rollout_manifest(
            "semantic",
            input_sha="semantic-input",
            value=0.75,
            strict=0.5,
            syntax=1.0,
        ),
        normal_rows=_result_rows(),
        semantic_rows=_result_rows(),
        comparison_role="raw_qwen",
    )

    metrics = compared["metrics"]
    assert compared["artifact_type"] == "semantic_context_transfer_comparison"
    assert metrics["semantic_context_transfer_comparison_role"] == "raw_qwen"
    assert metrics["normal_context_comparison_run_id"] == "normal"
    assert metrics["semantic_context_value_delta_vs_normal"] == pytest.approx(0.25)
    assert metrics["semantic_context_strict_delta_vs_normal"] == pytest.approx(0.0)
    assert metrics["semantic_context_syntax_delta_vs_normal"] == pytest.approx(0.0)
    assert metrics["semantic_context_transfer_comparable_row_count"] == 2
    assert metrics["semantic_context_helped"] is True
    assert metrics["semantic_context_transfer_blockers"] == []
    assert compared["command"][-2:] == ["# compared-with-normal-context", "normal"]


def test_compare_semantic_context_transfer_records_negative_evidence() -> None:
    compared = compare_semantic_context_transfer_manifests(
        normal_manifest=_rollout_manifest(
            "normal",
            input_sha="normal-input",
            value=0.75,
            strict=0.75,
            syntax=1.0,
        ),
        semantic_manifest=_rollout_manifest(
            "semantic",
            input_sha="semantic-input",
            value=0.5,
            strict=0.5,
            syntax=0.5,
        ),
        normal_rows=_result_rows(),
        semantic_rows=_result_rows(),
        comparison_role="hosted_sonnet",
    )

    metrics = compared["metrics"]
    assert metrics["semantic_context_helped"] is False
    assert metrics["semantic_context_transfer_blockers"] == [
        "value accuracy did not improve",
        "strict accuracy regressed beyond policy",
        "syntax rate regressed beyond policy",
    ]


def test_compare_semantic_context_transfer_rejects_same_input_sha() -> None:
    with pytest.raises(ValueError, match="different inputs"):
        compare_semantic_context_transfer_manifests(
            normal_manifest=_rollout_manifest("normal", input_sha="same"),
            semantic_manifest=_rollout_manifest("semantic", input_sha="same"),
            normal_rows=_result_rows(),
            semantic_rows=_result_rows(),
            comparison_role="raw_qwen",
        )


def test_compare_semantic_context_transfer_rejects_result_row_mismatch() -> None:
    semantic_rows = _result_rows()
    semantic_rows[1]["reference_sql"] = "SELECT 99;"

    with pytest.raises(ValueError, match="row identity mismatch"):
        compare_semantic_context_transfer_manifests(
            normal_manifest=_rollout_manifest("normal", input_sha="normal-input"),
            semantic_manifest=_rollout_manifest("semantic", input_sha="semantic-input"),
            normal_rows=_result_rows(),
            semantic_rows=semantic_rows,
            comparison_role="raw_qwen",
        )


def test_compare_semantic_context_transfer_manifest_files_writes_output(
    tmp_path: Path,
) -> None:
    normal_input = tmp_path / "normal-input.jsonl"
    semantic_input = tmp_path / "semantic-input.jsonl"
    value_index_manifest = _value_index_manifest(tmp_path / "value_index.manifest.json")
    _write_jsonl(normal_input, [_prepared_dialog()])
    _write_jsonl(semantic_input, [_prepared_dialog(semantic=True)])
    preflight_path = tmp_path / "preflight.json"
    preflight = write_semantic_context_transfer_preflight(
        normal_input_path=normal_input,
        semantic_input_path=semantic_input,
        value_index_manifest_path=value_index_manifest,
        output_path=preflight_path,
    )

    normal_output = tmp_path / "results" / "normal.jsonl"
    semantic_output = tmp_path / "results" / "semantic.jsonl"
    _write_jsonl(normal_output, _result_rows())
    _write_jsonl(semantic_output, _result_rows())
    normal_manifest = _rollout_manifest(
        "normal",
        input_sha=preflight["normal_input_sha256"],
        output_path=str(normal_output.relative_to(tmp_path)),
        value=0.5,
    )
    semantic_manifest = _rollout_manifest(
        "semantic",
        input_sha=preflight["semantic_input_sha256"],
        output_path=str(semantic_output.relative_to(tmp_path)),
        value=0.75,
    )
    normal_manifest_path = tmp_path / "normal.manifest.json"
    semantic_manifest_path = tmp_path / "semantic.manifest.json"
    output_path = tmp_path / "compared.manifest.json"
    normal_manifest_path.write_text(json.dumps(normal_manifest) + "\n")
    semantic_manifest_path.write_text(json.dumps(semantic_manifest) + "\n")

    compared = compare_semantic_context_transfer_manifest_files(
        normal_manifest_path=normal_manifest_path,
        semantic_manifest_path=semantic_manifest_path,
        preflight_manifest_path=preflight_path,
        output_path=output_path,
        comparison_role="structured_brief_lora",
        repo_root=tmp_path,
    )

    assert json.loads(output_path.read_text()) == compared
    assert compared["metrics"]["semantic_context_transfer_preflight_sha256"] == sha256_file(
        preflight_path
    )
    assert compared["metrics"]["semantic_context_transfer_value_index_manifest_sha256"] == (
        sha256_file(value_index_manifest)
    )
