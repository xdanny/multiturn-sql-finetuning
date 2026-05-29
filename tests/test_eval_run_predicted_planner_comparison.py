from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_predicted_planner_comparison import (
    run_predicted_planner_comparison,
    validate_comparison_inputs,
    write_comparison_preflight,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _record(*, mode: str, sql: str = "SELECT name FROM singer;") -> dict:
    record = {
        "id": "dialog-a",
        "database_id": "music",
        "source": "unit",
        "evaluation_mode": mode,
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
        "gold_plans": [
            {
                "parseable": True,
                "relevant_tables": ["singer"],
                "relevant_columns": ["singer.name"],
                "join_path": [],
                "query_skeleton": {"select": True},
                "projection_shape": {
                    "selected_count": 1,
                    "selected_expressions": ["singer.name"],
                    "preserve_duplicates": True,
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "sql"},
            {"role": "user", "content": "List singers."},
            {"role": "assistant", "content": sql},
        ],
    }
    if mode == "predicted_planner":
        record["predicted_plans"] = [
            {
                "parseable": True,
                "prediction_source": "unit_test",
                "relevant_tables": ["singer"],
                "relevant_columns": ["singer.name"],
                "join_path": [],
                "query_skeleton": {"select": True},
                "projection_shape": {
                    "selected_count": 1,
                    "selected_expressions": ["singer.name"],
                    "preserve_duplicates": True,
                },
            }
        ]
    return record


def test_validate_comparison_inputs_requires_matching_rows(tmp_path) -> None:
    direct_path = tmp_path / "direct.jsonl"
    predicted_path = tmp_path / "predicted.jsonl"
    _write_jsonl(direct_path, [_record(mode="non_oracle_generation")])
    _write_jsonl(
        predicted_path,
        [_record(mode="predicted_planner", sql="SELECT id FROM singer;")],
    )

    with pytest.raises(ValueError, match="row identity mismatch"):
        validate_comparison_inputs(
            direct_input=direct_path,
            predicted_input=predicted_path,
            limit=None,
        )


def test_validate_comparison_inputs_rejects_prompt_invalid_predicted_plan(tmp_path) -> None:
    direct_path = tmp_path / "direct.jsonl"
    predicted_path = tmp_path / "predicted.jsonl"
    predicted = _record(mode="predicted_planner")
    predicted["predicted_plans"][0]["relevant_tables"] = []
    predicted["predicted_plans"][0]["relevant_columns"] = []

    _write_jsonl(direct_path, [_record(mode="non_oracle_generation")])
    _write_jsonl(predicted_path, [predicted])

    with pytest.raises(ValueError, match="invalid prompt plan"):
        validate_comparison_inputs(
            direct_input=direct_path,
            predicted_input=predicted_path,
            limit=None,
        )


def test_write_comparison_preflight_records_ready_input_pair(tmp_path) -> None:
    direct_path = tmp_path / "direct.jsonl"
    predicted_path = tmp_path / "predicted.jsonl"
    preflight_path = tmp_path / "preflight.json"
    _write_jsonl(direct_path, [_record(mode="non_oracle_generation")])
    _write_jsonl(predicted_path, [_record(mode="predicted_planner")])

    preflight = write_comparison_preflight(
        direct_input=direct_path,
        predicted_input=predicted_path,
        output_path=preflight_path,
        limit=None,
    )

    written = json.loads(preflight_path.read_text())
    assert preflight["status"] == "ready_for_endpoint_pair"
    assert written["row_count"] == 1
    assert written["direct_evaluation_mode"] == "non_oracle_generation"
    assert written["predicted_evaluation_mode"] == "predicted_planner"
    assert written["claim_boundary"] == "preflight only; no SQL execution claim"


def test_run_predicted_planner_comparison_runs_both_eval_paths_then_compares(
    tmp_path, monkeypatch
) -> None:
    direct_path = tmp_path / "direct.jsonl"
    predicted_path = tmp_path / "predicted.jsonl"
    output_dir = tmp_path / "results"
    _write_jsonl(direct_path, [_record(mode="non_oracle_generation")])
    _write_jsonl(predicted_path, [_record(mode="predicted_planner")])

    generate_calls: list[list[dict[str, str]]] = []

    def fake_generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        generate_calls.append(messages)
        return "SELECT name FROM singer;", 1.0

    compare_calls: list[dict] = []

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {"compared": True}

    monkeypatch.setattr(
        "eval.run_predicted_planner_comparison.compare_predicted_planner_manifest_files",
        fake_compare,
    )

    exit_code = run_predicted_planner_comparison(
        direct_input=direct_path,
        predicted_input=predicted_path,
        output_dir=output_dir,
        run_id="lexical_probe",
        model_name="local-9b",
        endpoint="http://localhost:8000/v1",
        database_root=tmp_path / "db",
        limit=1,
        generate_fn=fake_generate,
    )

    assert exit_code == 0
    assert len(generate_calls) == 2
    direct_rows = [
        json.loads(line)
        for line in (output_dir / "lexical_probe.direct.jsonl").read_text().splitlines()
    ]
    predicted_rows = [
        json.loads(line)
        for line in (output_dir / "lexical_probe.predicted_planner.jsonl").read_text().splitlines()
    ]
    assert direct_rows[0]["prompt_variant"] == "direct_sql_control"
    assert predicted_rows[0]["prompt_variant"] == "predicted_planner"
    assert direct_rows[0]["model_name"] == "local-9b"
    assert predicted_rows[0]["model_name"] == "local-9b"
    assert compare_calls == [
        {
            "predicted_manifest_path": output_dir
            / "lexical_probe.predicted_planner.manifest.json",
            "direct_manifest_path": output_dir / "lexical_probe.direct.manifest.json",
            "output_path": output_dir / "lexical_probe.compared.manifest.json",
            "repo_root": Path("."),
        }
    ]
