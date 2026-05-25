from __future__ import annotations

import ast
from pathlib import Path

import notebooks.labs.local_multiturn_sql_lab_support as lab_support
from notebooks.labs.local_multiturn_sql_lab_support import Accelerator

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_multiturn_lab_runs_without_requiring_gpu() -> None:
    device = lab_support.available_accelerator()

    assert device.kind in {"cuda", "mps", "xpu", "cpu"}
    assert isinstance(device.label, str)
    assert device.label

    report = lab_support.run_multiturn_lab()

    assert report["device"].kind == "cpu"
    assert report["detected_accelerator"].kind in {"cuda", "mps", "xpu", "cpu"}
    assert set(report["systems"]) == {
        "direct_sql_baseline",
        "planner_first_sql",
        "semantic_value_sql",
        "semantic_dsl_planner",
    }
    assert report["systems"]["semantic_dsl_planner"]["value_accuracy"] == 1.0
    assert (
        report["systems"]["semantic_dsl_planner"]["value_accuracy"]
        > report["systems"]["direct_sql_baseline"]["value_accuracy"]
    )


def test_multiturn_lab_defaults_to_cpu_even_when_accelerator_is_detected(monkeypatch) -> None:
    monkeypatch.setattr(
        lab_support,
        "available_accelerator",
        lambda: Accelerator(kind="cuda", label="cuda test device", torch_available=True),
    )

    report = lab_support.run_multiturn_lab()

    assert report["device"].kind == "cpu"
    assert report["device"].label == "cpu"
    assert report["detected_accelerator"].kind == "cuda"


def test_multiturn_lab_exposes_behavior_failures_not_just_scores() -> None:
    report = lab_support.run_multiturn_lab()
    direct_rows = [
        row for row in report["rows"] if row["system"] == "direct_sql_baseline"
    ]
    semantic_rows = [
        row for row in report["rows"] if row["system"] == "semantic_dsl_planner"
    ]

    assert {row["turn_id"] for row in direct_rows} == {"turn_1", "turn_2", "turn_3"}
    assert {row["turn_id"] for row in semantic_rows} == {"turn_1", "turn_2", "turn_3"}
    assert {
        row["failure_type"] for row in direct_rows if row["failure_type"]
    } >= {"value_grounding", "context_carryover"}
    assert all(row["value_match"] for row in semantic_rows)
    assert any("MEASURE(revenue)" in row["intermediate_plan"] for row in semantic_rows)


def test_multiturn_lab_compares_training_targets_explicitly() -> None:
    report = lab_support.run_multiturn_lab()
    systems = report["systems"]

    assert systems["direct_sql_baseline"]["value_accuracy"] == 1 / 3
    assert systems["planner_first_sql"]["context_carryover_accuracy"] == 1.0
    assert systems["planner_first_sql"]["value_grounding_accuracy"] < 1.0
    assert systems["semantic_value_sql"]["value_accuracy"] == 1.0
    assert systems["semantic_value_sql"]["measure_preservation_rate"] == 0.0
    assert systems["semantic_dsl_planner"]["measure_preservation_rate"] == 1.0

    matrix = {row["system"]: row for row in report["method_matrix"]}
    assert matrix["direct_sql_baseline"]["fine_tuning_target"] == "assistant SQL"
    assert matrix["planner_first_sql"]["fine_tuning_target"] == "query plan then SQL"
    assert matrix["semantic_value_sql"]["fine_tuning_target"] == "semantic state then SQL"
    assert (
        matrix["semantic_dsl_planner"]["fine_tuning_target"]
        == "MEASURE-preserving DSL then SQL"
    )


def test_multiturn_lab_rows_score_semantic_subtasks() -> None:
    report = lab_support.run_multiturn_lab()
    semantic_sql_rows = [
        row for row in report["rows"] if row["system"] == "semantic_value_sql"
    ]
    dsl_rows = [
        row for row in report["rows"] if row["system"] == "semantic_dsl_planner"
    ]

    assert all(row["context_carryover"] for row in semantic_sql_rows)
    assert all(row["value_grounded"] for row in semantic_sql_rows)
    assert not any(row["measure_preserved"] for row in semantic_sql_rows)
    assert all(row["measure_preserved"] for row in dsl_rows)


def test_shareable_lab_notebook_is_plain_python_marimo_app() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "labs" / "local_multiturn_sql_lab.py"
    source = notebook_path.read_text()

    ast.parse(source, filename=str(notebook_path))
    assert "import marimo" in source
    assert "app = marimo.App" in source
    assert "run_multiturn_lab" in source
    assert 'if __name__ == "__main__":' in source
    assert "app.run()" in source


def test_blog_readme_points_to_shareable_lab_notebook() -> None:
    readme = (REPO_ROOT / "docs" / "blog" / "README.md").read_text()

    assert "notebooks/labs/local_multiturn_sql_lab.py" in readme
    assert "defaults to CPU" in readme
    assert "planner-first" in readme
    assert "semantic value grounding" in readme
    assert "MEASURE()" in readme
