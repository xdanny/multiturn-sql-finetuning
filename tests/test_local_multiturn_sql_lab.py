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
    assert set(report["systems"]) == {"direct_sql_baseline", "semantic_dsl_planner"}
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
