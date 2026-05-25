from __future__ import annotations

import ast
from pathlib import Path

import pytest

import notebooks.labs.local_multiturn_sql_lab_support as lab_support
from notebooks.labs.local_multiturn_sql_lab_support import Accelerator

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_multiturn_lab_runs_without_requiring_gpu() -> None:
    device = lab_support.available_accelerator()

    assert device.kind in {"cuda", "mps", "xpu", "none"}
    assert isinstance(device.label, str)
    assert device.label

    report = lab_support.run_multiturn_lab()

    assert report["device"].kind == "cpu"
    assert report["detected_accelerator"].kind in {"cuda", "mps", "xpu", "none"}
    assert report["runtime_policy"]["reported_accelerators"] == "CUDA, MPS, XPU"
    assert set(report["systems"]) == {
        "direct_sql_baseline",
        "planner_first_sql",
        "semantic_value_sql",
        "semantic_dsl_planner",
        "behavior_recovery_sql",
    }
    assert report["systems"]["semantic_dsl_planner"]["value_accuracy"] == 1.0
    assert (
        report["systems"]["semantic_dsl_planner"]["value_accuracy"]
        > report["systems"]["direct_sql_baseline"]["value_accuracy"]
    )


def test_multiturn_lab_exposes_post_walkthrough_sections() -> None:
    report = lab_support.run_multiturn_lab()
    sections = report["walkthrough_sections"]
    section_ids = [section["section_id"] for section in sections]

    assert section_ids == [
        "research_question",
        "single_turn_gap",
        "proxy_slice",
        "target_comparison",
        "execution_trace",
        "claim_boundary",
        "next_gates",
    ]
    assert "small specialized model" in sections[0]["reader_question"]
    assert "hosted SOTA" in sections[0]["reader_question"]
    assert "single-turn SQL plus chat history" in sections[1]["takeaway"]
    assert "100 CoSQL turns" in sections[2]["takeaway"]
    assert "five fine-tuning targets" in sections[3]["takeaway"]
    assert "four-turn SQLite scenario" in sections[4]["takeaway"]
    assert "not a benchmark result" in sections[5]["takeaway"]
    assert "non-oracle planner" in sections[6]["next_artifact"]
    assert "BIRD-Interact" in sections[6]["next_artifact"]


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
    assert report["runtime_policy"]["device_preference"] == "cpu"


@pytest.mark.parametrize("accelerator_kind", ["cuda", "mps", "xpu"])
def test_multiturn_lab_auto_reports_detected_accelerator_without_using_it(
    monkeypatch, accelerator_kind: str
) -> None:
    monkeypatch.setattr(
        lab_support,
        "available_accelerator",
        lambda: Accelerator(
            kind=accelerator_kind,
            label=f"{accelerator_kind} test device",
            torch_available=True,
        ),
    )

    report = lab_support.run_multiturn_lab(device_preference="auto")

    assert report["device"].kind == "cpu"
    assert report["device"].label == "cpu"
    assert report["detected_accelerator"].kind == accelerator_kind
    assert report["runtime_policy"]["device_preference"] == "auto"
    assert report["runtime_policy"]["fallback"] is None
    assert report["runtime_policy"]["accelerator_usage"] == "reported_only"


def test_multiturn_lab_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        lab_support,
        "available_accelerator",
        lambda: Accelerator(
            kind="none",
            label="no accelerator detected (torch not installed)",
            torch_available=False,
        ),
    )

    report = lab_support.run_multiturn_lab(device_preference="auto")

    assert report["device"].kind == "cpu"
    assert report["runtime_policy"]["device_preference"] == "auto"
    assert report["runtime_policy"]["fallback"] == "cpu"
    assert report["runtime_policy"]["accelerator_usage"] == "reported_only"


def test_multiturn_lab_rejects_unknown_device_preference() -> None:
    with pytest.raises(ValueError, match="device_preference"):
        lab_support.run_multiturn_lab(device_preference="gpu")


def test_multiturn_lab_exposes_behavior_failures_not_just_scores() -> None:
    report = lab_support.run_multiturn_lab()
    direct_rows = [
        row for row in report["rows"] if row["system"] == "direct_sql_baseline"
    ]
    semantic_rows = [
        row for row in report["rows"] if row["system"] == "semantic_dsl_planner"
    ]

    assert {row["turn_id"] for row in direct_rows} == {"turn_1", "turn_2", "turn_3", "turn_4"}
    assert {row["turn_id"] for row in semantic_rows} == {"turn_1", "turn_2", "turn_3", "turn_4"}
    assert {
        row["failure_type"] for row in direct_rows if row["failure_type"]
    } >= {"value_grounding", "context_carryover"}
    assert all(row["value_match"] for row in semantic_rows)
    assert any("MEASURE(revenue)" in row["intermediate_plan"] for row in semantic_rows)


def test_multiturn_lab_compares_training_targets_explicitly() -> None:
    report = lab_support.run_multiturn_lab()
    systems = report["systems"]

    assert systems["direct_sql_baseline"]["value_accuracy"] == 1 / 4
    assert systems["planner_first_sql"]["context_carryover_accuracy"] == 1.0
    assert systems["planner_first_sql"]["value_grounding_accuracy"] < 1.0
    assert systems["semantic_value_sql"]["value_accuracy"] == 1.0
    assert systems["semantic_value_sql"]["measure_preservation_rate"] == 0.0
    assert systems["semantic_dsl_planner"]["measure_preservation_rate"] == 1.0
    assert systems["behavior_recovery_sql"]["recovery_success_rate"] == 1.0

    matrix = {row["system"]: row for row in report["method_matrix"]}
    assert matrix["direct_sql_baseline"]["fine_tuning_target"] == "assistant SQL"
    assert matrix["planner_first_sql"]["fine_tuning_target"] == "query plan then SQL"
    assert matrix["semantic_value_sql"]["fine_tuning_target"] == "semantic state then SQL"
    assert (
        matrix["semantic_dsl_planner"]["fine_tuning_target"]
        == "MEASURE-preserving DSL then SQL"
    )
    assert (
        matrix["behavior_recovery_sql"]["fine_tuning_target"]
        == "execution feedback then repair"
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


def test_multiturn_lab_scores_recovery_as_a_separate_behavior() -> None:
    report = lab_support.run_multiturn_lab()
    recovery_rows = [
        row for row in report["rows"] if row["turn_id"] == "turn_4"
    ]
    behavior_row = [
        row for row in recovery_rows if row["system"] == "behavior_recovery_sql"
    ][0]
    prior_behavior_row = [
        row
        for row in report["rows"]
        if row["turn_id"] == "turn_3" and row["system"] == "behavior_recovery_sql"
    ][0]
    semantic_row = [
        row for row in recovery_rows if row["system"] == "semantic_value_sql"
    ][0]

    assert report["scenario_contract"]["shared_input_sha256"]
    assert report["scenario_contract"]["turn_count"] == 4
    assert report["scenario_contract"]["recovery_turn_id"] == "turn_4"
    assert prior_behavior_row["value_match"] is False
    assert prior_behavior_row["actual_rows"] == []
    assert prior_behavior_row["failure_type"] == "value_grounding"
    assert behavior_row["value_match"] is True
    assert behavior_row["recovery_success"] is True
    assert "repairs empty result" in behavior_row["intermediate_plan"]
    assert semantic_row["value_match"] is True
    assert semantic_row["recovery_success"] is False


def test_shareable_lab_notebook_is_plain_python_marimo_app() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "labs" / "local_multiturn_sql_lab.py"
    source = notebook_path.read_text()

    ast.parse(source, filename=str(notebook_path))
    assert "import marimo" in source
    assert "app = marimo.App" in source
    assert "run_multiturn_lab" in source
    assert "data_engineering_gates" in source
    assert "mo.ui.dropdown" in source
    assert 'value="cpu"' in source
    assert "device_preference=runtime_choice.value" in source
    for heading in [
        "## 1. Research question",
        "## 2. Why single-turn SQL fails here",
        "## 3. The proxy slice",
        "## 4. Candidate fine-tuning targets",
        "## 5. Execution trace",
        "## Data engineering gates",
        "## 6. Boundary and next gates",
    ]:
        assert heading in source
    assert 'if __name__ == "__main__":' in source
    assert "app.run()" in source


def test_blog_readme_points_to_shareable_lab_notebook() -> None:
    readme = (REPO_ROOT / "docs" / "blog" / "README.md").read_text()

    assert "notebooks/labs/local_multiturn_sql_lab.py" in readme
    assert "defaults to CPU" in readme
    assert "planner-first" in readme
    assert "semantic value grounding" in readme
    assert "MEASURE()" in readme
    assert "behavior/recovery" in readme
