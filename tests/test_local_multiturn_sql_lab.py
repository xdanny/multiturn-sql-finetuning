from __future__ import annotations

import ast
import json
import subprocess
import tomllib
from pathlib import Path

import pytest

import notebooks.labs.local_multiturn_sql_lab_support as lab_support

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_multiturn_lab_runs_without_requiring_gpu() -> None:
    device = lab_support.available_accelerator()

    assert device.kind in {"cuda", "mps", "xpu", "none"}
    assert isinstance(device.label, str)
    assert device.label

    report = lab_support.run_multiturn_lab()

    assert report["device"].kind in {"cpu", "cuda", "mps", "xpu"}
    assert report["detected_accelerator"].kind in {"cuda", "mps", "xpu", "none"}
    assert {status["kind"] for status in report["accelerator_report"]} == {
        "cuda",
        "mps",
        "xpu",
    }
    assert all("available" in status for status in report["accelerator_report"])
    assert report["runtime_policy"]["reported_accelerators"] == "CUDA, MPS, XPU"
    assert report["runtime_policy"]["accelerator_usage"] == "selected_if_available"
    assert report["synthetic_fixture_summary"]["fixture_count"] == 5
    assert {
        "value_normalization",
        "entity_resolution",
        "grain_fanout",
        "measure_preservation",
        "recovery",
    } <= set(report["synthetic_fixture_summary"]["failure_mode_counts"])
    assert {row["fixture_id"] for row in report["synthetic_fixture_table"]} == {
        "value_normalization_france",
        "entity_resolution_followup",
        "grain_fanout_bridge",
        "measure_preservation_metric",
        "recovery_empty_result",
    }
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


def test_multiturn_lab_can_force_cpu_even_when_accelerator_is_detected(monkeypatch) -> None:
    monkeypatch.setattr(
        lab_support,
        "accelerator_statuses",
        lambda: [
            {
                "kind": "cuda",
                "label": "cuda test device",
                "torch_available": True,
                "available": True,
                "usage": "available_for_auto",
            },
            {
                "kind": "mps",
                "label": "mps unavailable",
                "torch_available": True,
                "available": False,
                "usage": "available_for_auto",
            },
            {
                "kind": "xpu",
                "label": "xpu unavailable",
                "torch_available": True,
                "available": False,
                "usage": "available_for_auto",
            },
        ],
    )

    report = lab_support.run_multiturn_lab(device_preference="cpu")

    assert report["device"].kind == "cpu"
    assert report["device"].label == "cpu"
    assert report["detected_accelerator"].kind == "cuda"
    assert report["runtime_policy"]["device_preference"] == "cpu"
    assert report["runtime_policy"]["accelerator_usage"] == "forced_cpu"


@pytest.mark.parametrize("accelerator_kind", ["cuda", "mps", "xpu"])
def test_multiturn_lab_auto_selects_detected_accelerator(
    monkeypatch, accelerator_kind: str
) -> None:
    def fake_statuses() -> list[dict[str, object]]:
        return [
            {
                "kind": kind,
                "label": f"{kind} test device"
                if kind == accelerator_kind
                else f"{kind} unavailable",
                "torch_available": True,
                "available": kind == accelerator_kind,
                "usage": "available_for_auto",
            }
            for kind in ("cuda", "mps", "xpu")
        ]

    monkeypatch.setattr(lab_support, "accelerator_statuses", fake_statuses)

    report = lab_support.run_multiturn_lab(device_preference="auto")

    assert report["device"].kind == accelerator_kind
    assert report["device"].label == f"{accelerator_kind} test device"
    assert report["detected_accelerator"].kind == accelerator_kind
    assert report["runtime_policy"]["device_preference"] == "auto"
    assert report["runtime_policy"]["fallback"] is None
    assert report["runtime_policy"]["accelerator_usage"] == "selected_if_available"


def test_multiturn_lab_auto_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(
        lab_support,
        "accelerator_statuses",
        lambda: [
            {
                "kind": kind,
                "label": f"{kind} unavailable",
                "torch_available": False,
                "available": False,
                "usage": "available_for_auto",
            }
            for kind in ("cuda", "mps", "xpu")
        ],
    )

    report = lab_support.run_multiturn_lab(device_preference="auto")

    assert report["device"].kind == "cpu"
    assert report["runtime_policy"]["device_preference"] == "auto"
    assert report["runtime_policy"]["fallback"] == "cpu"
    assert report["runtime_policy"]["accelerator_usage"] == "selected_if_available"


@pytest.mark.parametrize("accelerator_kind", ["cuda", "mps", "xpu"])
def test_multiturn_lab_explicit_accelerator_falls_back_to_cpu_when_unavailable(
    monkeypatch, accelerator_kind: str
) -> None:
    monkeypatch.setattr(
        lab_support,
        "accelerator_statuses",
        lambda: [
            {
                "kind": kind,
                "label": f"{kind} unavailable",
                "torch_available": True,
                "available": False,
                "usage": "available_for_auto",
            }
            for kind in ("cuda", "mps", "xpu")
        ],
    )

    report = lab_support.run_multiturn_lab(device_preference=accelerator_kind)

    assert report["device"].kind == "cpu"
    assert report["runtime_policy"]["device_preference"] == accelerator_kind
    assert report["runtime_policy"]["fallback"] == "cpu"
    assert report["runtime_policy"]["fallback_reason"] == f"{accelerator_kind}_unavailable"


def test_multiturn_lab_rejects_unknown_device_preference() -> None:
    with pytest.raises(ValueError, match="device_preference"):
        lab_support.run_multiturn_lab(device_preference="tpu")


def test_multiturn_lab_reports_each_accelerator_backend_without_using_gpu() -> None:
    report = lab_support.run_multiturn_lab()

    statuses = {status["kind"]: status for status in report["accelerator_report"]}

    assert set(statuses) == {"cuda", "mps", "xpu"}
    assert all(isinstance(status["available"], bool) for status in statuses.values())
    assert all(status["usage"] == "available_for_auto" for status in statuses.values())
    assert all(status["label"] for status in statuses.values())
    assert report["device"].kind in {"cpu", "cuda", "mps", "xpu"}


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
    assert "notebooks.labs.local_multiturn_sql_lab_support" in source
    assert "notebooks.blog_support" not in source
    assert "data_engineering_gates" not in source
    assert "dataset_role_matrix" not in source
    assert "endpoint_run_scorecard" not in source
    assert "planner_scorecard" not in source
    assert "target_evidence_matrix" not in source
    assert "method_decision_rules" not in source
    assert "method_priority_backlog" not in source
    assert "metric_dsl_eval_contract" not in source
    assert "prompt_optimization_findings" not in source
    assert "mo.ui.dropdown" in source
    assert 'value="auto"' in source
    assert "device_preference=runtime_choice.value" in source
    for heading in [
        "## 1. Runtime",
        "## 2. Multi-turn task",
        "## 3. Candidate training targets",
        "## 4. Lab scorecard",
        "## 5. Failure trace",
        "## 6. Intermediate state",
        "## 7. Synthetic fixture pack",
        "## 8. What this proves",
    ]:
        assert heading in source
    assert 'if __name__ == "__main__":' in source
    assert "app.run()" in source


def test_shareable_lab_has_portable_jupyter_notebook_entrypoint() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "labs" / "local_multiturn_sql_lab.ipynb"

    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4
    assert notebook["metadata"]["language_info"]["name"] == "python"

    text = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook["cells"]
    )
    assert "run_multiturn_lab" in text
    assert 'value="auto"' in text
    assert "device_preference=runtime_choice.value" in text
    assert "accelerator_report" in text
    assert "CUDA" in text
    assert "MPS" in text
    assert "XPU" in text
    assert "synthetic_fixture_table" in text
    assert "## 7. Synthetic fixture pack" in text
    assert "notebooks.labs.local_multiturn_sql_lab_support" in text
    assert "notebooks.blog_support" not in text
    assert "endpoint_run_scorecard" not in text
    assert "dataset_role_matrix" not in text
    assert "planner_scorecard" not in text
    assert "target_evidence_matrix" not in text
    assert "method_decision_rules" not in text
    assert "method_priority_backlog" not in text
    assert "metric_dsl_eval_contract" not in text
    assert "prompt_optimization_findings" not in text
    assert "pd.DataFrame(report[\"rows\"])" in text
    assert "next_gates" in text
    assert "pip install" not in text
    assert "apt install" not in text
    assert "notebooks/blog/" not in text

    code = "\n".join(
        "".join(cell.get("source", ""))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert 'value="auto"' in code
    assert "report = run_multiturn_lab(device_preference=runtime_choice.value)" in code
    assert "report[\"accelerator_report\"]" in code
    assert "pd.DataFrame(report[\"method_matrix\"])" in code
    assert "pd.DataFrame(report[\"rows\"])" in code
    assert "pd.DataFrame(report[\"synthetic_fixture_table\"])" in code
    assert "next_gates = pd.DataFrame" in code
    assert "dataset_role_matrix()" not in code
    assert "target_evidence_matrix()" not in code
    assert "method_priority_backlog()" not in code
    assert "endpoint_run_scorecard()" not in code

    namespace: dict[str, object] = {}
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        exec("".join(cell.get("source", "")), namespace)

    report = namespace["report"]
    assert report["device"].kind in {"cpu", "cuda", "mps", "xpu"}
    assert report["runtime_policy"]["accelerator_usage"] == "selected_if_available"
    assert {status["kind"] for status in report["accelerator_report"]} == {
        "cuda",
        "mps",
        "xpu",
    }


def test_shareable_lab_jupyter_notebook_is_tracked_by_git() -> None:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "notebooks/labs/local_multiturn_sql_lab.ipynb",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_jupyter_lab_command_is_backed_by_dev_dependency() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    dev_deps = pyproject["project"]["optional-dependencies"]["dev"]

    assert any(dep.startswith("jupyterlab") for dep in dev_deps)


def test_blog_readme_points_to_shareable_lab_notebook() -> None:
    readme = (REPO_ROOT / "docs" / "blog" / "README.md").read_text()

    assert "attached codebase" in readme
    assert "/labs/local-multiturn-sql-finetuning/" in readme
    assert "source code with Marimo" in readme
    assert "notebooks/labs/local_multiturn_sql_lab.ipynb" in readme
    assert "notebooks/blog/" not in readme
    assert "section notebook" not in readme.lower()
    assert "marimo edit notebooks/labs/local_multiturn_sql_lab.py" in readme
    assert "CPU-safe" in readme
    assert "auto-selects CUDA, MPS, or XPU" in readme
    assert "planner-first" in readme
    assert "semantic-layer" in readme
    assert "MEASURE()" in readme
    assert "behavior/recovery" in readme
