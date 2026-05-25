from __future__ import annotations

import ast
import json
from pathlib import Path

from notebooks.blog_support import (
    accuracy_scorecard,
    claim_table,
    endpoint_run_scorecard,
    export_blog_evidence,
    metric_dsl_demo,
    metric_dsl_eval_contract,
    notebook_walkthrough,
    planner_scorecard,
    semantic_strategy_table,
    target_comparison,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

BLOG_NOTEBOOKS = {
    "docs/blog/01_problem_and_result.md": "notebooks/blog/01_problem_and_result.py",
    "docs/blog/02_wsl_5090_setup.md": "notebooks/blog/02_wsl_5090_setup.py",
    "docs/blog/03_data_and_eval.md": "notebooks/blog/03_data_and_eval.py",
    "docs/blog/04_training_iterations.md": "notebooks/blog/04_training_iterations.py",
    "docs/blog/05_vllm_blackwell_deep_dive.md": "notebooks/blog/05_vllm_blackwell_deep_dive.py",
    "docs/blog/06_data_engineering_for_multiturn_sql_eval.md": "notebooks/blog/06_data_engineering_for_multiturn_sql_eval.py",
}


def test_blog_chapters_link_to_matching_marimo_notebooks() -> None:
    for blog_path, notebook_path in BLOG_NOTEBOOKS.items():
        blog_source = (REPO_ROOT / blog_path).read_text()
        assert f"Notebook: `{notebook_path}`" in blog_source
        assert (REPO_ROOT / notebook_path).exists()


def test_blog_notebooks_are_plain_python_marimo_apps() -> None:
    for notebook_path in BLOG_NOTEBOOKS.values():
        source = (REPO_ROOT / notebook_path).read_text()
        ast.parse(source, filename=notebook_path)
        assert "import marimo" in source
        assert "app = marimo.App" in source
        assert 'if __name__ == "__main__":' in source
        assert "app.run()" in source


def test_training_iteration_notebook_runs_the_finetuning_target_lab() -> None:
    source = (REPO_ROOT / "notebooks/blog/04_training_iterations.py").read_text()

    assert "run_multiturn_lab" in source
    assert "behavior_recovery_sql" in source
    assert "recovery_success_rate" in source


def test_notebook_support_loads_current_artifacts() -> None:
    scores = accuracy_scorecard()
    assert set(scores["mode"]) == {"non_oracle_generation", "oracle_planner_diagnostic"}
    assert scores.loc[scores["run"] == "Base Qwen 3.5 9B", "score"].iloc[0] == 0.37
    assert scores.loc[scores["run"] == "Best non-oracle prompt", "score"].iloc[0] == 0.64
    assert scores.loc[scores["run"] == "Oracle-trained ceiling", "score"].iloc[0] == 0.89

    claims = claim_table()
    assert "rollout_beats_teacher_forced_history" in set(claims["claim_id"])
    assert "metric_dsl_evaluation_manifest" in set(claims["claim_id"])
    assert "metric_dsl_beats_direct_sql" in set(claims["claim_id"])
    rollout = claims[claims["claim_id"] == "rollout_beats_teacher_forced_history"].iloc[0]
    assert rollout["claim_status"] == "pending"
    assert "teacher-forced" in rollout["evidence"]
    metric_comparison = claims[claims["claim_id"] == "metric_dsl_beats_direct_sql"].iloc[0]
    assert metric_comparison["claim_status"] == "pending"
    assert "metric-DSL" in metric_comparison["allowed_public_claim"]

    planner = planner_scorecard()
    assert "column_f1" in set(planner["metric"])
    assert planner.loc[planner["metric"] == "macro_planner_score", "score"].iloc[0] > 0

    strategies = semantic_strategy_table()
    assert "Semantic layer / MEASURE() preservation" in set(strategies["strategy"])

    demo = metric_dsl_demo()
    assert "SUM(orders.amount) AS revenue" in demo["compiled_sql"]
    assert demo["raw_sql_like_score"]["measure_preservation"] == 0.0

    metric_contract = metric_dsl_eval_contract()
    assert set(metric_contract["metric"]) >= {
        "metric_dsl_parse_rate",
        "metric_dsl_compile_rate",
        "measure_preservation",
        "value_execution_accuracy",
        "metric_dsl_value_delta_vs_direct_sql",
    }
    assert set(metric_contract["status"]) == {"pending_manifest", "pending_comparison"}

    walkthrough = notebook_walkthrough()
    assert {
        "checkpoint",
        "reader_question",
        "notebook",
        "evidence_output",
        "claim_boundary",
    } <= set(walkthrough.columns)
    assert "notebooks/labs/local_multiturn_sql_lab.py" in set(walkthrough["notebook"])
    for notebook_path in BLOG_NOTEBOOKS.values():
        assert notebook_path in set(walkthrough["notebook"])
    assert any(
        "single-turn" in question and "multi-turn" in question
        for question in walkthrough["reader_question"]
    )
    assert any("MEASURE()" in boundary for boundary in walkthrough["claim_boundary"])
    training_checkpoint = walkthrough[walkthrough["checkpoint"] == "04 fine-tuning targets"].iloc[0]
    assert "behavior/recovery" in training_checkpoint["reader_question"]
    assert "runnable target lab" in training_checkpoint["evidence_output"]

    targets = target_comparison()
    assert {
        "fine_tuning_target",
        "hypothesis",
        "current_evidence",
        "claim_status",
        "next_gate",
    } <= set(targets.columns)
    assert set(targets["fine_tuning_target"]) == {
        "Direct SQL SFT",
        "Planner/DSL first, SQL second",
        "Semantic-layer tuning",
        "MEASURE()-preserving metric DSL",
        "Behavior/recovery tuning",
    }
    direct = targets[targets["fine_tuning_target"] == "Direct SQL SFT"].iloc[0]
    assert "0.640" in direct["current_evidence"]
    assert direct["claim_status"] == "supported_proxy"
    measure = targets[
        targets["fine_tuning_target"] == "MEASURE()-preserving metric DSL"
    ].iloc[0]
    assert measure["claim_status"] == "pending"
    assert "direct-SQL baseline" in measure["next_gate"]
    recovery = targets[targets["fine_tuning_target"] == "Behavior/recovery tuning"].iloc[0]
    assert "shareable lab" in recovery["current_evidence"]
    assert recovery["claim_status"] == "pending"

    endpoint_runs = endpoint_run_scorecard()
    assert {
        "run",
        "prompt_variant",
        "strict_accuracy",
        "value_accuracy",
        "syntax_accuracy",
        "mean_latency_ms",
    } <= set(endpoint_runs.columns)
    assert "Base Qwen 3.5 9B" in set(endpoint_runs["run"])
    assert "100-step LoRA" in set(endpoint_runs["run"])
    assert (
        endpoint_runs.loc[
            endpoint_runs["run"] == "100-step LoRA", "strict_accuracy"
        ].iloc[0]
        == "0.530"
    )


def test_export_blog_evidence_writes_publishable_assets(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)

    assert manifest["schema_version"] == 1
    assert manifest["source_repo"] == "multiturn-sql-finetuning"
    assert set(manifest["assets"]) == {
        "accuracy_ladder_svg",
        "planner_baseline_svg",
        "claim_table_md",
        "metric_dsl_contract_md",
        "notebook_walkthrough_md",
        "target_comparison_md",
        "endpoint_run_scorecard_md",
    }
    assert set(manifest["source_artifacts"]) >= {
        "docs/claim_ledgers/cosql_dev_100.jsonl",
        "docs/planner_baseline_cosql_dev_100_summary.json",
    }

    accuracy_svg = (tmp_path / manifest["assets"]["accuracy_ladder_svg"]).read_text()
    assert "Best non-oracle prompt" in accuracy_svg
    assert "0.640" in accuracy_svg
    assert "Oracle-trained ceiling" in accuracy_svg

    planner_svg = (tmp_path / manifest["assets"]["planner_baseline_svg"]).read_text()
    assert "Planner baseline" in planner_svg
    assert "column_f1" in planner_svg
    assert "0.117" in planner_svg

    claim_table_md = (tmp_path / manifest["assets"]["claim_table_md"]).read_text()
    assert "metric_dsl_beats_direct_sql" in claim_table_md
    assert "pending" in claim_table_md

    metric_table_md = (tmp_path / manifest["assets"]["metric_dsl_contract_md"]).read_text()
    assert "metric_dsl_value_delta_vs_direct_sql" in metric_table_md
    assert "pending_comparison" in metric_table_md

    walkthrough_md = (tmp_path / manifest["assets"]["notebook_walkthrough_md"]).read_text()
    assert "notebooks/labs/local_multiturn_sql_lab.py" in walkthrough_md
    assert "notebooks/blog/01_problem_and_result.py" in walkthrough_md
    assert "notebooks/blog/06_data_engineering_for_multiturn_sql_eval.py" in walkthrough_md
    assert "MEASURE()" in walkthrough_md
    assert "behavior/recovery" in walkthrough_md
    assert "runnable target lab" in walkthrough_md

    target_md = (tmp_path / manifest["assets"]["target_comparison_md"]).read_text()
    assert "Direct SQL SFT" in target_md
    assert "0.640" in target_md
    assert "MEASURE()-preserving metric DSL" in target_md
    assert "Behavior/recovery tuning" in target_md
    assert "shareable lab" in target_md

    endpoint_md = (tmp_path / manifest["assets"]["endpoint_run_scorecard_md"]).read_text()
    assert "Base Qwen 3.5 9B" in endpoint_md
    assert "100-step LoRA" in endpoint_md
    assert "0.530" in endpoint_md
    assert "0.640" in endpoint_md


def test_checked_in_blog_evidence_assets_are_current(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)
    checked_in_dir = REPO_ROOT / "docs" / "blog" / "generated"

    assert (checked_in_dir / "manifest.json").exists()
    checked_in_manifest = json.loads((checked_in_dir / "manifest.json").read_text())
    assert checked_in_manifest == manifest

    for asset_path in manifest["assets"].values():
        assert (checked_in_dir / asset_path).read_text() == (tmp_path / asset_path).read_text()


def test_research_goal_states_notebook_led_method_comparison() -> None:
    goal = (REPO_ROOT / "docs" / "research_goal.md").read_text()
    blog_readme = (REPO_ROOT / "docs" / "blog" / "README.md").read_text()

    for phrase in [
        "single-turn",
        "multi-turn analytical SQL",
        "small specialized local model",
        "Direct SQL SFT",
        "Planner/DSL first, SQL second",
        "Semantic-layer tuning",
        "MEASURE()-preserving metric DSL",
        "Behavior/recovery tuning",
    ]:
        assert phrase in goal

    assert "Every public claim should name the notebook checkpoint" in goal
    assert "notebook-walkthrough.md" in blog_readme
    assert "notebooks/blog/06_data_engineering_for_multiturn_sql_eval.py" in blog_readme
