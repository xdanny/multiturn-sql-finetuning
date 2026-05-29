from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import notebooks.blog_support as blog_support
from notebooks.blog_support import (
    accuracy_scorecard,
    claim_ledger,
    claim_table,
    data_artifact_contract,
    data_engineering_gates,
    dataset_role_matrix,
    endpoint_run_scorecard,
    export_blog_evidence,
    failure_taxonomy_delta,
    finetuning_step_plan,
    lab_failure_trace,
    lab_method_scorecard,
    lab_reader_flow,
    method_decision_rules,
    method_priority_backlog,
    method_readiness_report,
    metric_dsl_demo,
    metric_dsl_eval_contract,
    planner_readiness_summary,
    planner_scorecard,
    prompt_optimization_findings,
    schema_validation_findings,
    semantic_strategy_table,
    shareable_lab_attachment,
    synthetic_method_fixture_summary,
    target_comparison,
    target_evidence_matrix,
    value_grounding_label_summary,
    value_index_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_LAB_NOTEBOOK = "notebooks/labs/local_multiturn_sql_lab.ipynb"
PUBLIC_LAB_APP = "notebooks/labs/local_multiturn_sql_lab.py"
PUBLIC_LAB_HTML_URL = "/labs/local-multiturn-sql-finetuning/"

FORBIDDEN_SETUP_NOTEBOOKS = {
    "notebooks/blog/02_wsl_5090_setup.py",
    "notebooks/blog/05_vllm_blackwell_deep_dive.py",
}

FORBIDDEN_BLOG_CHAPTER_DOCS = {
    "docs/blog/01_benchmark_gap.md",
    "docs/blog/02_eval_protocol.md",
    "docs/blog/03_method_targets.md",
    "docs/blog/04_results_diagnostics.md",
    "docs/blog/05_next_experiments.md",
}


def manifest_asset_paths(manifest: dict) -> dict[str, str]:
    return {asset["id"]: asset["path"] for asset in manifest["assets"]}


def test_public_blog_artifacts_are_one_shareable_lab_notebook() -> None:
    lab_app = REPO_ROOT / PUBLIC_LAB_APP
    lab_ipynb = REPO_ROOT / PUBLIC_LAB_NOTEBOOK

    app_source = lab_app.read_text()
    ast.parse(app_source, filename=PUBLIC_LAB_APP)
    assert "import marimo" in app_source
    assert "app = marimo.App" in app_source
    assert "run_multiturn_lab" in app_source
    assert "notebooks.labs.local_multiturn_sql_lab_support" in app_source
    assert "notebooks.blog_support" not in app_source
    assert "endpoint_run_scorecard" not in app_source
    assert "dataset_role_matrix" not in app_source
    assert "device_preference=runtime_choice.value" in app_source
    assert 'value="auto"' in app_source
    assert "auto-select" in app_source
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
        assert heading in app_source
    assert 'if __name__ == "__main__":' in app_source
    assert "app.run()" in app_source

    notebook = json.loads(lab_ipynb.read_text())
    assert notebook["nbformat"] == 4
    text = "\n".join("".join(cell.get("source", "")) for cell in notebook["cells"])
    assert "run_multiturn_lab" in text
    assert 'value="auto"' in text
    assert "device_preference=runtime_choice.value" in text
    assert "CUDA" in text
    assert "MPS" in text
    assert "XPU" in text
    assert "synthetic_fixture_table" in text
    assert "## 7. Synthetic fixture pack" in text
    assert "notebooks.labs.local_multiturn_sql_lab_support" in text
    assert "notebooks.blog_support" not in text
    assert "endpoint_run_scorecard" not in text
    assert "failure_taxonomy_delta" not in text
    assert "schema_validation_findings" not in text
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
    for notebook_path in FORBIDDEN_SETUP_NOTEBOOKS:
        assert not (REPO_ROOT / notebook_path).exists()
    for chapter_path in FORBIDDEN_BLOG_CHAPTER_DOCS:
        assert not (REPO_ROOT / chapter_path).exists()


def test_public_blog_artifacts_do_not_publish_section_notebook_series() -> None:
    blog_dir = REPO_ROOT / "notebooks" / "blog"

    assert not list(blog_dir.glob("*.py"))
    assert not (REPO_ROOT / "docs/blog/generated/notebook-series.md").exists()


def test_shareable_lab_runs_the_finetuning_target_lab() -> None:
    source = (REPO_ROOT / PUBLIC_LAB_APP).read_text()
    scores = lab_method_scorecard()

    assert "run_multiturn_lab" in source
    assert set(scores["system"]) == {
        "direct_sql_baseline",
        "planner_first_sql",
        "semantic_value_sql",
        "semantic_dsl_planner",
        "behavior_recovery_sql",
    }
    assert "recovery_success_rate" in set(scores.columns)


def test_notebook_support_loads_current_artifacts() -> None:
    scores = accuracy_scorecard()
    assert set(scores["mode"]) == {"non_oracle_generation", "oracle_planner_diagnostic"}
    assert scores.loc[scores["run"] == "Base Qwen 3.5 9B", "score"].iloc[0] == 0.37
    assert scores.loc[scores["run"] == "Best non-oracle prompt", "score"].iloc[0] == 0.64
    assert scores.loc[scores["run"] == "Oracle-trained ceiling", "score"].iloc[0] == 0.89

    claims = claim_table()
    assert "rollout_beats_teacher_forced_history" in set(claims["claim_id"])
    assert "value_index_coverage" in set(claims["claim_id"])
    assert "semantic_value_retrieval_improves_sql" in set(claims["claim_id"])
    assert "metric-dsl-bootstrap.metric_dsl" in set(claims["claim_id"])
    assert "metric_dsl_beats_direct_sql" in set(claims["claim_id"])
    assert "local_beats_hosted_same_protocol" in set(claims["claim_id"])
    rollout = claims[claims["claim_id"] == "rollout_beats_teacher_forced_history"].iloc[0]
    assert rollout["claim_status"] == "pending"
    assert "teacher-forced" in rollout["evidence"]
    metric_comparison = claims[claims["claim_id"] == "metric_dsl_beats_direct_sql"].iloc[0]
    assert metric_comparison["claim_status"] == "pending"
    assert "metric-DSL" in metric_comparison["allowed_public_claim"]
    metric_quality = claims[claims["claim_id"] == "metric-dsl-bootstrap.metric_dsl"].iloc[0]
    assert metric_quality["claim_status"] == "supported_metric_dsl_quality"
    value_index_claim = claims[claims["claim_id"] == "value_index_coverage"].iloc[0]
    assert value_index_claim["claim_status"] == "supported_value_retrieval_coverage"

    planner = planner_scorecard()
    assert "column_f1" in set(planner["metric"])
    assert planner.loc[planner["metric"] == "macro_planner_score", "score"].iloc[0] > 0

    readiness = planner_readiness_summary()
    assert {
        "artifact",
        "metric",
        "value",
        "interpretation",
    } <= set(readiness.columns)
    assert "planner_readiness_cosql_dev_100.json" in set(readiness["artifact"])
    assert "column_zero_rate" in set(readiness["metric"])
    assert "recommendation" in set(readiness["metric"])

    value_index = value_index_summary()
    assert {
        "artifact",
        "metric",
        "value",
        "interpretation",
    } <= set(value_index.columns)
    assert "value_index_cosql_dev_100_summary.json" in set(value_index["artifact"])
    assert "resolved_value_indexed_rate" in set(value_index["metric"])
    assert "mention_alias_indexed_rate" in set(value_index["metric"])

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

    dataset_roles = dataset_role_matrix()
    assert {
        "data_source",
        "project_role",
        "what_it_tests",
        "why_single_turn_is_not_enough",
        "current_status",
        "next_artifact",
    } <= set(dataset_roles.columns)
    assert set(dataset_roles["data_source"]) == {
        "BIRD-Interact",
        "BIRD mini-dev",
        "CoSQL",
        "SParC",
        "Synthetic schema-rich SQL",
        "Tiny SQLite lab",
    }
    cosql = dataset_roles[dataset_roles["data_source"] == "CoSQL"].iloc[0]
    sparc = dataset_roles[dataset_roles["data_source"] == "SParC"].iloc[0]
    synthetic = dataset_roles[
        dataset_roles["data_source"] == "Synthetic schema-rich SQL"
    ].iloc[0]
    bird_interact = dataset_roles[
        dataset_roles["data_source"] == "BIRD-Interact"
    ].iloc[0]
    assert "current fixed proxy" in cosql["project_role"]
    assert "context-dependent" in sparc["what_it_tests"]
    assert "schema-rich" in synthetic["project_role"]
    assert "north-star" in bird_interact["project_role"]
    assert any(
        "value grounding" in reason
        for reason in dataset_roles["why_single_turn_is_not_enough"]
    )
    assert any(
        "BIRD-Interact transfer" in artifact
        for artifact in dataset_roles["next_artifact"]
    )

    assert hasattr(blog_support, "single_to_multiturn_gap")
    gap = blog_support.single_to_multiturn_gap()
    assert {
        "mechanism",
        "single_turn_assumption",
        "multi_turn_breakage",
        "dataset_surface",
        "repo_artifact_needed",
    } <= set(gap.columns)
    assert set(gap["mechanism"]) == {
        "state carryover",
        "value grounding",
        "grain shift",
        "metric intent",
        "result-aware recovery",
        "generated-history drift",
    }
    assert any("BIRD-style" in assumption for assumption in gap["single_turn_assumption"])
    assert any("CoSQL" in surface for surface in gap["dataset_surface"])
    assert any("SParC" in surface for surface in gap["dataset_surface"])
    assert any("Synthetic schema-rich SQL" in surface for surface in gap["dataset_surface"])
    assert any("BIRD-Interact" in surface for surface in gap["dataset_surface"])
    assert any("France -> FR" in breakage for breakage in gap["multi_turn_breakage"])
    assert any("MEASURE()" in artifact for artifact in gap["repo_artifact_needed"])

    assert hasattr(blog_support, "hosted_comparison_protocol")
    hosted_protocol = blog_support.hosted_comparison_protocol()
    assert {
        "protocol_gate",
        "required_evidence",
        "why_it_matters",
        "minimum_acceptance",
        "claim_ids",
    } <= set(hosted_protocol.columns)
    assert set(hosted_protocol["protocol_gate"]) == {
        "same input rows",
        "same scorer and output schema",
        "same oracle boundary",
        "hosted model manifest",
        "local model manifest",
        "generated-history rollout",
        "latency and cost",
        "BIRD-Interact transfer",
    }
    assert any(
        "positive local-vs-hosted delta" in acceptance
        for acceptance in hosted_protocol["minimum_acceptance"]
    )
    assert any(
        "hosted_sota_same_protocol" in claim_ids
        for claim_ids in hosted_protocol["claim_ids"]
    )

    assert hasattr(blog_support, "evaluation_harness_map")
    harness = blog_support.evaluation_harness_map()
    assert {
        "research_target",
        "module_path",
        "command_surface",
        "implemented_gate",
        "current_status",
        "claim_ids",
    } <= set(harness.columns)
    assert set(harness["research_target"]) >= {
        "direct SQL control",
        "planner quality before SQL",
        "predicted-planner SQL",
        "semantic/value artifacts",
        "MEASURE()-preserving DSL",
        "generated-history rollout",
        "hosted/local comparison",
    }
    for module_path in harness["module_path"]:
        assert (REPO_ROOT / module_path).exists(), module_path
    assert any("eval.run_eval" in command for command in harness["command_surface"])
    assert any("eval.planner_optimize" in command for command in harness["command_surface"])
    assert any("eval.metric_dsl_eval" in command for command in harness["command_surface"])
    assert any("eval.rollout_eval" in command for command in harness["command_surface"])
    assert any("eval.compare_hosted_baseline" in command for command in harness["command_surface"])
    assert any(
        "same scorer" in gate and "same rows" in gate
        for gate in harness["implemented_gate"]
    )
    assert any(
        "metric_dsl_beats_direct_sql" in claim_ids
        for claim_ids in harness["claim_ids"]
    )

    readiness_report = method_readiness_report()
    assert {
        "method",
        "readiness_level",
        "control_ready_now",
        "rankable_now",
        "supported_claim_ids",
        "blocking_claim_ids",
        "module_path",
        "next_command",
        "next_artifact",
        "claim_boundary",
    } <= set(readiness_report.columns)
    assert list(readiness_report["method"]) == [
        "Direct SQL SFT",
        "Planner/DSL first, SQL second",
        "Semantic-layer tuning",
        "MEASURE()-preserving metric DSL",
        "Behavior/recovery tuning",
        "Hosted and BIRD-Interact comparison",
    ]
    direct_readiness = readiness_report[
        readiness_report["method"] == "Direct SQL SFT"
    ].iloc[0]
    planner_readiness = readiness_report[
        readiness_report["method"] == "Planner/DSL first, SQL second"
    ].iloc[0]
    assert direct_readiness["readiness_level"] == "control_ready"
    assert direct_readiness["control_ready_now"] is True
    assert direct_readiness["rankable_now"] is False
    assert direct_readiness["supported_claim_ids"] == [
        "qwen35_9b_base_cosql_dev_100turns",
        "multiturn_sql_100_cosql_dev_100turns",
    ]
    assert (
        "semantic_prompt_minimal_executable_cosql_dev_100turns"
        not in direct_readiness["supported_claim_ids"]
    )
    semantic_readiness = readiness_report[
        readiness_report["method"] == "Semantic-layer tuning"
    ].iloc[0]
    assert semantic_readiness["supported_claim_ids"] == [
        "semantic_prompt_minimal_executable_cosql_dev_100turns",
        "value_index_coverage",
    ]
    assert "semantic_value_retrieval_improves_sql" in semantic_readiness[
        "blocking_claim_ids"
    ]
    assert planner_readiness["readiness_level"] == "needs_endpoint_comparison"
    assert planner_readiness["control_ready_now"] is False
    assert planner_readiness["rankable_now"] is False
    assert "eval.run_predicted_planner_comparison" in planner_readiness["next_command"]
    assert any(
        "eval.metric_dsl_eval" in command for command in readiness_report["next_command"]
    )
    assert any(
        "eval.rollout_eval" in command for command in readiness_report["next_command"]
    )
    assert all((REPO_ROOT / path).exists() for path in readiness_report["module_path"])

    step_plan = finetuning_step_plan()
    assert set(step_plan["step_id"]) == {
        "direct_sql_control_smoke",
        "planner_first_sql_pair",
        "semantic_value_retrieval_pair",
        "metric_dsl_vs_direct_sql",
        "behavior_recovery_rollout_pair",
        "hosted_bird_interact_gate",
    }
    assert "metric_dsl_beats_direct_sql" in set(step_plan["clears_claim_ids"])
    assert any(
        "semantic_value_retrieval_improves_sql" in claim_ids
        for claim_ids in step_plan["clears_claim_ids"]
    )
    assert step_plan["cheap_preflight_available"].all()
    assert all("train=" in rows_ready for rows_ready in step_plan["rows_ready"])

    lab_attachment = shareable_lab_attachment()
    assert {
        "artifact",
        "notebook",
        "repo_url",
        "published_html_url",
        "run_command",
        "alternate_command",
        "device_policy",
        "reader_flow",
        "what_runs",
        "claim_boundary",
    } <= set(lab_attachment.columns)
    lab_row = lab_attachment.iloc[0]
    assert "shareable lab notebook and attached codebase" in lab_row["artifact"]
    assert lab_row["notebook"] == PUBLIC_LAB_APP
    assert "github.com/xdanny/multiturn-sql-finetuning" in lab_row["repo_url"]
    assert lab_row["published_html_url"] == PUBLIC_LAB_HTML_URL
    assert lab_row["run_command"] == f"marimo edit {PUBLIC_LAB_APP}"
    assert lab_row["alternate_command"] == f"jupyter lab {PUBLIC_LAB_NOTEBOOK}"
    assert "auto-selects CUDA, MPS, or XPU" in lab_row["device_policy"]
    assert "falls back to CPU" in lab_row["device_policy"]
    assert "CUDA" in lab_row["device_policy"]
    assert "MPS" in lab_row["device_policy"]
    assert "XPU" in lab_row["device_policy"]
    assert "Research question" in lab_row["reader_flow"]
    assert "open the published HTML lab" in lab_row["reader_flow"]
    assert "run the lab checkpoints" in lab_row["reader_flow"]
    assert "compare fine-tuning targets" in lab_row["reader_flow"]
    assert "read the evidence gates" in lab_row["reader_flow"]
    assert "chapter" not in lab_row["artifact"]
    assert "section notebooks" not in lab_row["reader_flow"]

    reader_flow = lab_reader_flow()
    assert {
        "step",
        "lab_step",
        "notebook",
        "run_command",
        "alternate_command",
        "reader_action",
        "evidence_to_inspect",
        "claim_boundary",
    } <= set(reader_flow.columns)
    assert "post_section" not in set(reader_flow.columns)
    assert list(reader_flow["step"]) == list(range(1, len(reader_flow) + 1))
    assert "Benchmark gap" in set(reader_flow["lab_step"])
    assert "Evaluation protocol" in set(reader_flow["lab_step"])
    assert "Fine-tuning targets" in set(reader_flow["lab_step"])
    assert "Results and diagnostics" in set(reader_flow["lab_step"])
    assert "Next experiments" in set(reader_flow["lab_step"])
    assert any("single-turn" in action for action in reader_flow["reader_action"])
    assert any("not a benchmark result" in boundary for boundary in reader_flow["claim_boundary"])
    assert set(reader_flow["notebook"]) == {PUBLIC_LAB_APP}
    assert all(reader_flow["run_command"] == f"marimo edit {PUBLIC_LAB_APP}")
    assert all(reader_flow["alternate_command"] == f"jupyter lab {PUBLIC_LAB_NOTEBOOK}")

    decision_rules = method_decision_rules()
    assert {
        "fine_tuning_target",
        "control_arm",
        "win_condition",
        "required_comparison",
        "current_blocker",
        "claim_ids",
    } <= set(decision_rules.columns)
    assert set(decision_rules["fine_tuning_target"]) == {
        "Direct SQL SFT",
        "Planner/DSL first, SQL second",
        "Semantic-layer tuning",
        "MEASURE()-preserving metric DSL",
        "Behavior/recovery tuning",
    }
    assert all(
        "same rows" in comparison and "same scorer" in comparison
        for comparison in decision_rules["required_comparison"]
    )
    assert any(
        "hosted_sota_same_protocol" in claim_ids
        for claim_ids in decision_rules["claim_ids"]
    )
    assert any(
        "local_beats_hosted_same_protocol" in claim_ids
        for claim_ids in decision_rules["claim_ids"]
    )
    assert any("non-oracle planner" in blocker for blocker in decision_rules["current_blocker"])
    assert any("generated-history" in blocker for blocker in decision_rules["current_blocker"])

    known_claim_ids = set(claim_ledger()["claim_id"])
    decision_claim_ids = {
        claim_id.strip()
        for claim_ids in decision_rules["claim_ids"]
        for claim_id in claim_ids.split(",")
        if claim_id.strip()
    }
    assert decision_claim_ids <= known_claim_ids

    priority = method_priority_backlog()
    assert {
        "priority",
        "fine_tuning_target",
        "why_now",
        "dataset_focus",
        "success_metric",
        "build_next",
        "falsifies_if",
        "claim_ids",
    } <= set(priority.columns)
    assert list(priority["priority"]) == [1, 2, 3, 4, 5]
    assert priority.iloc[0]["fine_tuning_target"] == "Planner/DSL first, SQL second"
    assert "oracle gap" in priority.iloc[0]["why_now"]
    assert "CoSQL" in priority.iloc[0]["dataset_focus"]
    assert any(
        row["fine_tuning_target"] == "MEASURE()-preserving metric DSL"
        and "metric-heavy" in row["dataset_focus"]
        and "measure_preservation" in row["success_metric"]
        for _, row in priority.iterrows()
    )
    assert any(
        row["fine_tuning_target"] == "Behavior/recovery tuning"
        and "generated-history" in row["build_next"]
        for _, row in priority.iterrows()
    )
    assert any(
        row["fine_tuning_target"] == "Semantic-layer tuning"
        and "value/entity" in row["build_next"]
        for _, row in priority.iterrows()
    )
    priority_claim_ids = {
        claim_id.strip()
        for claim_ids in priority["claim_ids"]
        for claim_id in claim_ids.split(",")
        if claim_id.strip()
    }
    assert priority_claim_ids <= known_claim_ids

    assert hasattr(blog_support, "experiment_ladder")
    ladder = blog_support.experiment_ladder()
    assert {
        "rung",
        "experiment",
        "question",
        "required_artifact",
        "claim_gate",
    } <= set(ladder.columns)
    assert list(ladder["rung"]) == list(range(1, 8))
    assert list(ladder["experiment"]) == [
        "Direct SQL control",
        "Planner quality before SQL",
        "Predicted-planner SQL",
        "Semantic-layer target",
        "MEASURE()-preserving DSL",
        "Generated-history rollout",
        "Hosted and BIRD-Interact comparison",
    ]
    assert "same local endpoint" in ladder.iloc[0]["question"]
    assert "score the plan before SQL" in ladder.iloc[1]["claim_gate"]
    assert "non-oracle predicted plans" in ladder.iloc[2]["question"]
    assert "governed entities" in ladder.iloc[3]["question"]
    assert "MEASURE()" in ladder.iloc[4]["question"]
    assert "teacher-forced" in ladder.iloc[5]["claim_gate"]
    assert "hosted baselines" in ladder.iloc[6]["required_artifact"]
    assert "BIRD-Interact" in ladder.iloc[6]["claim_gate"]

    lab_scores = lab_method_scorecard()
    assert {
        "system",
        "fine_tuning_target",
        "value_accuracy",
        "context_carryover_accuracy",
        "value_grounding_accuracy",
        "measure_preservation_rate",
        "recovery_success_rate",
        "lab_takeaway",
    } <= set(lab_scores.columns)
    assert set(lab_scores["system"]) == {
        "direct_sql_baseline",
        "planner_first_sql",
        "semantic_value_sql",
        "semantic_dsl_planner",
        "behavior_recovery_sql",
    }
    direct_lab_score = lab_scores[
        lab_scores["system"] == "direct_sql_baseline"
    ].iloc[0]
    dsl_lab_score = lab_scores[
        lab_scores["system"] == "semantic_dsl_planner"
    ].iloc[0]
    recovery_lab_score = lab_scores[
        lab_scores["system"] == "behavior_recovery_sql"
    ].iloc[0]
    assert direct_lab_score["value_accuracy"] == 0.25
    assert dsl_lab_score["value_accuracy"] == 1.0
    assert dsl_lab_score["measure_preservation_rate"] == 1.0
    assert recovery_lab_score["recovery_success_rate"] == 1.0
    assert all("not a benchmark result" in takeaway for takeaway in lab_scores["lab_takeaway"])

    lab_trace = lab_failure_trace()
    assert {
        "turn_id",
        "question",
        "system",
        "failure_type",
        "requires_recovery",
        "value_match",
        "recovery_success",
        "actual_rows",
        "expected_rows",
        "intermediate_plan",
        "why_it_matters",
    } <= set(lab_trace.columns)
    trace_keys = {(row["turn_id"], row["system"]) for _, row in lab_trace.iterrows()}
    assert ("turn_2", "direct_sql_baseline") in trace_keys
    assert ("turn_3", "direct_sql_baseline") in trace_keys
    assert ("turn_4", "behavior_recovery_sql") in trace_keys
    assert {
        "direct_sql_baseline",
        "planner_first_sql",
        "semantic_value_sql",
        "semantic_dsl_planner",
        "behavior_recovery_sql",
    } <= set(lab_trace.loc[lab_trace["turn_id"] == "turn_4", "system"])
    assert "value_grounding" in set(lab_trace["failure_type"])
    assert "context_carryover" in set(lab_trace["failure_type"])
    assert any(
        row["value_match"] is True and row["recovery_success"] is False
        for _, row in lab_trace[lab_trace["turn_id"] == "turn_4"].iterrows()
    )
    assert any(
        row["value_match"] is True and row["recovery_success"] is True
        for _, row in lab_trace[lab_trace["turn_id"] == "turn_4"].iterrows()
    )
    assert any("France -> FR" in note for note in lab_trace["why_it_matters"])
    assert any("repairs empty result" in plan for plan in lab_trace["intermediate_plan"])

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
    semantic = targets[targets["fine_tuning_target"] == "Semantic-layer tuning"].iloc[0]
    assert "0.530" in direct["current_evidence"]
    assert "0.640" not in direct["current_evidence"]
    assert direct["claim_status"] == "supported_proxy"
    assert "0.640" in semantic["current_evidence"]
    measure = targets[
        targets["fine_tuning_target"] == "MEASURE()-preserving metric DSL"
    ].iloc[0]
    assert measure["claim_status"] == "pending"
    assert "direct-SQL baseline" in measure["next_gate"]
    recovery = targets[targets["fine_tuning_target"] == "Behavior/recovery tuning"].iloc[0]
    assert "shareable lab" in recovery["current_evidence"]
    assert recovery["claim_status"] == "pending"

    evidence_matrix = target_evidence_matrix()
    assert {
        "fine_tuning_target",
        "lab_behavior",
        "manifest_backed_evidence",
        "source_claim_ids",
        "evidence_level",
        "missing_gate",
        "current_decision",
    } <= set(evidence_matrix.columns)
    assert set(evidence_matrix["fine_tuning_target"]) == {
        "Direct SQL SFT",
        "Planner/DSL first, SQL second",
        "Semantic-layer tuning",
        "MEASURE()-preserving metric DSL",
        "Behavior/recovery tuning",
    }
    planner_row = evidence_matrix[
        evidence_matrix["fine_tuning_target"] == "Planner/DSL first, SQL second"
    ].iloc[0]
    metric_row = evidence_matrix[
        evidence_matrix["fine_tuning_target"] == "MEASURE()-preserving metric DSL"
    ].iloc[0]
    direct_row = evidence_matrix[
        evidence_matrix["fine_tuning_target"] == "Direct SQL SFT"
    ].iloc[0]
    assert "macro=0.571" in planner_row["manifest_backed_evidence"]
    assert "predicted_planner_sql_execution" in planner_row["source_claim_ids"]
    assert "pending" in planner_row["evidence_level"]
    assert "bootstrap metric-DSL manifest" in metric_row["manifest_backed_evidence"]
    assert "supported_metric_dsl_quality" in metric_row["evidence_level"]
    assert "0.530 strict" in direct_row["manifest_backed_evidence"]
    assert "baseline every structured target must beat" in direct_row["current_decision"]

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

    failure_delta = failure_taxonomy_delta()
    assert {
        "candidate",
        "baseline",
        "value_accuracy",
        "fixed_turns",
        "regressed_turns",
        "net_fixed",
        "fixed_error_primary",
        "regressed_error_primary",
        "takeaway",
    } <= set(failure_delta.columns)
    best_delta = failure_delta[failure_delta["candidate"] == "multiturn-sql-semantic-50[minimal_executable]"].iloc[0]
    assert best_delta["net_fixed"] == 5
    assert "value grounding" in best_delta["takeaway"]
    assert "projection" in best_delta["takeaway"]

    schema_findings = schema_validation_findings()
    assert {
        "run",
        "source_artifact",
        "schema_mismatch_rows",
        "unknown_column",
        "wrong_table_column",
        "ambiguous_unqualified_column",
        "example_turn",
        "example_diagnostic",
    } <= set(schema_findings.columns)
    assert any(schema_findings["wrong_table_column"] > 0)
    assert any("repairable" in diagnostic for diagnostic in schema_findings["example_diagnostic"])

    gates = data_engineering_gates()
    assert {
        "gate",
        "artifact",
        "problem_exposed",
        "why_it_matters",
        "current_status",
        "next_repo_action",
        "blocks_claim",
        "source_artifacts",
        "claim_ids",
    } <= set(gates.columns)
    assert set(gates["gate"]) >= {
        "fixed_proxy_slice",
        "generated_history_rollout",
        "value_entity_normalization",
        "join_fanout_fixtures",
        "alias_schema_validation",
        "semantic_metric_manifest",
        "hosted_same_protocol_baseline",
        "bird_interact_transfer",
    }
    assert any("teacher-forced" in status for status in gates["current_status"])
    assert any("France -> FR" in problem for problem in gates["problem_exposed"])
    assert any("bridge tables" in problem for problem in gates["problem_exposed"])
    assert any("BIRD-Interact" in action for action in gates["next_repo_action"])
    assert all(gates["blocks_claim"].str.len() > 0)
    assert all(gates["source_artifacts"].str.len() > 0)
    assert all(gates["claim_ids"].str.len() > 0)

    synthetic_fixtures = synthetic_method_fixture_summary()
    assert {
        "artifact",
        "metric",
        "value",
        "interpretation",
    } <= set(synthetic_fixtures.columns)
    assert "synthetic_method_fixtures_summary.json" in set(synthetic_fixtures["artifact"])
    assert synthetic_fixtures.loc[
        synthetic_fixtures["metric"] == "fixture_count",
        "value",
    ].iloc[0] == 5
    assert any(
        row["metric"] == "failure_modes" and "grain_fanout" in row["value"]
        for _, row in synthetic_fixtures.iterrows()
    )
    assert any(
        row["metric"] == "training_targets" and "metric_dsl" in row["value"]
        for _, row in synthetic_fixtures.iterrows()
    )

    known_claim_ids = set(claim_ledger()["claim_id"])
    gate_claim_ids = {
        claim_id.strip()
        for claim_ids in gates["claim_ids"]
        for claim_id in claim_ids.split(",")
        if claim_id.strip()
    }
    assert gate_claim_ids <= known_claim_ids
    assert {
        "rollout_beats_teacher_forced_history",
        "metric-dsl-bootstrap.metric_dsl",
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
    } <= gate_claim_ids

    value_summary = value_grounding_label_summary()
    assert {
        "artifact",
        "metric",
        "value",
        "interpretation",
    } <= set(value_summary.columns)
    assert "value_grounding_labels_cosql_dev_100.jsonl" in set(value_summary["artifact"])
    assert value_summary.loc[
        value_summary["metric"] == "value_reference_count",
        "value",
    ].iloc[0] > 0
    assert "missing_from_user_text_count" in set(value_summary["metric"])

    artifact_contract = data_artifact_contract()
    assert {
        "artifact",
        "failure_isolated",
        "labels_or_fields",
        "consumer",
        "verification_gate",
        "claim_ids",
    } <= set(artifact_contract.columns)
    assert set(artifact_contract["artifact"]) >= {
        "value_index",
        "entity_resolution_labels",
        "grain_fanout_fixtures",
        "semantic_model_manifest",
        "schema_alias_validator",
        "generated_history_trace",
    }
    value_index = artifact_contract[
        artifact_contract["artifact"] == "value_index"
    ].iloc[0]
    semantic_manifest = artifact_contract[
        artifact_contract["artifact"] == "semantic_model_manifest"
    ].iloc[0]
    rollout_trace = artifact_contract[
        artifact_contract["artifact"] == "generated_history_trace"
    ].iloc[0]
    assert "France -> FR" in value_index["failure_isolated"]
    assert "aliases" in value_index["labels_or_fields"]
    assert "MEASURE()" in semantic_manifest["labels_or_fields"]
    assert "teacher-forced" in rollout_trace["failure_isolated"]
    assert all("same rows" in gate or "manifest" in gate for gate in artifact_contract["verification_gate"])
    artifact_claim_ids = {
        claim_id.strip()
        for claim_ids in artifact_contract["claim_ids"]
        for claim_id in claim_ids.split(",")
        if claim_id.strip()
    }
    assert artifact_claim_ids <= known_claim_ids

    prompt_findings = prompt_optimization_findings()
    assert {
        "optimization_scope",
        "source_artifact",
        "best_variant",
        "best_source",
        "best_accuracy",
        "best_dspy_accuracy",
        "promoted_decision",
        "next_program_target",
        "claim_boundary",
    } <= set(prompt_findings.columns)
    assert set(prompt_findings["optimization_scope"]) == {
        "non_oracle_sql_prompt_smoke",
        "oracle_schema_pruned_prompt_search",
        "planner_program_optimization_gate",
    }
    smoke = prompt_findings[
        prompt_findings["optimization_scope"] == "non_oracle_sql_prompt_smoke"
    ].iloc[0]
    oracle = prompt_findings[
        prompt_findings["optimization_scope"] == "oracle_schema_pruned_prompt_search"
    ].iloc[0]
    planner_gate = prompt_findings[
        prompt_findings["optimization_scope"] == "planner_program_optimization_gate"
    ].iloc[0]
    assert smoke["best_accuracy"] == 0.4
    assert smoke["best_dspy_accuracy"] == 0.4
    assert "tie" in smoke["promoted_decision"]
    assert oracle["best_accuracy"] == 0.85
    assert oracle["best_dspy_accuracy"] == 0.83
    assert oracle["best_variant"] == "schema_pruned_minimal"
    assert "static" in oracle["promoted_decision"]
    assert planner_gate["best_accuracy"] is None
    assert "harness_ready" in planner_gate["promoted_decision"]
    assert "eval.planner_optimize" in planner_gate["promoted_decision"]
    assert "DSPy proposals" in planner_gate["next_program_target"]
    assert "eval.planner_predict" in planner_gate["next_program_target"]
    assert all("not a SOTA claim" in boundary for boundary in prompt_findings["claim_boundary"])


def test_export_blog_evidence_writes_publishable_assets(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)
    asset_paths = manifest_asset_paths(manifest)

    assert manifest["schema_version"] == 2
    assert manifest["source_repo"] == "multiturn-sql-finetuning"
    assert set(asset_paths) == {
        "accuracy_ladder_svg",
        "planner_baseline_svg",
        "claim_table_md",
        "dataset_role_matrix_md",
        "metric_dsl_contract_md",
        "shareable_lab_md",
        "lab_reader_flow_md",
        "single_to_multiturn_gap_md",
        "hosted_comparison_protocol_md",
        "evaluation_harness_map_md",
        "experiment_ladder_md",
        "method_readiness_report_md",
        "finetuning_step_plan_md",
        "method_decision_rules_md",
        "method_priority_backlog_md",
        "lab_method_scores_md",
        "lab_failure_trace_md",
        "data_engineering_gates_md",
        "synthetic_method_fixtures_md",
        "value_grounding_labels_md",
        "value_index_md",
        "planner_readiness_md",
        "prompt_optimization_findings_md",
        "data_artifact_contract_md",
        "target_comparison_md",
        "target_evidence_matrix_md",
        "endpoint_run_scorecard_md",
        "failure_taxonomy_delta_md",
        "schema_validation_findings_md",
        "gpu_finetuning_evidence_md",
    }
    assert {source["path"] for source in manifest["source_artifacts"]} >= {
        "configs/benchmark_protocols.yaml",
        "configs/finetuning_methods.yaml",
        "configs/finetuning_steps.yaml",
        "docs/claim_ledgers/cosql_dev_100.jsonl",
        "docs/data_artifacts/value_grounding_labels_cosql_dev_100.manifest.json",
        "docs/data_artifacts/value_index_cosql_dev_100.manifest.json",
        "docs/data_artifacts/synthetic_method_fixtures.manifest.json",
        "docs/training_runs/gpu_finetuning_evidence.json",
        "docs/predicted_planner_comparison_preflight.json",
        "docs/planner_readiness_cosql_dev_100.json",
        "docs/planner_baseline_cosql_dev_100_summary.json",
    }

    accuracy_svg = (tmp_path / asset_paths["accuracy_ladder_svg"]).read_text()
    assert "Best non-oracle prompt" in accuracy_svg
    assert "0.640" in accuracy_svg
    assert "Oracle-trained ceiling" in accuracy_svg

    planner_svg = (tmp_path / asset_paths["planner_baseline_svg"]).read_text()
    assert "Planner baseline" in planner_svg
    assert "column_f1" in planner_svg
    assert "0.117" in planner_svg

    planner_readiness_md = (
        tmp_path / asset_paths["planner_readiness_md"]
    ).read_text()
    assert "planner_readiness_cosql_dev_100.json" in planner_readiness_md
    assert "column_zero_rate" in planner_readiness_md
    assert "empty_projection_expression_rate" in planner_readiness_md
    assert "improve_planner_before_claim" in planner_readiness_md
    assert "readiness only; no SQL execution claim" in planner_readiness_md

    gpu_evidence_md = (
        tmp_path / asset_paths["gpu_finetuning_evidence_md"]
    ).read_text()
    assert "multiturn-sql-100" in gpu_evidence_md
    assert "0.63" in gpu_evidence_md
    assert "completed_smoke" in gpu_evidence_md
    assert "cuda_available True" in gpu_evidence_md
    assert "RTX 5090" in gpu_evidence_md

    claim_table_md = (tmp_path / asset_paths["claim_table_md"]).read_text()
    assert "metric_dsl_beats_direct_sql" in claim_table_md
    assert "pending" in claim_table_md

    dataset_roles_md = (
        tmp_path / asset_paths["dataset_role_matrix_md"]
    ).read_text()
    assert "BIRD-Interact" in dataset_roles_md
    assert "BIRD mini-dev" in dataset_roles_md
    assert "CoSQL" in dataset_roles_md
    assert "SParC" in dataset_roles_md
    assert "Synthetic schema-rich SQL" in dataset_roles_md
    assert "Tiny SQLite lab" in dataset_roles_md
    assert "value grounding" in dataset_roles_md
    assert "schema-rich" in dataset_roles_md

    metric_table_md = (tmp_path / asset_paths["metric_dsl_contract_md"]).read_text()
    assert "metric_dsl_value_delta_vs_direct_sql" in metric_table_md
    assert "pending_comparison" in metric_table_md

    shareable_lab_md = (tmp_path / asset_paths["shareable_lab_md"]).read_text()
    assert "shareable lab notebook and attached codebase" in shareable_lab_md
    assert PUBLIC_LAB_HTML_URL in shareable_lab_md
    assert PUBLIC_LAB_NOTEBOOK in shareable_lab_md
    assert PUBLIC_LAB_APP in shareable_lab_md
    assert f"marimo edit {PUBLIC_LAB_APP}" in shareable_lab_md
    assert f"jupyter lab {PUBLIC_LAB_NOTEBOOK}" in shareable_lab_md
    assert "auto-selects CUDA, MPS, or XPU" in shareable_lab_md
    assert "falls back to CPU" in shareable_lab_md
    assert "CUDA" in shareable_lab_md
    assert "MPS" in shareable_lab_md
    assert "XPU" in shareable_lab_md
    assert "Research question" in shareable_lab_md
    assert "open the published HTML lab" in shareable_lab_md
    assert "run the lab checkpoints" in shareable_lab_md
    assert "compare fine-tuning targets" in shareable_lab_md
    assert "read the evidence gates" in shareable_lab_md
    assert "chapter" not in shareable_lab_md
    assert "section notebooks" not in shareable_lab_md

    lab_flow_md = (tmp_path / asset_paths["lab_reader_flow_md"]).read_text()
    assert PUBLIC_LAB_NOTEBOOK in lab_flow_md
    assert PUBLIC_LAB_APP in lab_flow_md
    assert "lab_step" in lab_flow_md
    assert "post_section" not in lab_flow_md
    assert "Research question" in lab_flow_md
    assert "Benchmark gap" in lab_flow_md
    assert "Fine-tuning targets" in lab_flow_md
    assert "not a benchmark result" in lab_flow_md
    assert "notebooks/blog/" not in lab_flow_md

    gap_md = (tmp_path / asset_paths["single_to_multiturn_gap_md"]).read_text()
    assert "state carryover" in gap_md
    assert "value grounding" in gap_md
    assert "grain shift" in gap_md
    assert "metric intent" in gap_md
    assert "result-aware recovery" in gap_md
    assert "generated-history drift" in gap_md
    assert "BIRD-style" in gap_md
    assert "CoSQL" in gap_md
    assert "SParC" in gap_md
    assert "Synthetic schema-rich SQL" in gap_md
    assert "BIRD-Interact" in gap_md
    assert "France -> FR" in gap_md
    assert "MEASURE()" in gap_md
    assert "notebooks/blog/" not in gap_md

    hosted_protocol_md = (
        tmp_path / asset_paths["hosted_comparison_protocol_md"]
    ).read_text()
    assert "same input rows" in hosted_protocol_md
    assert "hosted model manifest" in hosted_protocol_md
    assert "local model manifest" in hosted_protocol_md
    assert "latency and cost" in hosted_protocol_md
    assert "generated-history rollout" in hosted_protocol_md
    assert "BIRD-Interact transfer" in hosted_protocol_md
    assert "positive local-vs-hosted delta" in hosted_protocol_md
    assert "notebooks/blog/" not in hosted_protocol_md

    harness_md = (tmp_path / asset_paths["evaluation_harness_map_md"]).read_text()
    assert "direct SQL control" in harness_md
    assert "planner quality before SQL" in harness_md
    assert "predicted-planner SQL" in harness_md
    assert "semantic/value artifacts" in harness_md
    assert "MEASURE()-preserving DSL" in harness_md
    assert "generated-history rollout" in harness_md
    assert "hosted/local comparison" in harness_md
    assert "eval.run_eval" in harness_md
    assert "eval.planner_optimize" in harness_md
    assert "eval.metric_dsl_eval" in harness_md
    assert "eval.rollout_eval" in harness_md
    assert "eval.compare_hosted_baseline" in harness_md
    assert "notebooks/blog/" not in harness_md

    method_readiness_md = (
        tmp_path / asset_paths["method_readiness_report_md"]
    ).read_text()
    assert "Direct SQL SFT" in method_readiness_md
    assert "Planner/DSL first, SQL second" in method_readiness_md
    assert "needs_endpoint_comparison" in method_readiness_md
    assert "eval.run_predicted_planner_comparison" in method_readiness_md
    assert "eval.metric_dsl_eval" in method_readiness_md
    assert "eval.rollout_eval" in method_readiness_md
    assert "notebooks/blog/" not in method_readiness_md

    finetuning_steps_md = (
        tmp_path / asset_paths["finetuning_step_plan_md"]
    ).read_text()
    assert "metric_dsl_vs_direct_sql" in finetuning_steps_md
    assert "behavior_recovery_rollout_pair" in finetuning_steps_md
    assert "semantic_value_retrieval_pair" in finetuning_steps_md
    assert "synthetic_schema_rich_method_fixture" in finetuning_steps_md
    assert "cheap_preflight_available" in finetuning_steps_md
    assert "metric_dsl_beats_direct_sql" in finetuning_steps_md
    assert "hosted_bird_interact_gate" in finetuning_steps_md
    assert "notebooks/blog/" not in finetuning_steps_md

    experiment_ladder_md = (
        tmp_path / asset_paths["experiment_ladder_md"]
    ).read_text()
    assert "Direct SQL control" in experiment_ladder_md
    assert "Planner quality before SQL" in experiment_ladder_md
    assert "Predicted-planner SQL" in experiment_ladder_md
    assert "Semantic-layer target" in experiment_ladder_md
    assert "MEASURE()-preserving DSL" in experiment_ladder_md
    assert "Generated-history rollout" in experiment_ladder_md
    assert "Hosted and BIRD-Interact comparison" in experiment_ladder_md
    assert "score the plan before SQL" in experiment_ladder_md
    assert "same local endpoint" in experiment_ladder_md
    assert "notebooks/blog/" not in experiment_ladder_md

    decision_rules_md = (
        tmp_path / asset_paths["method_decision_rules_md"]
    ).read_text()
    assert "Planner/DSL first, SQL second" in decision_rules_md
    assert "MEASURE()-preserving metric DSL" in decision_rules_md
    assert "same rows" in decision_rules_md
    assert "same scorer" in decision_rules_md
    assert "hosted_sota_same_protocol" in decision_rules_md
    assert "local_beats_hosted_same_protocol" in decision_rules_md
    assert "generated-history" in decision_rules_md

    priority_md = (
        tmp_path / asset_paths["method_priority_backlog_md"]
    ).read_text()
    assert "Planner/DSL first, SQL second" in priority_md
    assert "oracle gap" in priority_md
    assert "MEASURE()-preserving metric DSL" in priority_md
    assert "metric-heavy" in priority_md
    assert "Behavior/recovery tuning" in priority_md
    assert "generated-history" in priority_md

    lab_scores_md = (tmp_path / asset_paths["lab_method_scores_md"]).read_text()
    assert "direct_sql_baseline" in lab_scores_md
    assert "semantic_dsl_planner" in lab_scores_md
    assert "behavior_recovery_sql" in lab_scores_md
    assert "0.250" in lab_scores_md
    assert "1.000" in lab_scores_md
    assert "not a benchmark result" in lab_scores_md

    lab_trace_md = (tmp_path / asset_paths["lab_failure_trace_md"]).read_text()
    assert "turn_2" in lab_trace_md
    assert "turn_3" in lab_trace_md
    assert "turn_4" in lab_trace_md
    assert "value_grounding" in lab_trace_md
    assert "context_carryover" in lab_trace_md
    assert "France -> FR" in lab_trace_md
    assert "repairs empty result" in lab_trace_md
    assert "notebooks/blog/" not in lab_trace_md

    data_gates_md = (
        tmp_path / asset_paths["data_engineering_gates_md"]
    ).read_text()
    assert "fixed_proxy_slice" in data_gates_md
    assert "generated_history_rollout" in data_gates_md
    assert "value_entity_normalization" in data_gates_md
    assert "join_fanout_fixtures" in data_gates_md
    assert "alias_schema_validation" in data_gates_md
    assert "semantic_metric_manifest" in data_gates_md
    assert "hosted_same_protocol_baseline" in data_gates_md
    assert "bird_interact_transfer" in data_gates_md
    assert "teacher-forced" in data_gates_md
    assert "France -> FR" in data_gates_md
    assert "source_artifacts" in data_gates_md
    assert "claim_ids" in data_gates_md
    assert "rollout_beats_teacher_forced_history" in data_gates_md
    assert "hosted_sota_same_protocol" in data_gates_md
    assert "local_beats_hosted_same_protocol" in data_gates_md
    assert PUBLIC_LAB_NOTEBOOK in data_gates_md

    synthetic_fixtures_md = (
        tmp_path / asset_paths["synthetic_method_fixtures_md"]
    ).read_text()
    assert "synthetic_method_fixtures_summary.json" in synthetic_fixtures_md
    assert "fixture_count" in synthetic_fixtures_md
    assert "grain_fanout" in synthetic_fixtures_md
    assert "measure_preservation" in synthetic_fixtures_md
    assert "behavior_recovery" in synthetic_fixtures_md

    value_labels_md = (
        tmp_path / asset_paths["value_grounding_labels_md"]
    ).read_text()
    assert "value_grounding_labels_cosql_dev_100.jsonl" in value_labels_md
    assert "missing_from_user_text_count" in value_labels_md
    assert "exact_in_history_count" in value_labels_md

    value_index_md = (tmp_path / asset_paths["value_index_md"]).read_text()
    assert "value_index_cosql_dev_100_summary.json" in value_index_md
    assert "non_oracle_value_index_summary" in value_index_md
    assert "resolved_value_indexed_rate" in value_index_md
    assert "mention_alias_indexed_rate" in value_index_md
    assert "database_contents" in value_index_md

    artifact_contract_md = (
        tmp_path / asset_paths["data_artifact_contract_md"]
    ).read_text()
    assert "value_index" in artifact_contract_md
    assert "entity_resolution_labels" in artifact_contract_md
    assert "grain_fanout_fixtures" in artifact_contract_md
    assert "semantic_model_manifest" in artifact_contract_md
    assert "schema_alias_validator" in artifact_contract_md
    assert "generated_history_trace" in artifact_contract_md
    assert "France -> FR" in artifact_contract_md
    assert "MEASURE()" in artifact_contract_md
    assert "teacher-forced" in artifact_contract_md
    assert PUBLIC_LAB_NOTEBOOK in artifact_contract_md

    prompt_findings_md = (
        tmp_path / asset_paths["prompt_optimization_findings_md"]
    ).read_text()
    assert "non_oracle_sql_prompt_smoke" in prompt_findings_md
    assert "oracle_schema_pruned_prompt_search" in prompt_findings_md
    assert "planner_program_optimization_gate" in prompt_findings_md
    assert "schema_pruned_minimal" in prompt_findings_md
    assert "0.850" in prompt_findings_md
    assert "0.830" in prompt_findings_md
    assert "harness_ready" in prompt_findings_md
    assert "eval.planner_optimize" in prompt_findings_md
    assert "eval.planner_predict" in prompt_findings_md
    assert "eval.run_predicted_planner_comparison" in prompt_findings_md

    target_md = (tmp_path / asset_paths["target_comparison_md"]).read_text()
    assert "Direct SQL SFT" in target_md
    assert "0.530" in target_md
    assert "0.640" in target_md
    assert "MEASURE()-preserving metric DSL" in target_md
    assert "Behavior/recovery tuning" in target_md
    assert "shareable lab" in target_md

    target_evidence_md = (
        tmp_path / asset_paths["target_evidence_matrix_md"]
    ).read_text()
    assert "Direct SQL SFT" in target_evidence_md
    assert "Planner/DSL first, SQL second" in target_evidence_md
    assert "ready_for_endpoint_pair" in target_evidence_md
    assert "predicted_planner_sql_execution" in target_evidence_md
    assert "eval.run_predicted_planner_comparison" in target_evidence_md
    assert "metric_dsl_beats_direct_sql" in target_evidence_md
    assert "hosted baseline" in target_evidence_md
    assert "notebooks/blog/" not in target_evidence_md

    endpoint_md = (tmp_path / asset_paths["endpoint_run_scorecard_md"]).read_text()
    assert "Base Qwen 3.5 9B" in endpoint_md
    assert "100-step LoRA" in endpoint_md
    assert "0.530" in endpoint_md
    assert "0.640" in endpoint_md

    failure_delta_md = (
        tmp_path / asset_paths["failure_taxonomy_delta_md"]
    ).read_text()
    assert "multiturn-sql-semantic-50[minimal_executable]" in failure_delta_md
    assert "net_fixed" in failure_delta_md
    assert "value grounding" in failure_delta_md
    assert "projection" in failure_delta_md

    schema_findings_md = (
        tmp_path / asset_paths["schema_validation_findings_md"]
    ).read_text()
    assert "schema_mismatch_rows" in schema_findings_md
    assert "wrong_table_column" in schema_findings_md
    assert "repairable" in schema_findings_md


def test_blog_evidence_manifest_is_machine_checkable_contract(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)

    assert manifest["schema_version"] == 2
    assert manifest["source_repo"] == "multiturn-sql-finetuning"
    assert manifest["post_slug"] == "local-multiturn-sql-finetuning"
    assert "notebooks/blog/" in manifest["forbidden_public_substrings"]
    assert "notebook-contracts.md" in manifest["forbidden_public_substrings"]

    assets = {asset["id"]: asset for asset in manifest["assets"]}
    assert assets["claim_table_md"] == {
        "id": "claim_table_md",
        "path": "claim-table.md",
        "kind": "markdown",
        "required_in_post": False,
        "required_reference": True,
        "sha256": hashlib.sha256((tmp_path / "claim-table.md").read_bytes()).hexdigest(),
        "claim_ids": [
            "hosted_sota_same_protocol",
            "local_beats_hosted_same_protocol",
        ],
    }
    assert assets["accuracy_ladder_svg"]["kind"] == "svg"
    assert assets["accuracy_ladder_svg"]["required_in_post"] is True
    assert assets["accuracy_ladder_svg"]["required_reference"] is False
    assert len(assets["accuracy_ladder_svg"]["sha256"]) == 64
    assert {
        "qwen35_9b_base_cosql_dev_100turns",
        "multiturn_sql_100_cosql_dev_100turns",
        "semantic_prompt_minimal_executable_cosql_dev_100turns",
        "schema_pruned_minimal_schemafix_oracle_cosql_dev_100turns",
        "schema_pruned_trained100_oracle_cosql_dev_100turns",
    } <= set(assets["accuracy_ladder_svg"]["claim_ids"])
    assert assets["planner_baseline_svg"]["kind"] == "svg"
    assert assets["planner_baseline_svg"]["required_in_post"] is True
    assert assets["planner_baseline_svg"]["claim_ids"] == [
        "planner_lexical_schema_baseline"
    ]
    required_in_post_assets = [
        asset for asset in assets.values() if asset["required_in_post"]
    ]
    assert all(asset["claim_ids"] for asset in required_in_post_assets)
    assert assets["lab_reader_flow_md"]["required_reference"] is False
    assert assets["single_to_multiturn_gap_md"]["kind"] == "markdown"
    assert assets["single_to_multiturn_gap_md"]["required_reference"] is True
    assert "rollout_beats_teacher_forced_history" in assets["single_to_multiturn_gap_md"]["claim_ids"]
    assert assets["hosted_comparison_protocol_md"]["kind"] == "markdown"
    assert assets["hosted_comparison_protocol_md"]["required_reference"] is True
    assert "hosted_sota_same_protocol" in assets["hosted_comparison_protocol_md"]["claim_ids"]
    assert "local_beats_hosted_same_protocol" in assets["hosted_comparison_protocol_md"]["claim_ids"]
    assert "bird_interact_local_vs_hosted" in assets["hosted_comparison_protocol_md"]["claim_ids"]
    assert assets["evaluation_harness_map_md"]["kind"] == "markdown"
    assert assets["evaluation_harness_map_md"]["required_reference"] is True
    assert "predicted_planner_sql_execution" in assets["evaluation_harness_map_md"]["claim_ids"]
    assert "metric_dsl_beats_direct_sql" in assets["evaluation_harness_map_md"]["claim_ids"]
    assert assets["method_readiness_report_md"]["kind"] == "markdown"
    assert assets["method_readiness_report_md"]["required_reference"] is True
    assert "predicted_planner_sql_execution" in assets["method_readiness_report_md"]["claim_ids"]
    assert "metric_dsl_beats_direct_sql" in assets["method_readiness_report_md"]["claim_ids"]
    assert "rollout_beats_teacher_forced_history" in assets["method_readiness_report_md"]["claim_ids"]
    assert assets["finetuning_step_plan_md"]["kind"] == "markdown"
    assert assets["finetuning_step_plan_md"]["required_reference"] is True
    assert "semantic_value_retrieval_improves_sql" in assets["finetuning_step_plan_md"]["claim_ids"]
    assert "behavior_recovery_beats_direct_sql" in assets["finetuning_step_plan_md"]["claim_ids"]
    assert assets["experiment_ladder_md"]["kind"] == "markdown"
    assert assets["experiment_ladder_md"]["required_reference"] is True
    assert "predicted_planner_sql_execution" in assets["experiment_ladder_md"]["claim_ids"]
    assert "bird_interact_local_vs_hosted" in assets["experiment_ladder_md"]["claim_ids"]

    source_artifacts = {source["path"]: source for source in manifest["source_artifacts"]}
    assert (
        source_artifacts["docs/claim_ledgers/cosql_dev_100.jsonl"]["sha256"]
        == hashlib.sha256(
            (REPO_ROOT / "docs/claim_ledgers/cosql_dev_100.jsonl").read_bytes()
        ).hexdigest()
    )

    hosted_claim = manifest["claim_snapshot"]["hosted_sota_same_protocol"]
    assert hosted_claim["claim_status"] == "pending"
    assert hosted_claim["production_claim_allowed"] is False
    assert hosted_claim["can_support_sota_claim"] is False
    assert "hosted/SOTA comparison" in hosted_claim["allowed_public_claim"]

    ledger_rows = {
        row["claim_id"]: row
        for row in map(
            json.loads,
            (REPO_ROOT / "docs/claim_ledgers/cosql_dev_100.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    for claim_id in [
        "qwen35_9b_base_cosql_dev_100turns",
        "hosted_sota_same_protocol",
    ]:
        snapshot = manifest["claim_snapshot"][claim_id]
        ledger_row = ledger_rows[claim_id]
        for field in [
            "claim_status",
            "allowed_public_claim",
            "production_claim_allowed",
            "can_support_sota_claim",
        ]:
            assert snapshot[field] == ledger_row[field]


def test_checked_in_blog_evidence_assets_are_current(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)
    asset_paths = manifest_asset_paths(manifest)
    checked_in_dir = REPO_ROOT / "docs" / "blog" / "generated"

    assert (checked_in_dir / "manifest.json").exists()
    assert not (checked_in_dir / "notebook-walkthrough.md").exists()
    assert not (checked_in_dir / "notebook-contracts.md").exists()
    assert not (checked_in_dir / "notebook-series.md").exists()
    checked_in_manifest = json.loads((checked_in_dir / "manifest.json").read_text())
    assert checked_in_manifest == manifest

    for asset_path in asset_paths.values():
        assert (checked_in_dir / asset_path).read_text() == (tmp_path / asset_path).read_text()


def test_research_goal_states_notebook_led_method_comparison() -> None:
    goal = (REPO_ROOT / "docs" / "research_goal.md").read_text()
    blog_readme = (REPO_ROOT / "docs" / "blog" / "README.md").read_text()
    root_readme = (REPO_ROOT / "README.md").read_text()

    for phrase in [
        "single-turn",
        "multi-turn analytical SQL",
        "small specialized local model",
        "Direct SQL SFT",
        "Planner/DSL first, SQL second",
        "Semantic-layer tuning",
        "MEASURE()-preserving metric DSL",
        "Behavior/recovery tuning",
        "CoSQL",
        "SParC",
        "Synthetic schema-rich SQL",
        "BIRD-Interact",
        "data_artifact_contract",
        "Experiment Ladder",
        "hosted-comparison-protocol.md",
        "evaluation-harness-map.md",
        "method-readiness-report.md",
    ]:
        assert phrase in goal

    assert "Every public claim should name a lab section or generated evidence artifact" in goal
    assert PUBLIC_LAB_HTML_URL in goal
    assert PUBLIC_LAB_NOTEBOOK in goal
    assert PUBLIC_LAB_APP in goal
    assert "shareable-lab.md" in blog_readme
    assert "attached codebase" in blog_readme
    assert PUBLIC_LAB_HTML_URL in blog_readme
    assert PUBLIC_LAB_NOTEBOOK in blog_readme
    assert PUBLIC_LAB_APP in blog_readme
    assert "source code with Marimo" in blog_readme
    assert "notebooks/blog/" not in goal
    assert "notebooks/blog/" not in blog_readme
    assert "notebooks/blog/" not in root_readme
    assert "section notebook" not in goal.lower()
    assert "section notebook" not in blog_readme.lower()
    assert "section notebook" not in root_readme.lower()
    assert "internal checkpoint" not in blog_readme
    assert "setup notebook" not in blog_readme


def test_publishable_blog_evidence_exposes_single_public_lab_notebook(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)
    asset_paths = manifest_asset_paths(manifest)

    assert "notebook_contracts_md" not in asset_paths
    assert "notebook_series_md" not in asset_paths
    for asset_path in asset_paths.values():
        if not asset_path.endswith(".md"):
            continue
        content = (tmp_path / asset_path).read_text()
        assert "internal checkpoint" not in content
        assert "setup notebook" not in content
        assert "chapter notebook" not in content
        assert "notebooks/blog/" not in content
