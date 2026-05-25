from __future__ import annotations

import ast
import json
from pathlib import Path

from notebooks.blog_support import (
    accuracy_scorecard,
    blog_notebook_series,
    claim_ledger,
    claim_table,
    data_engineering_gates,
    endpoint_run_scorecard,
    export_blog_evidence,
    lab_failure_trace,
    lab_method_scorecard,
    lab_reader_flow,
    metric_dsl_demo,
    metric_dsl_eval_contract,
    planner_scorecard,
    prompt_optimization_findings,
    semantic_strategy_table,
    shareable_lab_attachment,
    target_comparison,
    target_evidence_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

PUBLIC_LAB_NOTEBOOK = "notebooks/labs/local_multiturn_sql_lab.ipynb"
PUBLIC_LAB_APP = "notebooks/labs/local_multiturn_sql_lab.py"

PUBLIC_SECTION_NOTEBOOKS = [
    "notebooks/blog/01_benchmark_gap.py",
    "notebooks/blog/02_eval_protocol.py",
    "notebooks/blog/03_method_targets.py",
    "notebooks/blog/04_results_diagnostics.py",
    "notebooks/blog/05_next_experiments.py",
]
PUBLIC_SECTION_PACKAGE_INIT = "notebooks/blog/__init__.py"

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


def test_public_blog_artifacts_are_one_shareable_lab_notebook() -> None:
    lab_app = REPO_ROOT / PUBLIC_LAB_APP
    lab_ipynb = REPO_ROOT / PUBLIC_LAB_NOTEBOOK

    app_source = lab_app.read_text()
    ast.parse(app_source, filename=PUBLIC_LAB_APP)
    assert "import marimo" in app_source
    assert "app = marimo.App" in app_source
    assert "run_multiturn_lab" in app_source
    assert "device_preference=runtime_choice.value" in app_source
    assert 'value="auto"' in app_source
    assert "auto-select" in app_source
    assert 'if __name__ == "__main__":' in app_source
    assert "app.run()" in app_source

    notebook = json.loads(lab_ipynb.read_text())
    assert notebook["nbformat"] == 4
    text = "\n".join("".join(cell.get("source", "")) for cell in notebook["cells"])
    assert "run_multiturn_lab" in text
    assert 'device_preference="auto"' in text
    assert "CUDA" in text
    assert "MPS" in text
    assert "XPU" in text
    assert "endpoint_run_scorecard" in text
    assert "planner_scorecard" in text
    assert "target_evidence_matrix" in text
    assert "metric_dsl_eval_contract" in text
    assert "prompt_optimization_findings" in text
    assert "pip install" not in text
    assert "apt install" not in text
    for notebook_path in FORBIDDEN_SETUP_NOTEBOOKS:
        assert not (REPO_ROOT / notebook_path).exists()
    for chapter_path in FORBIDDEN_BLOG_CHAPTER_DOCS:
        assert not (REPO_ROOT / chapter_path).exists()


def test_public_section_notebooks_are_runnable_marimo_apps() -> None:
    series = blog_notebook_series()

    assert list(series["notebook"]) == PUBLIC_SECTION_NOTEBOOKS
    assert list(series["step"]) == [1, 2, 3, 4, 5]
    assert set(series["post_section"]) == {
        "Benchmark gap",
        "Evaluation protocol",
        "Fine-tuning targets",
        "Results and diagnostics",
        "Next experiments",
    }
    assert all(series["run_command"].str.startswith("marimo edit notebooks/blog/"))
    assert all(series["claim_boundary"].str.len() > 0)
    assert all(series["attached_lab"] == PUBLIC_LAB_NOTEBOOK)

    for notebook_path in PUBLIC_SECTION_NOTEBOOKS:
        source = (REPO_ROOT / notebook_path).read_text()
        ast.parse(source, filename=notebook_path)
        assert "import marimo" in source
        assert "app = marimo.App" in source
        assert "blog_notebook_series" in source
        assert 'if __name__ == "__main__":' in source
        assert "app.run()" in source
        assert "pip install" not in source
        assert "apt install" not in source
    assert (REPO_ROOT / PUBLIC_SECTION_PACKAGE_INIT).exists()


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

    lab_attachment = shareable_lab_attachment()
    assert {
        "artifact",
        "notebook",
        "repo_url",
        "run_command",
        "alternate_command",
        "device_policy",
        "reader_flow",
        "what_runs",
        "claim_boundary",
    } <= set(lab_attachment.columns)
    lab_row = lab_attachment.iloc[0]
    assert "shareable lab notebook and attached codebase" in lab_row["artifact"]
    assert lab_row["notebook"] == PUBLIC_LAB_NOTEBOOK
    assert "github.com/xdanny/multiturn-sql-finetuning" in lab_row["repo_url"]
    assert f"jupyter lab {PUBLIC_LAB_NOTEBOOK}" in lab_row["run_command"]
    assert f"marimo edit {PUBLIC_LAB_APP}" in lab_row["alternate_command"]
    assert "auto-selects CUDA, MPS, or XPU" in lab_row["device_policy"]
    assert "falls back to CPU" in lab_row["device_policy"]
    assert "CUDA" in lab_row["device_policy"]
    assert "MPS" in lab_row["device_policy"]
    assert "XPU" in lab_row["device_policy"]
    assert "Research question" in lab_row["reader_flow"]
    assert "open the lab notebook" in lab_row["reader_flow"]
    assert "run the section notebooks" in lab_row["reader_flow"]
    assert "compare fine-tuning targets" in lab_row["reader_flow"]
    assert "read the evidence gates" in lab_row["reader_flow"]
    assert "chapter" not in lab_row["artifact"]

    reader_flow = lab_reader_flow()
    assert {
        "step",
        "post_section",
        "notebook",
        "run_command",
        "reader_action",
        "evidence_to_inspect",
        "claim_boundary",
        "attached_lab",
    } <= set(reader_flow.columns)
    assert list(reader_flow["step"]) == list(range(1, len(reader_flow) + 1))
    assert "Benchmark gap" in set(reader_flow["post_section"])
    assert "Evaluation protocol" in set(reader_flow["post_section"])
    assert "Fine-tuning targets" in set(reader_flow["post_section"])
    assert "Results and diagnostics" in set(reader_flow["post_section"])
    assert "Next experiments" in set(reader_flow["post_section"])
    assert any("single-turn" in action for action in reader_flow["reader_action"])
    assert any("not a benchmark result" in boundary for boundary in reader_flow["claim_boundary"])
    assert set(reader_flow["notebook"]) == set(PUBLIC_SECTION_NOTEBOOKS)
    assert all("marimo edit notebooks/blog/" in command for command in reader_flow["run_command"])
    assert all(reader_flow["attached_lab"] == PUBLIC_LAB_NOTEBOOK)


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
        "value_match",
        "actual_rows",
        "expected_rows",
        "intermediate_plan",
        "why_it_matters",
    } <= set(lab_trace.columns)
    trace_keys = {(row["turn_id"], row["system"]) for _, row in lab_trace.iterrows()}
    assert ("turn_2", "direct_sql_baseline") in trace_keys
    assert ("turn_3", "direct_sql_baseline") in trace_keys
    assert ("turn_4", "behavior_recovery_sql") in trace_keys
    assert "value_grounding" in set(lab_trace["failure_type"])
    assert "context_carryover" in set(lab_trace["failure_type"])
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
    assert "metric-DSL prediction/comparison manifest" in metric_row["manifest_backed_evidence"]
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
        "metric_dsl_evaluation_manifest",
        "hosted_sota_same_protocol",
        "bird_interact_local_vs_hosted",
    } <= gate_claim_ids

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
    assert "planner label F1" in planner_gate["next_program_target"]
    assert all("not a SOTA claim" in boundary for boundary in prompt_findings["claim_boundary"])


def test_export_blog_evidence_writes_publishable_assets(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)

    assert manifest["schema_version"] == 1
    assert manifest["source_repo"] == "multiturn-sql-finetuning"
    assert set(manifest["assets"]) == {
        "accuracy_ladder_svg",
        "planner_baseline_svg",
        "claim_table_md",
        "metric_dsl_contract_md",
        "shareable_lab_md",
        "notebook_series_md",
        "lab_reader_flow_md",
        "lab_method_scores_md",
        "lab_failure_trace_md",
        "data_engineering_gates_md",
        "prompt_optimization_findings_md",
        "target_comparison_md",
        "target_evidence_matrix_md",
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

    shareable_lab_md = (tmp_path / manifest["assets"]["shareable_lab_md"]).read_text()
    assert "shareable lab notebook and attached codebase" in shareable_lab_md
    assert PUBLIC_LAB_NOTEBOOK in shareable_lab_md
    assert PUBLIC_LAB_APP in shareable_lab_md
    assert f"jupyter lab {PUBLIC_LAB_NOTEBOOK}" in shareable_lab_md
    assert "auto-selects CUDA, MPS, or XPU" in shareable_lab_md
    assert "falls back to CPU" in shareable_lab_md
    assert "CUDA" in shareable_lab_md
    assert "MPS" in shareable_lab_md
    assert "XPU" in shareable_lab_md
    assert "Research question" in shareable_lab_md
    assert "open the lab notebook" in shareable_lab_md
    assert "run the section notebooks" in shareable_lab_md
    assert "compare fine-tuning targets" in shareable_lab_md
    assert "read the evidence gates" in shareable_lab_md
    assert "chapter" not in shareable_lab_md

    notebook_series_md = (tmp_path / manifest["assets"]["notebook_series_md"]).read_text()
    assert "Benchmark gap" in notebook_series_md
    assert "Evaluation protocol" in notebook_series_md
    assert "Fine-tuning targets" in notebook_series_md
    assert "Results and diagnostics" in notebook_series_md
    assert "Next experiments" in notebook_series_md
    assert PUBLIC_LAB_NOTEBOOK in notebook_series_md
    for notebook_path in PUBLIC_SECTION_NOTEBOOKS:
        assert notebook_path in notebook_series_md

    lab_flow_md = (tmp_path / manifest["assets"]["lab_reader_flow_md"]).read_text()
    assert PUBLIC_LAB_NOTEBOOK in lab_flow_md
    assert "Research question" in lab_flow_md
    assert "Benchmark gap" in lab_flow_md
    assert "Fine-tuning targets" in lab_flow_md
    assert "not a benchmark result" in lab_flow_md
    for notebook_path in PUBLIC_SECTION_NOTEBOOKS:
        assert notebook_path in lab_flow_md

    lab_scores_md = (tmp_path / manifest["assets"]["lab_method_scores_md"]).read_text()
    assert "direct_sql_baseline" in lab_scores_md
    assert "semantic_dsl_planner" in lab_scores_md
    assert "behavior_recovery_sql" in lab_scores_md
    assert "0.250" in lab_scores_md
    assert "1.000" in lab_scores_md
    assert "not a benchmark result" in lab_scores_md

    lab_trace_md = (tmp_path / manifest["assets"]["lab_failure_trace_md"]).read_text()
    assert "turn_2" in lab_trace_md
    assert "turn_3" in lab_trace_md
    assert "turn_4" in lab_trace_md
    assert "value_grounding" in lab_trace_md
    assert "context_carryover" in lab_trace_md
    assert "France -> FR" in lab_trace_md
    assert "repairs empty result" in lab_trace_md
    assert "notebooks/blog/" not in lab_trace_md

    data_gates_md = (
        tmp_path / manifest["assets"]["data_engineering_gates_md"]
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
    assert PUBLIC_LAB_NOTEBOOK in data_gates_md

    prompt_findings_md = (
        tmp_path / manifest["assets"]["prompt_optimization_findings_md"]
    ).read_text()
    assert "non_oracle_sql_prompt_smoke" in prompt_findings_md
    assert "oracle_schema_pruned_prompt_search" in prompt_findings_md
    assert "planner_program_optimization_gate" in prompt_findings_md
    assert "schema_pruned_minimal" in prompt_findings_md
    assert "0.850" in prompt_findings_md
    assert "0.830" in prompt_findings_md
    assert "planner label F1" in prompt_findings_md
    assert "shareable lab notebook" in prompt_findings_md

    target_md = (tmp_path / manifest["assets"]["target_comparison_md"]).read_text()
    assert "Direct SQL SFT" in target_md
    assert "0.640" in target_md
    assert "MEASURE()-preserving metric DSL" in target_md
    assert "Behavior/recovery tuning" in target_md
    assert "shareable lab" in target_md

    target_evidence_md = (
        tmp_path / manifest["assets"]["target_evidence_matrix_md"]
    ).read_text()
    assert "Direct SQL SFT" in target_evidence_md
    assert "Planner/DSL first, SQL second" in target_evidence_md
    assert "predicted_planner_sql_execution" in target_evidence_md
    assert "metric_dsl_beats_direct_sql" in target_evidence_md
    assert "hosted baseline" in target_evidence_md
    assert "notebooks/blog/" not in target_evidence_md

    endpoint_md = (tmp_path / manifest["assets"]["endpoint_run_scorecard_md"]).read_text()
    assert "Base Qwen 3.5 9B" in endpoint_md
    assert "100-step LoRA" in endpoint_md
    assert "0.530" in endpoint_md
    assert "0.640" in endpoint_md


def test_checked_in_blog_evidence_assets_are_current(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)
    checked_in_dir = REPO_ROOT / "docs" / "blog" / "generated"

    assert (checked_in_dir / "manifest.json").exists()
    assert not (checked_in_dir / "notebook-walkthrough.md").exists()
    assert not (checked_in_dir / "notebook-contracts.md").exists()
    checked_in_manifest = json.loads((checked_in_dir / "manifest.json").read_text())
    assert checked_in_manifest == manifest

    for asset_path in manifest["assets"].values():
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
    ]:
        assert phrase in goal

    assert "Every public claim should name a section notebook, the lab notebook section, or generated evidence artifact" in goal
    assert PUBLIC_LAB_NOTEBOOK in goal
    assert PUBLIC_LAB_APP in goal
    for notebook_path in PUBLIC_SECTION_NOTEBOOKS:
        assert notebook_path in goal
        assert notebook_path in blog_readme
        assert notebook_path in root_readme
    assert "shareable-lab.md" in blog_readme
    assert "attached codebase" in blog_readme
    assert PUBLIC_LAB_NOTEBOOK in blog_readme
    assert PUBLIC_LAB_APP in blog_readme
    assert "internal checkpoint" not in blog_readme
    assert "setup notebook" not in blog_readme


def test_publishable_blog_evidence_exposes_single_public_lab_notebook(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)

    assert "notebook_contracts_md" not in manifest["assets"]
    assert "notebook_series_md" in manifest["assets"]
    for asset_path in manifest["assets"].values():
        if not asset_path.endswith(".md"):
            continue
        content = (tmp_path / asset_path).read_text()
        assert "internal checkpoint" not in content
        assert "setup notebook" not in content
        assert "chapter notebook" not in content
