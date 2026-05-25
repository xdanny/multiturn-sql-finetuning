from __future__ import annotations

import ast
import json
from pathlib import Path

from notebooks.blog_support import (
    accuracy_scorecard,
    claim_ledger,
    claim_table,
    data_engineering_gates,
    endpoint_run_scorecard,
    export_blog_evidence,
    lab_reader_flow,
    metric_dsl_demo,
    metric_dsl_eval_contract,
    planner_scorecard,
    semantic_strategy_table,
    shareable_lab_attachment,
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

    lab_attachment = shareable_lab_attachment()
    assert {
        "artifact",
        "notebook",
        "repo_url",
        "run_command",
        "device_policy",
        "reader_flow",
        "what_runs",
        "claim_boundary",
    } <= set(lab_attachment.columns)
    lab_row = lab_attachment.iloc[0]
    assert lab_row["artifact"] == "reader-facing lab notebook"
    assert lab_row["notebook"] == "notebooks/labs/local_multiturn_sql_lab.py"
    assert "github.com/xdanny/multiturn-sql-finetuning" in lab_row["repo_url"]
    assert "marimo edit notebooks/labs/local_multiturn_sql_lab.py" in lab_row["run_command"]
    assert "CPU by default" in lab_row["device_policy"]
    assert "CUDA" in lab_row["device_policy"]
    assert "MPS" in lab_row["device_policy"]
    assert "XPU" in lab_row["device_policy"]
    assert "Research question" in lab_row["reader_flow"]
    assert "single-turn gap" in lab_row["reader_flow"]
    assert "target comparison" in lab_row["reader_flow"]
    assert "claim boundary" in lab_row["reader_flow"]
    assert "notebooks/blog/02_wsl_5090_setup.py" not in lab_row.to_string()

    reader_flow = lab_reader_flow()
    assert {
        "step",
        "post_section",
        "lab_section",
        "reader_action",
        "evidence_to_inspect",
        "claim_boundary",
    } <= set(reader_flow.columns)
    assert list(reader_flow["step"]) == list(range(1, len(reader_flow) + 1))
    assert "Run the lab" in set(reader_flow["post_section"])
    assert "Why one-shot SQL isn't enough" in set(reader_flow["post_section"])
    assert "Fine-tuning loop" in set(reader_flow["post_section"])
    assert "Preserving MEASURE()" in set(reader_flow["post_section"])
    assert "The road ahead" in set(reader_flow["post_section"])
    assert any("single-turn" in action for action in reader_flow["reader_action"])
    assert any("not a benchmark result" in boundary for boundary in reader_flow["claim_boundary"])
    assert not any("notebooks/blog/" in row for row in reader_flow.astype(str).to_numpy().ravel())

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
        "lab_reader_flow_md",
        "data_engineering_gates_md",
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

    shareable_lab_md = (tmp_path / manifest["assets"]["shareable_lab_md"]).read_text()
    assert "reader-facing lab notebook" in shareable_lab_md
    assert "notebooks/labs/local_multiturn_sql_lab.py" in shareable_lab_md
    assert "marimo edit notebooks/labs/local_multiturn_sql_lab.py" in shareable_lab_md
    assert "CPU by default" in shareable_lab_md
    assert "CUDA" in shareable_lab_md
    assert "MPS" in shareable_lab_md
    assert "XPU" in shareable_lab_md
    assert "Research question" in shareable_lab_md
    assert "single-turn gap" in shareable_lab_md
    assert "target comparison" in shareable_lab_md
    assert "claim boundary" in shareable_lab_md
    assert "notebooks/blog/" not in shareable_lab_md

    lab_flow_md = (tmp_path / manifest["assets"]["lab_reader_flow_md"]).read_text()
    assert "Run the lab" in lab_flow_md
    assert "Why one-shot SQL isn't enough" in lab_flow_md
    assert "Fine-tuning loop" in lab_flow_md
    assert "Preserving MEASURE()" in lab_flow_md
    assert "The road ahead" in lab_flow_md
    assert "notebooks/blog/" not in lab_flow_md

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
    assert "notebooks/blog/" not in data_gates_md

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

    assert "Every public claim should name the reader-facing lab or generated evidence artifact" in goal
    assert "shareable-lab.md" in blog_readme
    assert "reader-facing lab" in blog_readme
    assert "notebooks/blog/02_wsl_5090_setup.py" not in blog_readme
    assert "internal checkpoint" not in blog_readme
    assert "chapter notebooks" not in blog_readme
    assert "notebooks/blog/" not in root_readme


def test_publishable_blog_evidence_exposes_only_the_shareable_lab(tmp_path) -> None:
    manifest = export_blog_evidence(tmp_path)

    assert "notebook_contracts_md" not in manifest["assets"]
    for asset_path in manifest["assets"].values():
        if not asset_path.endswith(".md"):
            continue
        content = (tmp_path / asset_path).read_text()
        assert "notebooks/blog/" not in content
        assert "internal checkpoint" not in content
        assert "setup notebook" not in content
