from __future__ import annotations

import json
from pathlib import Path

from eval.method_readiness import build_method_readiness, write_method_readiness_report

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_method_readiness_maps_claims_to_next_actions() -> None:
    rows = build_method_readiness(
        ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
        repo_root=REPO_ROOT,
    )

    methods = {row["method"]: row for row in rows}
    assert set(methods) == {
        "Direct SQL SFT",
        "Planner/DSL first, SQL second",
        "Semantic-layer tuning",
        "MEASURE()-preserving metric DSL",
        "Behavior/recovery tuning",
        "Hosted and BIRD-Interact comparison",
    }

    direct = methods["Direct SQL SFT"]
    assert direct["readiness_level"] == "control_ready"
    assert direct["control_ready_now"] is True
    assert direct["rankable_now"] is False
    assert direct["supported_claim_ids"] == [
        "qwen35_9b_base_cosql_dev_100turns",
        "multiturn_sql_100_cosql_dev_100turns",
    ]
    assert "semantic_prompt_minimal_executable_cosql_dev_100turns" not in direct[
        "supported_claim_ids"
    ]
    assert direct["blocking_claim_ids"] == [
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
    ]
    assert direct["open_blocking_claim_ids"] == [
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
    ]
    assert "eval.run_eval" in direct["next_command"]

    planner = methods["Planner/DSL first, SQL second"]
    assert planner["readiness_level"] == "needs_endpoint_comparison"
    assert planner["control_ready_now"] is False
    assert planner["rankable_now"] is False
    assert "planner_lexical_schema_baseline" in planner["supported_claim_ids"]
    assert "predicted_planner_sql_execution" in planner["blocking_claim_ids"]
    assert "predicted_planner_sql_execution" in planner["open_blocking_claim_ids"]
    assert "eval.run_predicted_planner_comparison" in planner["next_command"]

    semantic = methods["Semantic-layer tuning"]
    assert semantic["readiness_level"] == "needs_value_entity_retrieval"
    assert semantic["control_ready_now"] is False
    assert semantic["rankable_now"] is False
    assert semantic["supported_claim_ids"] == [
        "semantic_prompt_minimal_executable_cosql_dev_100turns"
    ]

    metric = methods["MEASURE()-preserving metric DSL"]
    assert metric["readiness_level"] == "needs_prediction_manifest"
    assert metric["control_ready_now"] is False
    assert metric["rankable_now"] is False
    assert metric["blocking_claim_ids"] == [
        "metric_dsl_evaluation_manifest",
        "metric_dsl_beats_direct_sql",
    ]
    assert "eval.metric_dsl_eval" in metric["next_command"]

    recovery = methods["Behavior/recovery tuning"]
    assert recovery["readiness_level"] == "needs_rollout_manifest"
    assert "rollout_beats_teacher_forced_history" in recovery["blocking_claim_ids"]
    assert "eval.rollout_eval" in recovery["next_command"]

    hosted = methods["Hosted and BIRD-Interact comparison"]
    assert hosted["readiness_level"] == "needs_hosted_protocol_run"
    assert hosted["control_ready_now"] is False
    assert hosted["rankable_now"] is False
    assert "hosted_sota_same_protocol" in hosted["blocking_claim_ids"]
    assert "local_beats_hosted_same_protocol" in hosted["blocking_claim_ids"]
    assert "eval.compare_hosted_baseline" in hosted["next_command"]

    for row in rows:
        assert (REPO_ROOT / row["module_path"]).exists(), row["module_path"]
        assert row["next_artifact"]
        assert row["claim_boundary"]


def test_write_method_readiness_report_is_deterministic(tmp_path) -> None:
    output = tmp_path / "method_readiness.json"
    rows = write_method_readiness_report(
        output_path=output,
        ledger_path=REPO_ROOT / "docs" / "claim_ledgers" / "cosql_dev_100.jsonl",
        repo_root=REPO_ROOT,
    )

    payload = json.loads(output.read_text())
    assert payload == {"schema_version": 1, "methods": rows}
    assert payload["methods"][0]["method"] == "Direct SQL SFT"
    assert payload["methods"][-1]["method"] == "Hosted and BIRD-Interact comparison"
