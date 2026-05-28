from __future__ import annotations

import json
from pathlib import Path

from eval.method_readiness import (
    build_method_readiness,
    required_readiness_failures,
    write_method_readiness_report,
)

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
    assert direct["smoke_rows_ready"] is True
    assert direct["smoke_rows_required"] is True
    assert direct["control_rows_required"] is False
    assert direct["control_rows_ready"] is True
    assert direct["evaluators_ready"] is True
    assert direct["training_rows"] == {"data/processed/train_smoke.jsonl": True}
    assert "eval.run_eval" in direct["next_command"]
    assert direct["next_command"].startswith("uv run --active --no-sync")

    planner = methods["Planner/DSL first, SQL second"]
    assert planner["readiness_level"] == "needs_endpoint_comparison"
    assert planner["control_ready_now"] is False
    assert planner["rankable_now"] is False
    assert planner["smoke_rows_required"] is False
    assert planner["smoke_rows_ready"] is True
    assert planner["control_rows_required"] is True
    assert planner["control_rows_ready"] is True
    assert planner["evaluators_ready"] is True
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
    assert semantic["smoke_rows_ready"] is True
    assert semantic["smoke_rows_required"] is True
    assert semantic["control_rows_required"] is True
    assert semantic["control_rows_ready"] is True
    assert semantic["evaluators_ready"] is True

    metric = methods["MEASURE()-preserving metric DSL"]
    assert metric["readiness_level"] == "needs_prediction_manifest"
    assert metric["control_ready_now"] is False
    assert metric["rankable_now"] is False
    assert metric["smoke_rows_ready"] is True
    assert metric["smoke_rows_required"] is True
    assert metric["control_rows_required"] is True
    assert metric["control_rows_ready"] is True
    assert metric["evaluators_ready"] is True
    assert metric["training_rows"] == {
        "docs/data_artifacts/metric_dsl_training_rows.jsonl": True
    }
    assert metric["control_rows"] == {
        "docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl": True
    }
    assert metric["blocking_claim_ids"] == [
        "metric_dsl_evaluation_manifest",
        "metric_dsl_beats_direct_sql",
    ]
    assert "eval.metric_dsl_eval" in metric["next_command"]

    recovery = methods["Behavior/recovery tuning"]
    assert recovery["readiness_level"] == "needs_rollout_manifest"
    assert recovery["smoke_rows_ready"] is True
    assert recovery["smoke_rows_required"] is True
    assert recovery["control_rows_required"] is True
    assert recovery["control_rows_ready"] is True
    assert recovery["evaluators_ready"] is True
    assert recovery["training_rows"] == {
        "docs/data_artifacts/behavior_recovery_training_rows.jsonl": True
    }
    assert "rollout_beats_teacher_forced_history" in recovery["blocking_claim_ids"]
    assert "eval.rollout_eval" in recovery["next_command"]

    hosted = methods["Hosted and BIRD-Interact comparison"]
    assert hosted["readiness_level"] == "needs_hosted_protocol_run"
    assert hosted["control_ready_now"] is False
    assert hosted["rankable_now"] is False
    assert hosted["smoke_rows_required"] is False
    assert hosted["smoke_rows_ready"] is True
    assert hosted["control_rows_required"] is False
    assert hosted["control_rows_ready"] is True
    assert hosted["evaluators_ready"] is True
    assert "hosted_sota_same_protocol" in hosted["blocking_claim_ids"]
    assert "local_beats_hosted_same_protocol" in hosted["blocking_claim_ids"]
    assert "eval.compare_hosted_baseline" in hosted["next_command"]

    for row in rows:
        assert (REPO_ROOT / row["module_path"]).exists(), row["module_path"]
        assert row["next_artifact"]
        assert row["claim_boundary"]
        assert "uv run --active --no-sync" in row["next_command"]


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


def test_required_readiness_failures_only_reports_required_missing_inputs() -> None:
    rows = [
        {
            "method": "Optional rows method",
            "smoke_rows_required": False,
            "smoke_rows_ready": True,
            "control_rows_required": False,
            "control_rows_ready": True,
            "evaluators_ready": True,
            "training_rows": {},
            "control_rows": {},
            "evaluator_paths": {"eval/example.py": True},
        },
        {
            "method": "Missing required rows",
            "smoke_rows_required": True,
            "smoke_rows_ready": False,
            "control_rows_required": True,
            "control_rows_ready": False,
            "evaluators_ready": False,
            "training_rows": {"data/missing_train.jsonl": False},
            "control_rows": {"data/missing_control.jsonl": False},
            "evaluator_paths": {"eval/missing_eval.py": False},
        },
    ]

    assert required_readiness_failures(rows) == [
        "Missing required rows: missing smoke rows: data/missing_train.jsonl",
        "Missing required rows: missing control rows: data/missing_control.jsonl",
        "Missing required rows: missing evaluators: eval/missing_eval.py",
    ]
