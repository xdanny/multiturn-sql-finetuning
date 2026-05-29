from __future__ import annotations

import json
from pathlib import Path

from eval.method_readiness import (
    build_method_readiness,
    load_method_configs,
    required_readiness_failures,
    write_method_readiness_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_method_readiness_loads_method_arms_from_config() -> None:
    configs = load_method_configs(REPO_ROOT / "configs" / "finetuning_methods.yaml")

    assert [config["method"] for config in configs] == [
        "Direct SQL SFT",
        "Planner/DSL first, SQL second",
        "Semantic-layer tuning",
        "MEASURE()-preserving metric DSL",
        "Behavior/recovery tuning",
        "Hosted and BIRD-Interact comparison",
    ]
    assert configs[0]["training_rows"] == ("data/processed/train_smoke.jsonl",)
    assert configs[0]["smoke_rows_required"] is True
    assert configs[0]["control_rows"] == ()
    assert configs[0]["control_rows_required"] is False
    assert configs[0]["next_command"].startswith("uv run --active --no-sync")
    assert "reference SQL" not in configs[0]["next_command"].lower()
    assert configs[0]["finetuning_objective"]
    assert configs[0]["benchmark_scope"]
    assert configs[0]["primary_metric"]
    assert configs[0]["leakage_boundary"]
    assert configs[0]["evidence_gate"]


def test_method_readiness_requires_method_contract_fields(tmp_path) -> None:
    config = tmp_path / "methods.yaml"
    config.write_text(
        """
schema_version: 1
methods:
  - method: Missing contract
    readiness_level: needs_rows
    supported_claim_ids: []
    blocking_claim_ids: []
    module_path: eval/run_eval.py
    training_rows: []
    smoke_rows_required: false
    control_rows: []
    control_rows_required: false
    prediction_input_rows: []
    prediction_input_rows_required: false
    evaluator_paths:
      - eval/run_eval.py
    next_command: uv run --active --no-sync python -m eval.run_eval
    next_artifact: result manifest
    claim_boundary: no claim
    rankable_when: never
    finetuning_objective: learn the target behavior
    benchmark_scope: synthetic smoke rows
    leakage_boundary: no scorer fields in prompts
    evidence_gate: same-row comparison manifest
""",
        encoding="utf-8",
    )

    try:
        load_method_configs(config)
    except ValueError as exc:
        assert "Missing contract: missing primary_metric" in str(exc)
    else:
        raise AssertionError("missing method contract field should fail")


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
    assert "control arm" in direct["finetuning_objective"]
    assert "CoSQL proxy" in direct["benchmark_scope"]
    assert "value-only execution accuracy" in direct["primary_metric"]
    assert "reference SQL" in direct["leakage_boundary"]
    assert "manifest" in direct["evidence_gate"]

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
        "semantic_prompt_minimal_executable_cosql_dev_100turns",
        "value_index_coverage",
    ]
    assert "semantic_value_retrieval_improves_sql" in semantic["blocking_claim_ids"]
    assert "semantic_value_retrieval_improves_sql" in semantic["open_blocking_claim_ids"]
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
    assert metric["prediction_input_rows_required"] is True
    assert metric["prediction_input_rows"] == {
        "docs/data_artifacts/metric_dsl_prediction_inputs.jsonl": True,
        "docs/data_artifacts/metric_dsl_direct_sql_prediction_inputs.jsonl": True,
    }
    assert metric["prediction_input_rows_ready"] is True
    assert metric["supported_claim_ids"] == ["metric-dsl-bootstrap.metric_dsl"]
    assert metric["blocking_claim_ids"] == ["metric_dsl_beats_direct_sql"]
    assert "eval.generate_metric_dsl_predictions" in metric["next_command"]
    assert "eval.run_metric_dsl_comparison" in metric["next_command"]
    assert metric["evaluator_paths"]["eval/generate_metric_dsl_predictions.py"] is True
    assert metric["evaluator_paths"]["eval/run_metric_dsl_comparison.py"] is True

    recovery = methods["Behavior/recovery tuning"]
    assert recovery["readiness_level"] == "needs_rollout_manifest"
    assert recovery["smoke_rows_ready"] is True
    assert recovery["smoke_rows_required"] is True
    assert recovery["control_rows_required"] is True
    assert recovery["control_rows_ready"] is True
    assert recovery["prediction_input_rows_required"] is True
    assert recovery["prediction_input_rows"] == {
        "docs/data_artifacts/behavior_recovery_rollout_inputs.jsonl": True
    }
    assert recovery["prediction_input_rows_ready"] is True
    assert recovery["evaluators_ready"] is True
    assert recovery["training_rows"] == {
        "docs/data_artifacts/behavior_recovery_training_rows.jsonl": True
    }
    assert "rollout_beats_teacher_forced_history" in recovery["blocking_claim_ids"]
    assert "behavior_recovery_beats_direct_sql" in recovery["blocking_claim_ids"]
    assert "behavior_recovery_beats_direct_sql" in recovery["open_blocking_claim_ids"]
    assert recovery["evaluator_paths"]["data/behavior_recovery_rollout_inputs.py"] is True
    assert recovery["evaluator_paths"]["eval/behavior_recovery_teacher_forced.py"] is True
    assert recovery["evaluator_paths"]["eval/run_behavior_recovery_comparison.py"] is True
    assert "eval.run_behavior_recovery_comparison" in recovery["next_command"]
    assert "--output-dir results/rollout" in recovery["next_command"]
    assert "direct-SQL control" in recovery["claim_boundary"]

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
        assert row["finetuning_objective"]
        assert row["benchmark_scope"]
        assert row["primary_metric"]
        assert row["leakage_boundary"]
        assert row["evidence_gate"]
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
            "prediction_input_rows_required": True,
            "prediction_input_rows_ready": False,
            "evaluators_ready": False,
            "training_rows": {"data/missing_train.jsonl": False},
            "control_rows": {"data/missing_control.jsonl": False},
            "prediction_input_rows": {"data/missing_prediction.jsonl": False},
            "evaluator_paths": {"eval/missing_eval.py": False},
        },
    ]

    assert required_readiness_failures(rows) == [
        "Missing required rows: missing smoke rows: data/missing_train.jsonl",
        "Missing required rows: missing control rows: data/missing_control.jsonl",
        "Missing required rows: missing prediction input rows: data/missing_prediction.jsonl",
        "Missing required rows: missing evaluators: eval/missing_eval.py",
    ]
