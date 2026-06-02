from __future__ import annotations

from pathlib import Path

from eval.experiment_registry import experiment_registry_map, load_experiment_registry

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_registry_names_roadmap_runs() -> None:
    experiments = load_experiment_registry(REPO_ROOT / "configs" / "experiments.yaml")

    experiment_ids = [row["experiment_id"] for row in experiments]
    assert experiment_ids == [
        "direct_sql_full_non_oracle_control",
        "predicted_planner_sql_vs_direct",
        "semantic_value_retrieval_vs_direct",
        "metric_dsl_vs_direct_sql",
        "generated_history_recovery_vs_direct",
        "hosted_bird_interact_transfer",
    ]

    for row in experiments:
        assert row["hypothesis_id"]
        assert row["train_split_id"]
        assert row["validation_split_id"]
        assert row["test_split_id"]
        assert row["model"] == "unsloth/Qwen3.5-9B"
        assert row["oracle_policy"] == "non_oracle_generation"
        assert row["scorer"]
        assert row["output_path"].startswith("results/runs/")

    direct = experiments[0]
    assert direct["method"] == "direct_sql"
    assert direct["control_experiment_id"] == ""

    planner = next(row for row in experiments if row["method"] == "predicted_planner_sql")
    assert planner["control_experiment_id"] == "direct_sql_full_non_oracle_control"
    assert planner["status"] == "planner_alias_normalized_readiness_order_negative"

    semantic = next(row for row in experiments if row["method"] == "semantic_value_retrieval")
    assert semantic["dataset_role"] == "clean_local_holdout"
    assert semantic["status"] == "semantic_pruned_clean_holdout_promoted"


def test_experiment_registry_loader_rejects_missing_required_field(tmp_path) -> None:
    config = tmp_path / "experiments.yaml"
    config.write_text(
        """
schema_version: 1
experiments:
  - experiment_id: bad_experiment
    checkpoint: 1
    status: planned
    hypothesis_id: direct_sql
    hypothesis: missing fields
    train_split_id: train
    validation_split_id: validation
    test_split_id: test
    dataset_role: validation
    benchmark_protocol_id: synthetic_schema_rich_method_fixture
    model: model
    method: direct_sql
    oracle_policy: non_oracle_generation
    output_path: results/runs/bad
    primary_metric: value_accuracy
    claim_boundary: test only
""",
        encoding="utf-8",
    )

    try:
        load_experiment_registry(config)
    except ValueError as exc:
        assert "bad_experiment: missing scorer" in str(exc)
    else:
        raise AssertionError("missing scorer should fail")


def test_experiment_registry_loader_rejects_missing_comparison_control(tmp_path) -> None:
    config = tmp_path / "experiments.yaml"
    config.write_text(
        """
schema_version: 1
experiments:
  - experiment_id: direct
    checkpoint: 3
    status: planned
    hypothesis_id: direct_sql
    hypothesis: direct control
    train_split_id: train
    validation_split_id: validation
    test_split_id: test
    dataset_role: validation
    benchmark_protocol_id: synthetic_schema_rich_method_fixture
    model: model
    method: direct_sql
    oracle_policy: non_oracle_generation
    scorer: scorer
    output_path: results/runs/direct
    primary_metric: value_accuracy
    claim_boundary: test only
  - experiment_id: method
    checkpoint: 5
    status: planned
    hypothesis_id: planner
    hypothesis: method comparison
    train_split_id: train
    validation_split_id: validation
    test_split_id: test
    dataset_role: validation
    benchmark_protocol_id: synthetic_schema_rich_method_fixture
    model: model
    method: predicted_planner_sql
    oracle_policy: non_oracle_generation
    scorer: scorer
    output_path: results/runs/method
    primary_metric: value_delta
    claim_boundary: test only
""",
        encoding="utf-8",
    )

    try:
        load_experiment_registry(config)
    except ValueError as exc:
        assert "method: comparison experiments must name control_experiment_id" in str(exc)
    else:
        raise AssertionError("missing comparison control should fail")


def test_experiment_registry_loader_rejects_unknown_control(tmp_path) -> None:
    config = tmp_path / "experiments.yaml"
    config.write_text(
        """
schema_version: 1
experiments:
  - experiment_id: method
    checkpoint: 5
    status: planned
    hypothesis_id: planner
    hypothesis: method comparison
    train_split_id: train
    validation_split_id: validation
    test_split_id: test
    dataset_role: validation
    benchmark_protocol_id: synthetic_schema_rich_method_fixture
    model: model
    method: predicted_planner_sql
    oracle_policy: non_oracle_generation
    scorer: scorer
    output_path: results/runs/method
    control_experiment_id: typo_control
    primary_metric: value_delta
    claim_boundary: test only
""",
        encoding="utf-8",
    )

    try:
        load_experiment_registry(config)
    except ValueError as exc:
        assert "method: unknown control_experiment_id 'typo_control'" in str(exc)
    else:
        raise AssertionError("unknown control should fail")


def test_experiment_registry_map_is_keyed_by_stable_id() -> None:
    experiments = experiment_registry_map(REPO_ROOT / "configs" / "experiments.yaml")

    assert experiments["semantic_value_retrieval_vs_direct"]["dataset_role"] == "clean_local_holdout"
    assert experiments["hosted_bird_interact_transfer"]["checkpoint"] == 9
