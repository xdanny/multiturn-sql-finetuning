from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from train.experiment_matrix import experiment_matrix_summary, load_experiment_matrix

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_experiment_matrix_names_each_hypothesis_and_gate() -> None:
    rows = load_experiment_matrix(repo_root=REPO_ROOT)

    ids = [row["hypothesis_id"] for row in rows]
    assert ids == [
        "direct_sql_sft",
        "planner_first_sql",
        "semantic_value_retrieval",
        "metric_dsl",
        "behavior_recovery",
        "value_schema_repair",
        "hosted_transfer",
    ]
    assert all(row["leakage_checks"] for row in rows)
    assert rows[0]["control_hypothesis_id"] == "base_model"
    assert rows[1]["control_hypothesis_id"] == "direct_sql_sft"
    assert rows[-1]["control_hypothesis_id"] == "best_local_proxy_method"


def test_experiment_matrix_summary_exposes_holdout_boundary() -> None:
    summary = experiment_matrix_summary(repo_root=REPO_ROOT)

    assert summary["hypothesis_count"] == 7
    assert "not a pristine scientific holdout" in summary["holdout_policy"]
    assert "not a result" in summary["claim_boundary"]
    by_id = {row["hypothesis_id"]: row for row in summary["hypotheses"]}
    assert by_id["metric_dsl"]["ready_for_validation"] is True
    assert (
        by_id["value_schema_repair"]["current_status"]
        == "prompt_validation_limit8_no_delta"
    )
    assert by_id["hosted_transfer"]["ready_for_locked_benchmark"] is False
    assert by_id["hosted_transfer"]["current_status"] == "not_started"


def test_experiment_matrix_rejects_forward_control_reference(tmp_path) -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "hypothesis_experiment_matrix.yaml").read_text()
    )
    config["hypotheses"][1]["control_hypothesis_id"] = "behavior_recovery"
    path = tmp_path / "bad_matrix.yaml"
    path.write_text(yaml.safe_dump(config))

    with pytest.raises(ValueError, match="control_hypothesis_id"):
        load_experiment_matrix(path, repo_root=REPO_ROOT)
