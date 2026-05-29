from __future__ import annotations

from pathlib import Path

from eval.hypothesis_rollup import build_hypothesis_rollup


def test_build_hypothesis_rollup_keeps_claim_boundary_conservative() -> None:
    payload = build_hypothesis_rollup(repo_root=Path("."))

    assert payload["artifact_type"] == "hypothesis_arm_rollup"
    assert payload["overall_decision"]["status"] == "no_method_promoted"
    assert len(payload["arms"]) == 6
    assert {arm["run_id"] for arm in payload["arms"]} == {
        "metric_dsl_prompt_baseline",
        "metric_dsl_arm_5steps",
        "behavior_recovery_5steps",
        "value_schema_repair_prompt",
        "value_choice_consistency_prompt",
        "alias_column_validity_prompt",
    }
    assert all(arm["evidence_sha256"] for arm in payload["arms"])
    assert all("selected_metrics" in arm for arm in payload["arms"])
    assert any(
        arm["decision"] == "passed_single_synthetic_diagnostic"
        and arm["hypothesis_id"] == "value_schema_repair"
        for arm in payload["arms"]
    )
