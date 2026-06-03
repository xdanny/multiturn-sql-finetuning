from __future__ import annotations

import json

from data.synthetic_method_fixtures import (
    SCORER_ONLY_FIELDS,
    build_synthetic_method_fixtures,
    prompt_visible_fixture_input,
    summarize_synthetic_method_fixtures,
    write_synthetic_method_fixture_artifacts,
)


def test_synthetic_method_fixtures_cover_required_multi_turn_failures() -> None:
    fixtures = build_synthetic_method_fixtures()

    assert {fixture["fixture_id"] for fixture in fixtures} == {
        "value_normalization_france",
        "entity_resolution_followup",
        "grain_fanout_bridge",
        "measure_preservation_metric",
        "recovery_empty_result",
    }
    assert all(fixture["reference_sql_visible_to_model"] is False for fixture in fixtures)
    assert all(fixture["leakage_policy"] == "prompt_visible_inputs_only" for fixture in fixtures)
    assert all(fixture["expected_rows"] for fixture in fixtures)

    failure_modes = {
        mode for fixture in fixtures for mode in fixture["failure_modes"]
    }
    assert {
        "value_normalization",
        "entity_resolution",
        "grain_fanout",
        "measure_preservation",
        "recovery",
    } <= failure_modes

    training_targets = {
        target for fixture in fixtures for target in fixture["training_targets"]
    }
    assert {
        "direct_sql_with_value_context",
        "semantic_context",
        "metric_dsl",
        "behavior_recovery",
    } <= training_targets


def test_synthetic_fixture_rows_expose_method_specific_contracts() -> None:
    fixtures = {fixture["fixture_id"]: fixture for fixture in build_synthetic_method_fixtures()}

    value_fixture = fixtures["value_normalization_france"]
    assert value_fixture["conversation"][-1]["user"] == "Only France."
    assert "France" in value_fixture["direct_sql_trap"]["wrong_sql"]
    assert value_fixture["expected_rows"] == [["FR", 125.0]]
    assert "value_index" in value_fixture["required_artifacts"]

    fanout_fixture = fixtures["grain_fanout_bridge"]
    assert fanout_fixture["evaluation_checks"]["duplicate_row_policy"] == "dedupe_bridge_rows"
    assert fanout_fixture["evaluation_checks"]["expected_metric_delta"] == 100.0
    assert fanout_fixture["direct_sql_trap"]["wrong_rows"] != fanout_fixture["expected_rows"]

    measure_fixture = fixtures["measure_preservation_metric"]
    assert measure_fixture["gold_metric_dsl"].startswith("MEASURE(revenue)")
    assert "revenue" in measure_fixture["semantic_model"]["measures"]
    assert measure_fixture["evaluation_checks"]["requires_measure_preservation"] is True

    recovery_fixture = fixtures["recovery_empty_result"]
    assert recovery_fixture["conversation"][-1]["observed_previous_rows"] == []
    assert recovery_fixture["evaluation_checks"]["requires_repair_action"] == "replace_display_value_with_storage_value"


def test_synthetic_method_fixture_summary_counts_artifacts() -> None:
    summary = summarize_synthetic_method_fixtures(build_synthetic_method_fixtures())

    assert summary["fixture_count"] == 5
    assert summary["schema_count"] == 1
    assert summary["failure_mode_counts"]["grain_fanout"] == 1
    assert summary["failure_mode_counts"]["value_normalization"] == 2
    assert summary["training_target_counts"]["metric_dsl"] == 2
    assert summary["required_artifact_counts"]["semantic_model_manifest"] >= 2
    assert summary["prompt_safe_fixture_count"] == 5


def test_prompt_visible_fixture_projection_excludes_scoring_fields() -> None:
    for fixture in build_synthetic_method_fixtures():
        visible = prompt_visible_fixture_input(fixture)

        assert fixture["prompt_visible_input"] == visible
        assert set(SCORER_ONLY_FIELDS).isdisjoint(visible)
        assert "scoring_contract" not in visible
        assert "schema_sql" in visible
        assert "conversation" in visible
        assert "semantic_model" in visible
        assert "reference_sql" in fixture["scoring_contract"]
        assert "expected_rows" in fixture["scoring_contract"]


def test_write_synthetic_method_fixture_artifacts(tmp_path) -> None:
    output_path = tmp_path / "synthetic_method_fixtures.jsonl"
    summary_path = tmp_path / "synthetic_method_fixtures_summary.json"
    manifest_path = tmp_path / "synthetic_method_fixtures.manifest.json"

    manifest = write_synthetic_method_fixture_artifacts(
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    written_manifest = json.loads(manifest_path.read_text())

    assert len(rows) == 5
    assert summary["fixture_count"] == 5
    assert manifest == written_manifest
    assert manifest["artifact_type"] == "synthetic_method_fixture_pack"
    assert manifest["output_sha256"]
    assert manifest["summary_sha256"]
    assert manifest["fixture_count"] == 5
