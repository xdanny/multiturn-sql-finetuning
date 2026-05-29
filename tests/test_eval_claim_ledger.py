from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from eval.claim_ledger import (
    build_claim_ledger,
    write_claim_ledger,
    write_claim_summary,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_metric_dsl_comparison_case(
    tmp_path: Path,
    *,
    direct_benchmark: str = "metric_dsl_direct_sql",
    direct_output_has_scores: bool = True,
    metric_command: list[str] | None = None,
    direct_output_sha256: str | None = None,
) -> Path:
    metric_input_path = tmp_path / "data" / "metric_dsl.jsonl"
    metric_output_path = tmp_path / "results" / "metric_dsl.jsonl"
    direct_input_path = tmp_path / "data" / "direct_sql.jsonl"
    direct_output_path = tmp_path / "results" / "direct_sql.jsonl"
    _write_jsonl(metric_input_path, [{"id": "metric-1", "predicted_dsl": "MEASURE(revenue)"}])
    _write_jsonl(
        metric_output_path,
        [
            {
                "id": "metric-1",
                "evaluation_mode": "metric_dsl",
                "reference_sql": "SELECT 1",
                "database_path": "metric.sqlite",
                "sql_execution_attempted": True,
                "semantic_model_oracle_derived": False,
            }
        ],
    )
    _write_jsonl(
        direct_input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        ],
    )
    direct_output_row = {"id": "metric-1", "evaluation_mode": "non_oracle_generation"}
    if direct_output_has_scores:
        direct_output_row.update({"value_execution_score": 0.5, "strict_execution_score": 0.5})
    _write_jsonl(direct_output_path, [direct_output_row])
    direct_output_hash = direct_output_sha256 or _sha256(direct_output_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "direct_sql",
                    "benchmark": direct_benchmark,
                    "model_name": "direct-sql-model",
                    "endpoint": "local",
                    "evaluation_mode": "non_oracle_generation",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(direct_input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(direct_input_path),
                    "output_path": str(direct_output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(direct_output_path),
                    "row_count": 1,
                    "metrics": {
                        "value_execution_accuracy": 0.5,
                        "strict_execution_accuracy": 0.5,
                    },
                    "command": ["run"],
                },
                {
                    "schema_version": 1,
                    "run_id": "metric_dsl",
                    "benchmark": "metric_dsl",
                    "model_name": "metric-dsl-model",
                    "endpoint": "offline",
                    "evaluation_mode": "metric_dsl",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(metric_input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(metric_input_path),
                    "output_path": str(metric_output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(metric_output_path),
                    "row_count": 1,
                    "metrics": {
                        "metric_dsl_parse_rate": 1.0,
                        "metric_dsl_compile_rate": 1.0,
                        "compiled_sql_execution_evaluated_rows": 1,
                        "measure_preservation": 1.0,
                        "value_execution_accuracy": 0.6,
                        "strict_execution_accuracy": 0.6,
                        "direct_sql_comparison_run_id": "direct_sql",
                        "direct_sql_model_name": "direct-sql-model",
                        "direct_sql_input_sha256": _sha256(direct_input_path),
                        "direct_sql_output_sha256": direct_output_hash,
                        "direct_sql_value_execution_accuracy": 0.5,
                        "metric_dsl_value_delta_vs_direct_sql": 0.1,
                        "metric_dsl_comparable_row_count": 1,
                    },
                    "command": metric_command
                    or ["run", "# compared-with-direct-sql", "direct_sql"],
                },
            ]
        )
    )
    return manifest_path


def test_build_claim_ledger_separates_proxy_diagnostic_and_pending_claims(tmp_path) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    output_path = tmp_path / "results" / "rescored" / "direct.jsonl"
    oracle_output_path = tmp_path / "results" / "rescored" / "oracle.jsonl"
    classified_path = tmp_path / "results" / "classified" / "direct.jsonl"
    planner_summary_path = tmp_path / "docs" / "planner_summary.json"

    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "SELECT 1"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "SELECT 2"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "history_policy": "gold_sql_teacher_forced",
                "turn_format": "multi_turn_dialog",
                "gold_plans": [{}, {}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a", "value_execution_score": 1.0}])
    _write_jsonl(oracle_output_path, [{"id": "b", "value_execution_score": 1.0}])
    _write_jsonl(
        classified_path,
        [
            {"id": "a", "error_primary": "correct"},
            {"id": "b", "error_primary": "schema_link"},
        ],
    )
    planner_summary_path.parent.mkdir(parents=True, exist_ok=True)
    planner_summary_path.write_text(
        json.dumps(
            {
                "rows": 2,
                "prediction_sources": {"lexical_schema_baseline": 2},
                "macro_planner_score": 0.25,
                "table_f1": 0.5,
                "column_f1": 0.125,
                "selected_count_match": 0.0,
            }
        )
        + "\n"
    )
    manifest_path = tmp_path / "docs" / "manifests.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "direct_sql",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "non_oracle_generation",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 2,
                    "metrics": {
                        "dialog_count": 1,
                        "strict_execution_accuracy": 0.5,
                        "value_execution_accuracy": 1.0,
                        "syntax_accuracy": 1.0,
                    },
                    "command": ["historical-result-rescore"],
                },
                {
                    "schema_version": 1,
                    "run_id": "oracle_sql",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "oracle_planner_diagnostic",
                    "oracle_allowed": True,
                    "prompt_variant": "oracle",
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(oracle_output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(oracle_output_path),
                    "row_count": 2,
                    "metrics": {
                        "dialog_count": 1,
                        "strict_execution_accuracy": 0.75,
                        "value_execution_accuracy": 1.0,
                        "syntax_accuracy": 1.0,
                    },
                    "command": ["historical-result-rescore"],
                },
            ]
        )
        + "\n"
    )

    rows = build_claim_ledger(
        manifest_path=manifest_path,
        repo_root=tmp_path,
        classified_dir=tmp_path / "results" / "classified",
        planner_summary_path=planner_summary_path,
    )

    direct = next(row for row in rows if row["claim_id"] == "direct_sql")
    oracle = next(row for row in rows if row["claim_id"] == "oracle_sql")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}

    assert direct["claim_status"] == "supported_proxy"
    assert direct["allowed_public_claim"] == "local proxy result only"
    assert direct["production_claim_allowed"] is True
    assert direct["can_support_sota_claim"] is False
    assert direct["artifact_valid"] is True
    assert direct["blocking_reason"] is None
    assert direct["input_contract_status"] == "contracted"
    assert direct["history_policy"] == "gold_sql_teacher_forced"
    assert direct["teacher_forced_history"] is True
    assert direct["classified_rows"] == 2
    assert direct["primary_error_counts"] == {"correct": 1, "schema_link": 1}
    assert direct["input_sha256_matches"] is True
    assert direct["output_sha256_matches"] is True

    assert oracle["claim_status"] == "diagnostic_upper_bound"
    assert oracle["allowed_public_claim"] == "oracle planner diagnostic only"
    assert oracle["production_claim_allowed"] is False
    assert oracle["artifact_valid"] is True

    assert "predicted_planner_sql_execution" in pending
    assert "model_generated_history_rollout" in pending
    assert "rollout_beats_teacher_forced_history" in pending
    assert "behavior_recovery_beats_direct_sql" in pending
    assert "semantic_value_retrieval_improves_sql" in pending
    assert "hosted_sota_same_protocol" in pending
    assert "bird_interact_local_vs_hosted" in pending
    assert pending["predicted_planner_sql_execution"]["blocking_reason"] == (
        "no predicted_planner result manifest"
    )
    assert pending["model_generated_history_rollout"]["blocking_reason"] == (
        "no model-generated-history rollout manifest"
    )
    assert pending["rollout_beats_teacher_forced_history"]["blocking_reason"] == (
        "no side-by-side rollout-vs-teacher-forced comparison"
    )
    assert pending["behavior_recovery_beats_direct_sql"]["blocking_reason"] == (
        "no generated-history recovery-vs-direct-SQL comparison"
    )
    assert pending["semantic_value_retrieval_improves_sql"]["blocking_reason"] == (
        "no row-matched semantic value-retrieval SQL comparison"
    )

    planner = next(row for row in rows if row["claim_id"] == "planner_lexical_schema_baseline")
    assert planner["claim_status"] == "supported_planner_quality"
    assert planner["macro_planner_score"] == 0.25
    assert planner["allowed_public_claim"] == "planner quality only, not SQL execution"
    assert planner["artifact_valid"] is True


def test_claim_ledger_includes_value_index_coverage_artifact(tmp_path) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    index_path = tmp_path / "docs" / "value_index.jsonl"
    summary_path = tmp_path / "docs" / "value_index_summary.json"
    manifest_path = tmp_path / "docs" / "value_index.manifest.json"
    result_manifest_path = tmp_path / "docs" / "manifests.json"
    _write_jsonl(input_path, [{"database_id": "store", "messages": []}])
    _write_jsonl(
        index_path,
        [
            {
                "artifact_type": "non_oracle_value_index_entry",
                "database_id": "store",
                "table": "customers",
                "column": "country_code",
                "raw_value": "FR",
                "aliases": ["FR", "fr"],
            }
        ],
    )
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "non_oracle_value_index_summary",
                "index_source": "database_contents",
                "entry_count": 1,
                "database_count": 1,
                "table_count": 1,
                "column_count": 1,
                "alias_count": 2,
                "coverage": {
                    "label_count": 2,
                    "resolved_value_indexed_count": 2,
                    "resolved_value_indexed_rate": 1.0,
                    "mention_alias_indexed_count": 1,
                    "mention_alias_indexed_rate": 0.5,
                },
            }
        )
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact_type": "non_oracle_value_index_v1",
                "index_source": "database_contents",
                "input_path": str(input_path.relative_to(tmp_path)),
                "input_sha256": _sha256(input_path),
                "output_path": str(index_path.relative_to(tmp_path)),
                "output_sha256": _sha256(index_path),
                "summary_path": str(summary_path.relative_to(tmp_path)),
                "summary_sha256": _sha256(summary_path),
                "row_count": 1,
                "database_count": 1,
                "table_count": 1,
                "column_count": 1,
                "label_source": "optional_gold_sql_coverage_eval",
            }
        )
    )
    result_manifest_path.write_text("[]")

    rows = build_claim_ledger(
        manifest_path=result_manifest_path,
        repo_root=tmp_path,
        value_index_manifest_path=manifest_path,
        value_index_summary_path=summary_path,
    )

    value_index = next(row for row in rows if row["claim_id"] == "value_index_coverage")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert value_index["claim_status"] == "supported_value_retrieval_coverage"
    assert value_index["allowed_public_claim"] == "value/entity retrieval coverage only"
    assert value_index["production_claim_allowed"] is False
    assert value_index["resolved_value_indexed_rate"] == 1.0
    assert value_index["mention_alias_indexed_rate"] == 0.5
    assert "semantic_value_retrieval_improves_sql" in pending


def test_hash_mismatch_blocks_supported_claim(tmp_path) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    output_path = tmp_path / "results" / "direct.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "bad_hash",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "non_oracle_generation",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": "not-the-real-hash",
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {"value_execution_accuracy": 1.0},
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    row = rows[0]
    assert row["claim_status"] == "pending"
    assert row["artifact_valid"] is False
    assert row["blocking_reason"] == "manifest hash mismatch"
    assert row["production_claim_allowed"] is False


def test_oracle_markers_block_non_oracle_claim(tmp_path) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    output_path = tmp_path / "results" / "direct.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Oracle SQL planning hints "
                            "(derived from reference SQL; diagnostic only):\n"
                            "Relevant tables: orders\n\nQuestion:\nList orders."
                        ),
                    },
                    {"role": "assistant", "content": "SELECT * FROM orders"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "oracle_leak",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "non_oracle_generation",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {"value_execution_accuracy": 1.0},
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    row = rows[0]
    assert row["claim_status"] == "pending"
    assert row["artifact_valid"] is False
    assert row["blocking_reason"] == "oracle planning hints found in non-oracle artifact"


def test_predicted_planner_manifest_requires_output_rows_with_predicted_mode(tmp_path) -> None:
    input_path = tmp_path / "data" / "predicted.jsonl"
    output_path = tmp_path / "results" / "predicted.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "predicted_planner",
                "gold_plans": [{}],
                "predicted_plans": [{}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a", "evaluation_mode": "non_oracle_generation"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "predicted_bad_output",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "predicted_planner",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {"value_execution_accuracy": 1.0},
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    row = rows[0]
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert row["claim_status"] == "pending"
    assert row["artifact_valid"] is False
    assert row["blocking_reason"] == "predicted_planner output rows missing predicted_planner mode"
    assert "predicted_planner_sql_execution" in pending


def test_predicted_planner_manifest_requires_direct_sql_comparison_to_clear_pending(
    tmp_path,
) -> None:
    input_path = tmp_path / "data" / "predicted.jsonl"
    output_path = tmp_path / "results" / "predicted.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "predicted_planner",
                "gold_plans": [{}],
                "predicted_plans": [{}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a", "evaluation_mode": "predicted_planner"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "predicted_without_comparison",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "predicted_planner",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {"value_execution_accuracy": 1.0},
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "predicted_planner_sql_execution" in pending
    assert pending["predicted_planner_sql_execution"]["blocking_reason"] == (
        "no same-model direct-SQL comparison with positive value delta"
    )


def test_predicted_planner_direct_sql_comparison_must_improve_before_clearing_pending(
    tmp_path,
) -> None:
    input_path = tmp_path / "data" / "predicted.jsonl"
    output_path = tmp_path / "results" / "predicted.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "predicted_planner",
                "gold_plans": [{}],
                "predicted_plans": [{}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a", "evaluation_mode": "predicted_planner"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "predicted_no_gain",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "predicted_planner",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {
                        "value_execution_accuracy": 0.4,
                        "direct_sql_comparison_run_id": "direct",
                        "direct_sql_model_name": "local-9b",
                        "direct_sql_value_execution_accuracy": 0.5,
                        "predicted_planner_value_delta_vs_direct_sql": -0.1,
                    },
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "predicted_planner_sql_execution" in pending


def test_predicted_planner_positive_direct_sql_delta_clears_pending_claim(tmp_path) -> None:
    input_path = tmp_path / "data" / "predicted.jsonl"
    output_path = tmp_path / "results" / "predicted.jsonl"
    direct_input_path = tmp_path / "data" / "direct.jsonl"
    direct_output_path = tmp_path / "results" / "direct.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "predicted_planner",
                "gold_plans": [{}],
                "predicted_plans": [{}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a", "evaluation_mode": "predicted_planner"}])
    _write_jsonl(
        direct_input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        ],
    )
    _write_jsonl(direct_output_path, [{"id": "a", "evaluation_mode": "non_oracle_generation"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "direct",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "non_oracle_generation",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(direct_input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(direct_input_path),
                    "output_path": str(direct_output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(direct_output_path),
                    "row_count": 1,
                    "metrics": {
                        "value_execution_accuracy": 0.5,
                        "strict_execution_accuracy": 0.5,
                    },
                    "command": ["run"],
                },
                {
                    "schema_version": 1,
                    "run_id": "predicted_gain",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "predicted_planner",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {
                        "value_execution_accuracy": 0.6,
                        "direct_sql_comparison_run_id": "direct",
                        "direct_sql_model_name": "local-9b",
                        "direct_sql_input_sha256": _sha256(direct_input_path),
                        "direct_sql_output_sha256": _sha256(direct_output_path),
                        "direct_sql_value_execution_accuracy": 0.5,
                        "predicted_planner_value_delta_vs_direct_sql": 0.1,
                        "direct_sql_comparable_row_count": 1,
                    },
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "predicted_planner_sql_execution" not in pending


def test_predicted_planner_positive_delta_without_direct_manifest_stays_pending(tmp_path) -> None:
    input_path = tmp_path / "data" / "predicted.jsonl"
    output_path = tmp_path / "results" / "predicted.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "predicted_planner",
                "gold_plans": [{}],
                "predicted_plans": [{}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a", "evaluation_mode": "predicted_planner"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "predicted_unbacked_gain",
                    "benchmark": "prepared",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "predicted_planner",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {
                        "value_execution_accuracy": 0.6,
                        "direct_sql_comparison_run_id": "direct",
                        "direct_sql_model_name": "local-9b",
                        "direct_sql_input_sha256": "direct-input",
                        "direct_sql_output_sha256": "direct-output",
                        "direct_sql_value_execution_accuracy": 0.5,
                        "predicted_planner_value_delta_vs_direct_sql": 0.1,
                        "direct_sql_comparable_row_count": 1,
                    },
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "predicted_planner_sql_execution" in pending
    assert pending["predicted_planner_sql_execution"]["blocking_reason"] == (
        "no same-model direct-SQL comparison with positive value delta"
    )


def test_metric_dsl_manifest_clears_metric_eval_pending_but_not_direct_sql_comparison(
    tmp_path,
) -> None:
    input_path = tmp_path / "data" / "metric_dsl.jsonl"
    output_path = tmp_path / "results" / "metric_dsl.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "metric-1",
                "predicted_dsl": "MEASURE(revenue)",
                "gold_dsl": "MEASURE(revenue)",
                "semantic_model": {"base_table": "orders", "measures": {"revenue": {"sql": "1"}}},
            }
        ],
    )
    _write_jsonl(
        output_path,
        [
            {
                "id": "metric-1",
                "evaluation_mode": "metric_dsl",
                "reference_sql": "SELECT 1",
                "database_path": "metric.sqlite",
                "sql_execution_attempted": True,
                "semantic_model_oracle_derived": False,
            }
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "metric_dsl",
                    "benchmark": "metric_dsl",
                    "model_name": "metric-dsl-model",
                    "endpoint": "offline",
                    "evaluation_mode": "metric_dsl",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {
                        "metric_dsl_parse_rate": 1.0,
                        "metric_dsl_compile_rate": 1.0,
                        "compiled_sql_execution_evaluated_rows": 1,
                        "measure_preservation": 1.0,
                        "value_execution_accuracy": 0.6,
                        "strict_execution_accuracy": 0.6,
                    },
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    metric = next(row for row in rows if row["claim_id"] == "metric_dsl")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert metric["claim_status"] == "supported_metric_dsl_quality"
    assert "metric_dsl_evaluation_manifest" not in pending
    assert "metric_dsl_beats_direct_sql" in pending


def test_metric_dsl_manifest_requires_quality_metrics_for_supported_status(tmp_path) -> None:
    input_path = tmp_path / "data" / "metric_dsl.jsonl"
    output_path = tmp_path / "results" / "metric_dsl.jsonl"
    _write_jsonl(input_path, [{"id": "metric-1", "predicted_dsl": "MEASURE(revenue)"}])
    _write_jsonl(
        output_path,
        [
            {
                "id": "metric-1",
                "evaluation_mode": "metric_dsl",
                "reference_sql": "SELECT 1",
                "database_path": "metric.sqlite",
                "sql_execution_attempted": True,
                "semantic_model_oracle_derived": False,
            }
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "metric_dsl_without_quality",
                    "benchmark": "metric_dsl",
                    "model_name": "metric-dsl-model",
                    "endpoint": "offline",
                    "evaluation_mode": "metric_dsl",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {
                        "metric_dsl_parse_rate": 1.0,
                        "metric_dsl_compile_rate": 0.0,
                        "compiled_sql_execution_evaluated_rows": 1,
                        "measure_preservation": 1.0,
                        "value_execution_accuracy": 0.6,
                    },
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    metric = next(row for row in rows if row["claim_id"] == "metric_dsl_without_quality")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert metric["claim_status"] == "pending"
    assert metric["artifact_valid"] is False
    assert metric["blocking_reason"] == "metric_dsl manifest quality metrics are incomplete"
    assert "metric_dsl_evaluation_manifest" in pending


def test_metric_dsl_positive_comparison_requires_matching_direct_manifest(tmp_path) -> None:
    input_path = tmp_path / "data" / "metric_dsl.jsonl"
    output_path = tmp_path / "results" / "metric_dsl.jsonl"
    _write_jsonl(input_path, [{"id": "metric-1", "predicted_dsl": "MEASURE(revenue)"}])
    _write_jsonl(
        output_path,
        [
            {
                "id": "metric-1",
                "evaluation_mode": "metric_dsl",
                "reference_sql": "SELECT 1",
                "database_path": "metric.sqlite",
                "sql_execution_attempted": True,
                "semantic_model_oracle_derived": False,
            }
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "metric_dsl",
                    "benchmark": "metric_dsl",
                    "model_name": "metric-dsl-model",
                    "endpoint": "offline",
                    "evaluation_mode": "metric_dsl",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {
                        "metric_dsl_parse_rate": 1.0,
                        "metric_dsl_compile_rate": 1.0,
                        "compiled_sql_execution_evaluated_rows": 1,
                        "measure_preservation": 1.0,
                        "value_execution_accuracy": 0.6,
                        "strict_execution_accuracy": 0.6,
                        "direct_sql_comparison_run_id": "direct_sql",
                        "direct_sql_model_name": "direct-sql-model",
                        "direct_sql_input_sha256": "direct-input",
                        "direct_sql_output_sha256": "direct-output",
                        "direct_sql_value_execution_accuracy": 0.5,
                        "metric_dsl_value_delta_vs_direct_sql": 0.1,
                        "metric_dsl_comparable_row_count": 1,
                    },
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "metric_dsl_beats_direct_sql" in pending


def test_metric_dsl_positive_comparison_requires_comparer_provenance(tmp_path) -> None:
    manifest_path = _write_metric_dsl_comparison_case(
        tmp_path,
        metric_command=["run"],
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "metric_dsl_beats_direct_sql" in pending
    assert pending["metric_dsl_beats_direct_sql"]["blocking_reason"] == (
        "no non-oracle direct-SQL comparison with positive metric-DSL value delta"
    )


def test_metric_dsl_positive_comparison_requires_matching_direct_hashes(tmp_path) -> None:
    manifest_path = _write_metric_dsl_comparison_case(
        tmp_path,
        direct_output_sha256="wrong-direct-output-hash",
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "metric_dsl_beats_direct_sql" in pending


def test_metric_dsl_positive_comparison_requires_metric_heavy_direct_benchmark(tmp_path) -> None:
    manifest_path = _write_metric_dsl_comparison_case(
        tmp_path,
        direct_benchmark="prepared",
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "metric_dsl_beats_direct_sql" in pending


def test_metric_dsl_positive_comparison_requires_scored_direct_output_rows(tmp_path) -> None:
    manifest_path = _write_metric_dsl_comparison_case(
        tmp_path,
        direct_output_has_scores=False,
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "metric_dsl_beats_direct_sql" in pending


def test_metric_dsl_positive_comparison_clears_direct_sql_pending(tmp_path) -> None:
    manifest_path = _write_metric_dsl_comparison_case(
        tmp_path,
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "metric_dsl_evaluation_manifest" not in pending
    assert "metric_dsl_beats_direct_sql" not in pending


def test_claim_ledger_can_read_metric_dsl_from_separate_manifest_file(tmp_path) -> None:
    combined_manifest_path = _write_metric_dsl_comparison_case(tmp_path)
    manifests = json.loads(combined_manifest_path.read_text())
    direct_manifest_path = tmp_path / "direct_manifest.json"
    metric_manifest_path = tmp_path / "metric_manifest.json"
    direct_manifest_path.write_text(json.dumps(manifests[0]))
    metric_manifest_path.write_text(json.dumps(manifests[1]))

    rows = build_claim_ledger(
        manifest_paths=(direct_manifest_path, metric_manifest_path),
        repo_root=tmp_path,
    )

    direct = next(row for row in rows if row["claim_id"] == "direct_sql")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert direct["claim_status"] == "supported_method_control"
    assert direct["production_claim_allowed"] is False
    assert "metric_dsl_evaluation_manifest" not in pending
    assert "metric_dsl_beats_direct_sql" not in pending


def _write_hosted_manifest_case(
    tmp_path: Path,
    *,
    metrics: dict,
    oracle_allowed: bool = False,
    evaluation_mode: str = "non_oracle_generation",
    include_matching_local: bool = True,
    local_value_accuracy: float = 1.0,
    hosted_value_accuracy: float | None = None,
    local_compared_to_hosted: bool = False,
) -> Path:
    input_path = tmp_path / "data" / "eval.jsonl"
    output_path = tmp_path / "results" / "hosted.jsonl"
    local_output_path = tmp_path / "results" / "local.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": evaluation_mode,
                "gold_plans": [{}],
            }
        ],
    )
    output_rows = [
        {
            "id": "a",
            "evaluation_mode": evaluation_mode,
            "value_execution_score": 1.0,
            "strict_execution_score": 1.0,
        }
    ]
    _write_jsonl(output_path, output_rows)
    _write_jsonl(local_output_path, output_rows)
    manifests = []
    if include_matching_local:
        local_metrics = {"value_execution_accuracy": local_value_accuracy}
        hosted_value = (
            float(metrics.get("value_execution_accuracy", local_value_accuracy))
            if hosted_value_accuracy is None
            else hosted_value_accuracy
        )
        if local_compared_to_hosted:
            local_metrics.update(
                {
                    "hosted_comparison_run_id": "hosted_baseline",
                    "hosted_model_name": "frontier-model",
                    "hosted_input_sha256": _sha256(input_path),
                    "hosted_output_sha256": _sha256(output_path),
                    "hosted_value_execution_accuracy": hosted_value,
                    "hosted_strict_execution_accuracy": metrics.get(
                        "strict_execution_accuracy", hosted_value
                    ),
                    "hosted_mean_latency_ms": metrics.get("mean_latency_ms")
                    or metrics.get("mean_generation_latency_ms"),
                    "hosted_total_cost_usd": metrics.get("total_cost_usd")
                    or metrics.get("estimated_cost_usd"),
                    "local_value_delta_vs_hosted": local_value_accuracy - hosted_value,
                    "local_strict_delta_vs_hosted": (
                        local_value_accuracy - float(metrics.get("strict_execution_accuracy", hosted_value))
                    ),
                    "hosted_comparable_row_count": 1,
                }
            )
        manifests.append(
            {
                "schema_version": 1,
                "run_id": "local_baseline",
                "benchmark": "prepared",
                "model_name": "local-9b",
                "endpoint": "http://127.0.0.1:8000/v1",
                "evaluation_mode": evaluation_mode,
                "oracle_allowed": False,
                "prompt_variant": None,
                "input_path": str(input_path.relative_to(tmp_path)),
                "input_sha256": _sha256(input_path),
                "output_path": str(local_output_path.relative_to(tmp_path)),
                "output_sha256": _sha256(local_output_path),
                "row_count": 1,
                "metrics": local_metrics,
                "command": ["run-local"]
                + (
                    ["# compared-with-hosted", "hosted_baseline"]
                    if local_compared_to_hosted
                    else []
                ),
            }
        )
    manifests.append(
        {
            "schema_version": 1,
            "run_id": "hosted_baseline",
            "benchmark": "prepared",
            "model_name": "frontier-model",
            "endpoint": "https://api.example.test/v1",
            "evaluation_mode": evaluation_mode,
            "oracle_allowed": oracle_allowed,
            "prompt_variant": None,
            "input_path": str(input_path.relative_to(tmp_path)),
            "input_sha256": _sha256(input_path),
            "output_path": str(output_path.relative_to(tmp_path)),
            "output_sha256": _sha256(output_path),
            "row_count": 1,
            "metrics": metrics,
            "command": ["run"],
        }
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifests))
    return manifest_path


def test_hosted_non_oracle_manifest_clears_pending_claim_and_marks_sota_support(
    tmp_path,
) -> None:
    manifest_path = _write_hosted_manifest_case(
        tmp_path,
        metrics={
            "value_execution_accuracy": 1.0,
            "mean_latency_ms": 500.0,
            "total_cost_usd": 0.25,
        },
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    hosted = next(row for row in rows if row["claim_id"] == "hosted_baseline")
    local = next(row for row in rows if row["claim_id"] == "local_baseline")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert hosted["artifact_valid"] is True
    assert hosted["production_claim_allowed"] is True
    assert hosted["can_support_sota_claim"] is False
    assert local["can_support_sota_claim"] is False
    assert "hosted_sota_same_protocol" not in pending
    assert "local_beats_hosted_same_protocol" in pending


def test_local_vs_hosted_positive_comparison_clears_outperformance_claim(
    tmp_path,
) -> None:
    manifest_path = _write_hosted_manifest_case(
        tmp_path,
        metrics={
            "value_execution_accuracy": 0.6,
            "strict_execution_accuracy": 0.5,
            "mean_latency_ms": 500.0,
            "total_cost_usd": 0.25,
        },
        local_value_accuracy=0.8,
        local_compared_to_hosted=True,
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    local = next(row for row in rows if row["claim_id"] == "local_baseline")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert local["can_support_sota_claim"] is True
    assert local["hosted_comparison_run_id"] == "hosted_baseline"
    assert local["hosted_model_name"] == "frontier-model"
    assert local["hosted_value_execution_accuracy"] == 0.6
    assert local["local_value_delta_vs_hosted"] == pytest.approx(0.2)
    assert "hosted_sota_same_protocol" not in pending
    assert "local_beats_hosted_same_protocol" not in pending


def test_local_vs_hosted_comparison_requires_positive_value_delta(tmp_path) -> None:
    manifest_path = _write_hosted_manifest_case(
        tmp_path,
        metrics={
            "value_execution_accuracy": 0.8,
            "strict_execution_accuracy": 0.8,
            "mean_latency_ms": 500.0,
            "total_cost_usd": 0.25,
        },
        local_value_accuracy=0.6,
        local_compared_to_hosted=True,
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    local = next(row for row in rows if row["claim_id"] == "local_baseline")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert local["can_support_sota_claim"] is False
    assert "hosted_sota_same_protocol" not in pending
    assert "local_beats_hosted_same_protocol" in pending


def test_hosted_non_oracle_manifest_requires_matching_local_protocol(tmp_path) -> None:
    manifest_path = _write_hosted_manifest_case(
        tmp_path,
        metrics={
            "value_execution_accuracy": 1.0,
            "mean_latency_ms": 500.0,
            "total_cost_usd": 0.25,
        },
        include_matching_local=False,
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    hosted = next(row for row in rows if row["claim_id"] == "hosted_baseline")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert hosted["artifact_valid"] is True
    assert hosted["can_support_sota_claim"] is False
    assert "hosted_sota_same_protocol" in pending
    assert "local_beats_hosted_same_protocol" in pending


def test_hosted_non_oracle_manifest_requires_execution_metric(tmp_path) -> None:
    manifest_path = _write_hosted_manifest_case(
        tmp_path,
        metrics={
            "mean_latency_ms": 500.0,
            "total_cost_usd": 0.25,
        },
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    hosted = next(row for row in rows if row["claim_id"] == "hosted_baseline")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert hosted["artifact_valid"] is True
    assert hosted["can_support_sota_claim"] is False
    assert "hosted_sota_same_protocol" in pending


def test_hosted_manifest_accepts_generation_latency_metric_alias(tmp_path) -> None:
    manifest_path = _write_hosted_manifest_case(
        tmp_path,
        metrics={
            "value_execution_accuracy": 1.0,
            "mean_generation_latency_ms": 500.0,
            "total_cost_usd": 0.25,
        },
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "hosted_sota_same_protocol" not in pending
    assert "local_beats_hosted_same_protocol" in pending


def test_hosted_pending_claim_requires_cost_and_latency_metrics(tmp_path) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    output_path = tmp_path / "results" / "hosted.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "hosted_without_cost",
                    "benchmark": "prepared",
                    "model_name": "frontier-model",
                    "endpoint": "https://api.example.test/v1",
                    "evaluation_mode": "non_oracle_generation",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {"value_execution_accuracy": 1.0},
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "hosted_sota_same_protocol" in pending
    assert pending["hosted_sota_same_protocol"]["blocking_reason"] == (
        "no same-protocol hosted-model manifest"
    )


def test_hosted_oracle_diagnostic_does_not_clear_hosted_sota_pending_claim(tmp_path) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    output_path = tmp_path / "results" / "hosted_oracle.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "oracle_planner_diagnostic",
                "uses_oracle_planning_hints": True,
                "planning_label_source": "gold_reference_sql",
                "gold_plans": [{}],
            }
        ],
    )
    _write_jsonl(output_path, [{"id": "a", "evaluation_mode": "oracle_planner_diagnostic"}])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "hosted_oracle",
                    "benchmark": "prepared",
                    "model_name": "frontier-model",
                    "endpoint": "https://api.example.test/v1",
                    "evaluation_mode": "oracle_planner_diagnostic",
                    "oracle_allowed": True,
                    "prompt_variant": "oracle",
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {
                        "value_execution_accuracy": 1.0,
                        "mean_latency_ms": 500.0,
                        "total_cost_usd": 0.25,
                    },
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    oracle = next(row for row in rows if row["claim_id"] == "hosted_oracle")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert oracle["claim_status"] == "diagnostic_upper_bound"
    assert oracle["production_claim_allowed"] is False
    assert "hosted_sota_same_protocol" in pending


def test_rollout_manifest_clears_generated_history_pending_claim(tmp_path) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    output_path = tmp_path / "results" / "rollout.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        ],
    )
    _write_jsonl(
        output_path,
        [
            {
                "id": "dialog-a:0",
                "evaluation_mode": "non_oracle_generation",
                "history_policy": "model_generated_sql_rollout",
            }
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "rollout",
                    "benchmark": "prepared_rollout",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "non_oracle_generation",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {
                        "value_execution_accuracy": 1.0,
                        "history_policy": "model_generated_sql_rollout",
                    },
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    rollout = next(row for row in rows if row["claim_id"] == "rollout")
    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert rollout["claim_status"] == "supported_proxy"
    assert rollout["history_policy"] == "model_generated_sql_rollout"
    assert "model_generated_history_rollout" not in pending
    assert "rollout_beats_teacher_forced_history" in pending


def test_rollout_comparison_must_improve_before_clearing_behavior_pending_claim(
    tmp_path,
) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    output_path = tmp_path / "results" / "rollout.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        ],
    )
    _write_jsonl(
        output_path,
        [
            {
                "id": "dialog-a:0",
                "evaluation_mode": "non_oracle_generation",
                "history_policy": "model_generated_sql_rollout",
            }
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "schema_version": 1,
                    "run_id": "rollout_no_gain",
                    "benchmark": "prepared_rollout",
                    "model_name": "local-9b",
                    "endpoint": "local",
                    "evaluation_mode": "non_oracle_generation",
                    "oracle_allowed": False,
                    "prompt_variant": None,
                    "input_path": str(input_path.relative_to(tmp_path)),
                    "input_sha256": _sha256(input_path),
                    "output_path": str(output_path.relative_to(tmp_path)),
                    "output_sha256": _sha256(output_path),
                    "row_count": 1,
                    "metrics": {
                        "value_execution_accuracy": 0.4,
                        "history_policy": "model_generated_sql_rollout",
                        "teacher_forced_comparison_run_id": "teacher_forced",
                        "teacher_forced_input_sha256": _sha256(input_path),
                        "teacher_forced_model_name": "local-9b",
                        "teacher_forced_value_execution_accuracy": 0.5,
                        "rollout_value_delta_vs_teacher_forced": -0.1,
                    },
                    "command": ["run"],
                }
            ]
        )
    )

    rows = build_claim_ledger(manifest_path=manifest_path, repo_root=tmp_path)

    pending = {row["claim_id"]: row for row in rows if row["claim_status"] == "pending"}
    assert "model_generated_history_rollout" not in pending
    assert "rollout_beats_teacher_forced_history" in pending


def test_rollout_comparison_requires_referenced_teacher_forced_manifest(
    tmp_path,
) -> None:
    input_path = tmp_path / "data" / "eval.jsonl"
    rollout_output_path = tmp_path / "results" / "rollout.jsonl"
    teacher_output_path = tmp_path / "results" / "teacher.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "SELECT 1"},
                ],
                "evaluation_mode": "non_oracle_generation",
                "gold_plans": [{}],
            }
        ],
    )
    _write_jsonl(
        rollout_output_path,
        [
            {
                "id": "dialog-a:0",
                "evaluation_mode": "non_oracle_generation",
                "history_policy": "model_generated_sql_rollout",
                "value_execution_score": 0.8,
                "strict_execution_score": 0.8,
            }
        ],
    )
    _write_jsonl(
        teacher_output_path,
        [
            {
                "id": "dialog-a:0",
                "evaluation_mode": "non_oracle_generation",
                "history_policy": "gold_sql_teacher_forced",
                "value_execution_score": 0.5,
                "strict_execution_score": 0.5,
            }
        ],
    )
    rollout_manifest = {
        "schema_version": 1,
        "run_id": "rollout_gain",
        "benchmark": "prepared_rollout",
        "model_name": "local-9b",
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "prompt_variant": None,
        "input_path": str(input_path.relative_to(tmp_path)),
        "input_sha256": _sha256(input_path),
        "output_path": str(rollout_output_path.relative_to(tmp_path)),
        "output_sha256": _sha256(rollout_output_path),
        "row_count": 1,
        "metrics": {
            "value_execution_accuracy": 0.8,
            "strict_execution_accuracy": 0.8,
            "history_policy": "model_generated_sql_rollout",
            "teacher_forced_comparison_run_id": "teacher_forced",
            "teacher_forced_input_sha256": _sha256(input_path),
            "teacher_forced_model_name": "local-9b",
            "teacher_forced_value_execution_accuracy": 0.5,
            "rollout_value_delta_vs_teacher_forced": 0.3,
            "teacher_forced_comparable_row_count": 1,
        },
        "command": ["run", "# compared-with", "teacher_forced"],
    }
    teacher_manifest = {
        "schema_version": 1,
        "run_id": "teacher_forced",
        "benchmark": "prepared",
        "model_name": "local-9b",
        "endpoint": "local",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "prompt_variant": None,
        "input_path": str(input_path.relative_to(tmp_path)),
        "input_sha256": _sha256(input_path),
        "output_path": str(teacher_output_path.relative_to(tmp_path)),
        "output_sha256": _sha256(teacher_output_path),
        "row_count": 1,
        "metrics": {
            "value_execution_accuracy": 0.5,
            "strict_execution_accuracy": 0.5,
            "history_policy": "gold_sql_teacher_forced",
        },
        "command": ["run"],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([rollout_manifest]) + "\n")

    rows_without_teacher = build_claim_ledger(
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )
    pending_without_teacher = {
        row["claim_id"] for row in rows_without_teacher if row["claim_status"] == "pending"
    }
    assert "rollout_beats_teacher_forced_history" in pending_without_teacher

    manifest_path.write_text(json.dumps([teacher_manifest, rollout_manifest]) + "\n")
    rows_with_teacher = build_claim_ledger(
        manifest_path=manifest_path,
        repo_root=tmp_path,
    )
    pending_with_teacher = {
        row["claim_id"] for row in rows_with_teacher if row["claim_status"] == "pending"
    }
    assert "rollout_beats_teacher_forced_history" not in pending_with_teacher


def test_planner_summary_with_oracle_prompt_rows_is_pending(tmp_path) -> None:
    planner_summary_path = tmp_path / "planner.json"
    planner_summary_path.write_text(
        json.dumps(
            {
                "rows": 1,
                "oracle_prompt_rows": 1,
                "macro_planner_score": 1.0,
                "table_f1": 1.0,
                "column_f1": 1.0,
                "selected_count_match": 1.0,
            }
        )
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]\n")

    rows = build_claim_ledger(
        manifest_path=manifest_path,
        repo_root=tmp_path,
        planner_summary_path=planner_summary_path,
    )

    planner = next(row for row in rows if row["claim_id"] == "planner_lexical_schema_baseline")
    assert planner["claim_status"] == "pending"
    assert planner["artifact_valid"] is False
    assert planner["blocking_reason"] == "planner summary contains oracle prompt rows"


def test_write_claim_ledger_and_summary(tmp_path) -> None:
    rows = [
        {
            "claim_id": "a",
            "claim_status": "supported_proxy",
            "evaluation_mode": "non_oracle_generation",
            "allowed_public_claim": "local proxy result only",
            "blocking_reason": None,
            "required_artifact": None,
            "value_execution_accuracy": 0.6,
            "strict_execution_accuracy": 0.4,
            "row_count": 10,
            "dialog_count": 3,
        },
        {
            "claim_id": "b",
            "claim_status": "diagnostic_upper_bound",
            "evaluation_mode": "oracle_planner_diagnostic",
            "allowed_public_claim": "oracle planner diagnostic only",
            "blocking_reason": None,
            "required_artifact": None,
            "value_execution_accuracy": 0.9,
            "strict_execution_accuracy": 0.8,
            "row_count": 10,
            "dialog_count": 3,
        },
    ]

    jsonl_path = tmp_path / "ledger.jsonl"
    summary_path = tmp_path / "summary.csv"
    write_claim_ledger(rows, jsonl_path)
    write_claim_summary(rows, summary_path)

    written_rows = [json.loads(line) for line in jsonl_path.read_text().splitlines()]
    assert written_rows == rows
    assert b"\r\n" not in summary_path.read_bytes()

    with summary_path.open() as f:
        summary = list(csv.DictReader(f))

    assert summary == [
        {
            "claim_id": "a",
            "claim_status": "supported_proxy",
            "evaluation_mode": "non_oracle_generation",
            "allowed_public_claim": "local proxy result only",
            "blocking_reason": "",
            "required_artifact": "",
            "value_execution_accuracy": "0.6",
            "strict_execution_accuracy": "0.4",
            "row_count": "10",
            "dialog_count": "3",
        },
        {
            "claim_id": "b",
            "claim_status": "diagnostic_upper_bound",
            "evaluation_mode": "oracle_planner_diagnostic",
            "allowed_public_claim": "oracle planner diagnostic only",
            "blocking_reason": "",
            "required_artifact": "",
            "value_execution_accuracy": "0.9",
            "strict_execution_accuracy": "0.8",
            "row_count": "10",
            "dialog_count": "3",
        },
    ]


def test_summary_includes_pending_rows(tmp_path) -> None:
    rows = [
        {
            "claim_id": "predicted_planner_sql_execution",
            "claim_status": "pending",
            "evaluation_mode": "predicted_planner",
            "allowed_public_claim": "pending predicted-planner SQL execution",
            "blocking_reason": "no predicted_planner result manifest",
            "required_artifact": "same-protocol endpoint SQL result manifest",
        }
    ]
    summary_path = tmp_path / "summary.csv"

    write_claim_summary(rows, summary_path)

    with summary_path.open() as f:
        summary = list(csv.DictReader(f))
    assert summary == [
        {
            "claim_id": "predicted_planner_sql_execution",
            "claim_status": "pending",
            "evaluation_mode": "predicted_planner",
            "allowed_public_claim": "pending predicted-planner SQL execution",
            "blocking_reason": "no predicted_planner result manifest",
            "required_artifact": "same-protocol endpoint SQL result manifest",
            "value_execution_accuracy": "",
            "strict_execution_accuracy": "",
            "row_count": "",
            "dialog_count": "",
        }
    ]
