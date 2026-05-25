from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

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

    planner = next(row for row in rows if row["claim_id"] == "planner_lexical_schema_baseline")
    assert planner["claim_status"] == "supported_planner_quality"
    assert planner["macro_planner_score"] == 0.25
    assert planner["allowed_public_claim"] == "planner quality only, not SQL execution"
    assert planner["artifact_valid"] is True


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
    assert row["claim_status"] == "pending"
    assert row["artifact_valid"] is False
    assert row["blocking_reason"] == "predicted_planner output rows missing predicted_planner mode"


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
