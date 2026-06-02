from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.generated_history_recovery_readiness import (
    build_generated_history_recovery_candidates,
    summarize_generated_history_recovery_candidates,
    write_generated_history_recovery_candidate_artifacts,
)


def _dialog(
    *,
    dialog_id: str = "dialog-a",
    split_role: str = "clean_local_holdout",
    assistant_turns: int = 2,
) -> dict:
    messages = [
        {"role": "system", "content": "Return only SQL."},
        {"role": "user", "content": "Question:\nShow revenue by country."},
        {"role": "assistant", "content": "SELECT country, SUM(amount) FROM orders GROUP BY country;"},
    ]
    gold_plans = [
        {
            "query_skeleton": {"select": True, "group_by": True},
            "projection_shape": {"aggregations": ["sum"], "selected_count": 2},
        }
    ]
    if assistant_turns > 1:
        messages.extend(
            [
                {"role": "user", "content": "Question:\nOnly France."},
                {"role": "assistant", "content": "SELECT SUM(amount) FROM orders WHERE country = 'FR';"},
            ]
        )
        gold_plans.append(
            {
                "query_skeleton": {"select": True, "where": True},
                "projection_shape": {"aggregations": ["sum"], "selected_count": 1},
            }
        )
    return {
        "dialog_id": dialog_id,
        "source": "cosql_dev_clean_holdout_v1",
        "history_policy": "gold_sql_teacher_forced",
        "evaluation_mode": "non_oracle_generation",
        "database_id": "store",
        "split_id": "cosql_dev_clean_holdout_v1",
        "split_role": split_role,
        "split_row_id": f"cosql_dev:0001:{dialog_id}",
        "messages": messages,
        "gold_plans": gold_plans,
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_generated_history_recovery_candidates_select_multiturn_clean_holdout(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "clean_holdout.jsonl"
    _write_jsonl(
        input_path,
        [
            _dialog(dialog_id="multi", assistant_turns=2),
            _dialog(dialog_id="single", assistant_turns=1),
        ],
    )

    rows = build_generated_history_recovery_candidates(input_path)

    assert [row["dialog_id"] for row in rows] == ["multi"]
    assert rows[0]["artifact_type"] == "generated_history_recovery_rollout_candidate"
    assert rows[0]["split_role"] == "clean_local_holdout"
    assert rows[0]["history_policy"] == "gold_sql_teacher_forced"
    assert rows[0]["rollout_target_history_policy"] == "model_generated_sql_rollout"
    assert rows[0]["assistant_turn_count"] == 2
    assert rows[0]["recoverable_later_turn_count"] == 1
    assert rows[0]["reference_sql_visible_to_model_prompt"] is False
    assert rows[0]["future_turns_visible_to_model_prompt"] is False
    assert rows[0]["scorer_labels_visible_to_model_prompt"] is False
    assert rows[0]["readiness_blockers"] == [
        "multi-dialog recovery adapter rollout missing"
    ]


def test_generated_history_recovery_candidates_reject_non_holdout_split(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "proxy.jsonl"
    _write_jsonl(input_path, [_dialog(split_role="proxy_dev_seen")])

    with pytest.raises(ValueError, match="split_role=clean_local_holdout"):
        build_generated_history_recovery_candidates(input_path)


def test_generated_history_recovery_summary_records_scope_and_blocker(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "clean_holdout.jsonl"
    _write_jsonl(
        input_path,
        [
            _dialog(dialog_id="a", assistant_turns=2),
            _dialog(dialog_id="b", assistant_turns=3),
        ],
    )
    rows = build_generated_history_recovery_candidates(input_path)

    summary = summarize_generated_history_recovery_candidates(rows, input_path=input_path)

    assert summary["artifact_type"] == "generated_history_recovery_readiness_summary"
    assert summary["candidate_dialog_count"] == 2
    assert summary["candidate_turn_count"] == 4
    assert summary["recoverable_later_turn_count"] == 2
    assert summary["split_roles"] == {"clean_local_holdout": 2}
    assert summary["promotion_status"] == "not_ready"
    assert summary["readiness_blockers"] == {
        "multi-dialog recovery adapter rollout missing": 2
    }
    assert "eval.run_behavior_recovery_comparison" in summary["next_step"]


def test_write_generated_history_recovery_candidate_artifacts(tmp_path: Path) -> None:
    input_path = tmp_path / "clean_holdout.jsonl"
    output_path = tmp_path / "generated_history_recovery_candidates.jsonl"
    summary_path = tmp_path / "generated_history_recovery_readiness.json"
    manifest_path = tmp_path / "generated_history_recovery_candidates.manifest.json"
    _write_jsonl(input_path, [_dialog()])

    manifest = write_generated_history_recovery_candidate_artifacts(
        input_path=input_path,
        output_path=output_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        command=["write-cp8-candidates"],
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert summary["candidate_dialog_count"] == 1
    assert manifest["artifact_type"] == "generated_history_recovery_candidate_manifest"
    assert manifest["candidate_dialog_count"] == 1
    assert manifest["input_sha256"]
    assert manifest["output_sha256"]
    assert manifest["summary_sha256"]
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
