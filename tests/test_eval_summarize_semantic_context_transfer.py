from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.summarize_semantic_context_transfer import (
    summarize_semantic_context_transfer_evidence,
    summarize_semantic_context_transfer_evidence_files,
)


def _context_comparison(
    *,
    run_id: str,
    model_name: str,
    normal_value: float,
    semantic_value: float,
    normal_strict: float = 0.5,
    semantic_strict: float = 0.5,
    normal_syntax: float = 1.0,
    semantic_syntax: float = 1.0,
    row_count: int = 43,
) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "semantic_context_transfer_comparison",
        "run_id": run_id,
        "benchmark": "prepared_rollout",
        "model_name": model_name,
        "endpoint": "https://example.test/v1",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "row_count": row_count,
        "metrics": {
            "history_policy": "model_generated_sql_rollout",
            "value_execution_accuracy": semantic_value,
            "strict_execution_accuracy": semantic_strict,
            "syntax_accuracy": semantic_syntax,
            "normal_context_value_execution_accuracy": normal_value,
            "normal_context_strict_execution_accuracy": normal_strict,
            "normal_context_syntax_accuracy": normal_syntax,
            "semantic_context_value_delta_vs_normal": semantic_value - normal_value,
            "semantic_context_strict_delta_vs_normal": semantic_strict - normal_strict,
            "semantic_context_syntax_delta_vs_normal": semantic_syntax - normal_syntax,
            "semantic_context_transfer_comparable_row_count": row_count,
            "semantic_context_helped": semantic_value > normal_value,
        },
    }


def _hosted_comparison(
    *,
    run_id: str,
    local_value: float,
    hosted_value: float,
    local_strict: float = 0.5,
    hosted_strict: float = 0.6,
    row_count: int = 43,
) -> dict:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "benchmark": "prepared_rollout",
        "model_name": "local",
        "endpoint": "http://127.0.0.1:8000/v1",
        "evaluation_mode": "non_oracle_generation",
        "oracle_allowed": False,
        "row_count": row_count,
        "metrics": {
            "history_policy": "model_generated_sql_rollout",
            "value_execution_accuracy": local_value,
            "strict_execution_accuracy": local_strict,
            "hosted_value_execution_accuracy": hosted_value,
            "hosted_strict_execution_accuracy": hosted_strict,
            "local_value_delta_vs_hosted": local_value - hosted_value,
            "local_strict_delta_vs_hosted": local_strict - hosted_strict,
            "hosted_comparable_row_count": row_count,
        },
    }


def test_summarize_semantic_context_transfer_marks_gap_narrowed() -> None:
    summary = summarize_semantic_context_transfer_evidence(
        local_context_comparison=_context_comparison(
            run_id="local-context",
            model_name="local-lora",
            normal_value=0.50,
            semantic_value=0.65,
        ),
        hosted_context_comparison=_context_comparison(
            run_id="hosted-context",
            model_name="sonnet",
            normal_value=0.70,
            semantic_value=0.72,
        ),
        normal_local_vs_hosted_comparison=_hosted_comparison(
            run_id="normal-gap",
            local_value=0.50,
            hosted_value=0.70,
        ),
        semantic_local_vs_hosted_comparison=_hosted_comparison(
            run_id="semantic-gap",
            local_value=0.65,
            hosted_value=0.72,
        ),
    )

    assert summary["artifact_type"] == "semantic_context_transfer_evidence_summary"
    assert summary["comparable_row_count"] == 43
    assert summary["local_semantic_value_delta_vs_normal"] == pytest.approx(0.15)
    assert summary["hosted_semantic_value_delta_vs_normal"] == pytest.approx(0.02)
    assert summary["normal_context_hosted_value_gap"] == pytest.approx(0.20)
    assert summary["semantic_context_hosted_value_gap"] == pytest.approx(0.07)
    assert summary["semantic_context_hosted_value_gap_delta_vs_normal"] == pytest.approx(
        -0.13
    )
    assert summary["local_semantic_delta_exceeds_hosted_delta"] is True
    assert summary["hosted_value_gap_narrowed"] is True
    assert summary["promotion_status"] == "gap_narrowed"
    assert summary["promotion_ready"] is True


def test_summarize_semantic_context_transfer_keeps_local_win_without_gap_claim() -> None:
    summary = summarize_semantic_context_transfer_evidence(
        local_context_comparison=_context_comparison(
            run_id="local-context",
            model_name="local-lora",
            normal_value=0.50,
            semantic_value=0.55,
        ),
        hosted_context_comparison=_context_comparison(
            run_id="hosted-context",
            model_name="sonnet",
            normal_value=0.70,
            semantic_value=0.90,
        ),
        normal_local_vs_hosted_comparison=_hosted_comparison(
            run_id="normal-gap",
            local_value=0.50,
            hosted_value=0.70,
        ),
        semantic_local_vs_hosted_comparison=_hosted_comparison(
            run_id="semantic-gap",
            local_value=0.55,
            hosted_value=0.90,
        ),
    )

    assert summary["local_semantic_context_helped"] is True
    assert summary["hosted_value_gap_narrowed"] is False
    assert summary["local_semantic_delta_exceeds_hosted_delta"] is False
    assert summary["promotion_status"] == "local_context_helped_gap_not_narrowed"
    assert summary["promotion_ready"] is False


def test_summarize_semantic_context_transfer_rejects_row_count_mismatch() -> None:
    with pytest.raises(ValueError, match="same row count"):
        summarize_semantic_context_transfer_evidence(
            local_context_comparison=_context_comparison(
                run_id="local-context",
                model_name="local-lora",
                normal_value=0.50,
                semantic_value=0.65,
                row_count=42,
            ),
            hosted_context_comparison=_context_comparison(
                run_id="hosted-context",
                model_name="sonnet",
                normal_value=0.70,
                semantic_value=0.72,
            ),
            normal_local_vs_hosted_comparison=_hosted_comparison(
                run_id="normal-gap",
                local_value=0.50,
                hosted_value=0.70,
            ),
            semantic_local_vs_hosted_comparison=_hosted_comparison(
                run_id="semantic-gap",
                local_value=0.65,
                hosted_value=0.72,
            ),
        )


def test_summarize_semantic_context_transfer_files_writes_input_artifact_hashes(
    tmp_path: Path,
) -> None:
    local_context = tmp_path / "local.context.json"
    hosted_context = tmp_path / "hosted.context.json"
    normal_gap = tmp_path / "normal.gap.json"
    semantic_gap = tmp_path / "semantic.gap.json"
    output = tmp_path / "summary.json"
    local_context.write_text(
        json.dumps(
            _context_comparison(
                run_id="local-context",
                model_name="local-lora",
                normal_value=0.50,
                semantic_value=0.65,
            )
        )
    )
    hosted_context.write_text(
        json.dumps(
            _context_comparison(
                run_id="hosted-context",
                model_name="sonnet",
                normal_value=0.70,
                semantic_value=0.72,
            )
        )
    )
    normal_gap.write_text(
        json.dumps(
            _hosted_comparison(run_id="normal-gap", local_value=0.50, hosted_value=0.70)
        )
    )
    semantic_gap.write_text(
        json.dumps(
            _hosted_comparison(run_id="semantic-gap", local_value=0.65, hosted_value=0.72)
        )
    )

    summary = summarize_semantic_context_transfer_evidence_files(
        local_context_comparison_path=local_context,
        hosted_context_comparison_path=hosted_context,
        normal_local_vs_hosted_comparison_path=normal_gap,
        semantic_local_vs_hosted_comparison_path=semantic_gap,
        output_path=output,
    )

    assert json.loads(output.read_text()) == summary
    assert summary["input_artifacts"]["local_context_comparison_path"] == str(local_context)
    assert summary["input_artifacts"]["semantic_local_vs_hosted_comparison_path"] == str(
        semantic_gap
    )
