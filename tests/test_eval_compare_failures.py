from __future__ import annotations

import csv
import json

from eval.compare_failures import (
    load_classified_files,
    write_model_summary,
    write_pairwise_comparison,
)


def _row(
    *,
    row_id: str,
    model_name: str,
    score: float,
    error_primary: str,
    error_secondary: list[str] | None = None,
    prompt_variant: str = "",
) -> dict:
    return {
        "id": row_id,
        "dialog_id": "dialog",
        "model_name": model_name,
        "prompt_variant": prompt_variant,
        "value_execution_score": score,
        "strict_execution_score": score,
        "syntax_valid": True,
        "error_primary": error_primary,
        "error_secondary": error_secondary or [],
    }


def test_load_classified_files_uses_model_and_prompt_variant_label(tmp_path) -> None:
    path = tmp_path / "candidate.jsonl"
    path.write_text(
        json.dumps(
            _row(
                row_id="1",
                model_name="model",
                prompt_variant="variant",
                score=1.0,
                error_primary="correct",
            )
        )
        + "\n"
    )

    runs = load_classified_files([path])

    assert list(runs) == ["model[variant]"]


def test_write_model_summary_counts_error_categories(tmp_path) -> None:
    output = tmp_path / "summary.csv"
    runs = {
        "model": [
            _row(row_id="1", model_name="model", score=1.0, error_primary="correct"),
            _row(row_id="2", model_name="model", score=0.0, error_primary="schema_link"),
        ]
    }

    write_model_summary(runs, output)
    rows = list(csv.DictReader(output.open()))

    assert rows[0]["run"] == "model"
    assert rows[0]["value_accuracy"] == "0.5"
    assert rows[0]["correct"] == "1"
    assert rows[0]["schema_link"] == "1"


def test_write_model_summary_counts_secondary_history_resolution(tmp_path) -> None:
    output = tmp_path / "summary.csv"
    runs = {
        "model": [
            _row(
                row_id="1",
                model_name="model",
                score=0.0,
                error_primary="schema_link",
                error_secondary=["history_resolution"],
            )
        ]
    }

    write_model_summary(runs, output)
    rows = list(csv.DictReader(output.open()))

    assert rows[0]["history_resolution"] == "0"
    assert rows[0]["history_resolution_turns"] == "1"


def test_write_pairwise_comparison_reports_fixed_and_regressed_turns(tmp_path) -> None:
    output = tmp_path / "pairwise.csv"
    runs = {
        "base": [
            _row(row_id="1", model_name="base", score=0.0, error_primary="schema_link"),
            _row(row_id="2", model_name="base", score=1.0, error_primary="correct"),
        ],
        "candidate": [
            _row(row_id="1", model_name="candidate", score=1.0, error_primary="correct"),
            _row(row_id="2", model_name="candidate", score=0.0, error_primary="aggregation"),
        ],
    }

    write_pairwise_comparison(runs, baseline_label="base", output=output)
    row = list(csv.DictReader(output.open()))[0]

    assert row["candidate"] == "candidate"
    assert row["fixed_turns"] == "1"
    assert row["regressed_turns"] == "1"
    assert row["fixed_error_primary"] == '{"schema_link": 1}'
    assert row["regressed_error_primary"] == '{"aggregation": 1}'
