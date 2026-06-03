from __future__ import annotations

import csv
import json

from eval.prompt_optimize import (
    PromptVariant,
    apply_prompt_variant,
    load_prompt_variants,
    summarize_results,
    write_summary,
)


def test_apply_prompt_variant_appends_policy_to_system_message() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question"},
    ]

    updated = apply_prompt_variant(messages, PromptVariant("join", "Prefer listed joins."))

    assert updated[0]["content"].startswith("sys")
    assert "Additional prompt policy (join)" in updated[0]["content"]
    assert messages[0]["content"] == "sys"


def test_apply_prompt_variant_adds_system_message_when_missing() -> None:
    updated = apply_prompt_variant(
        [{"role": "user", "content": "question"}],
        PromptVariant("minimal", "Return SQL only."),
    )

    assert updated[0]["role"] == "system"
    assert "Return SQL only" in updated[0]["content"]


def test_load_prompt_variants_from_json(tmp_path) -> None:
    path = tmp_path / "variants.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "grain",
                    "instruction": "Check grain before aggregating.",
                    "source": "unit",
                }
            ]
        )
    )

    variants = load_prompt_variants(path)

    assert variants == [
        PromptVariant("grain", "Check grain before aggregating.", "unit"),
    ]


def test_summarize_results_reports_accuracy_and_latency() -> None:
    summary = summarize_results(
        [
            {"execution_score": 1.0, "syntax_valid": True, "generation_latency_ms": 10.0},
            {"execution_score": 0.0, "syntax_valid": False, "generation_latency_ms": 20.0},
        ]
    )

    assert summary == {
        "accuracy": 0.5,
        "syntax_accuracy": 0.5,
        "mean_latency_ms": 15.0,
        "samples": 2,
    }


def test_write_summary_writes_csv(tmp_path) -> None:
    output = tmp_path / "summary.csv"
    write_summary(
        [
            {
                "prompt_variant": "baseline",
                "prompt_variant_source": "static",
                "accuracy": 0.5,
                "syntax_accuracy": 1.0,
                "mean_latency_ms": 12.0,
                "samples": 2,
            }
        ],
        output,
    )

    rows = list(csv.DictReader(output.open()))

    assert rows[0]["prompt_variant"] == "baseline"
    assert rows[0]["accuracy"] == "0.5"
