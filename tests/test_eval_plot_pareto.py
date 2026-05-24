from __future__ import annotations

import json

from eval.plot_pareto import (
    load_results,
    resolve_result_files,
    summarize_dialog_results,
    summarize_results,
)


def test_summarize_results_groups_model_and_source(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    rows = [
        {
            "model_name": "base",
            "source": "sparc",
            "execution_score": 1.0,
            "syntax_valid": True,
            "generation_latency_ms": 10.0,
        },
        {
            "model_name": "base",
            "source": "sparc",
            "execution_score": 0.0,
            "syntax_valid": True,
            "generation_latency_ms": 20.0,
        },
    ]
    (results_dir / "base.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = summarize_results(load_results(results_dir))

    assert summary.loc[0, "accuracy"] == 0.5
    assert summary.loc[0, "strict_accuracy"] == 0.5
    assert summary.loc[0, "value_accuracy"] == 0.5
    assert summary.loc[0, "syntax_accuracy"] == 1.0
    assert summary.loc[0, "mean_latency_ms"] == 15.0
    assert summary.loc[0, "samples"] == 2
    assert summary.loc[0, "result_files"] == "base.jsonl"


def test_load_results_filters_by_include_glob(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    row = {
        "model_name": "base",
        "source": "cosql",
        "execution_score": 1.0,
        "syntax_valid": True,
        "generation_latency_ms": 10.0,
    }
    stale = row | {"model_name": "stale"}
    (results_dir / "base_dev_3.jsonl").write_text(json.dumps(row) + "\n")
    (results_dir / "base_smoke.jsonl").write_text(json.dumps(stale) + "\n")

    paths = resolve_result_files(results_dir, ["*_dev_3.jsonl"])
    results = load_results(results_dir, ["*_dev_3.jsonl"])

    assert [path.name for path in paths] == ["base_dev_3.jsonl"]
    assert results["model_name"].tolist() == ["base"]


def test_summarize_dialog_results_reports_interaction_match_rate(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    rows = [
        {
            "model_name": "base",
            "source": "cosql",
            "dialog_id": "d1",
            "execution_score": 1.0,
            "syntax_valid": True,
            "generation_latency_ms": 10.0,
        },
        {
            "model_name": "base",
            "source": "cosql",
            "dialog_id": "d1",
            "execution_score": 1.0,
            "syntax_valid": True,
            "generation_latency_ms": 12.0,
        },
        {
            "model_name": "base",
            "source": "cosql",
            "dialog_id": "d2",
            "execution_score": 0.0,
            "syntax_valid": True,
            "generation_latency_ms": 20.0,
        },
    ]
    (results_dir / "base.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = summarize_dialog_results(load_results(results_dir))

    assert summary.loc[0, "dialog_execution_accuracy"] == 0.5
    assert summary.loc[0, "dialog_syntax_accuracy"] == 1.0
    assert summary.loc[0, "interaction_match_rate"] == 0.5
    assert summary.loc[0, "dialogs"] == 2
    assert summary.loc[0, "turns"] == 3


def test_summarize_results_separates_prompt_variants(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    rows = [
        {
            "model_name": "semantic",
            "prompt_variant": "baseline",
            "source": "cosql",
            "execution_score": 0.0,
            "syntax_valid": True,
            "generation_latency_ms": 10.0,
        },
        {
            "model_name": "semantic",
            "prompt_variant": "grounded",
            "source": "cosql",
            "execution_score": 1.0,
            "syntax_valid": True,
            "generation_latency_ms": 20.0,
        },
    ]
    (results_dir / "variants.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = summarize_results(load_results(results_dir))

    assert sorted(summary["prompt_variant"].tolist()) == ["baseline", "grounded"]
    assert summary["accuracy"].tolist() == [0.0, 1.0]


def test_summarize_results_reports_strict_and_value_accuracy_when_present(tmp_path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    rows = [
        {
            "model_name": "semantic",
            "source": "cosql",
            "execution_score": 1.0,
            "strict_execution_score": 0.0,
            "value_execution_score": 1.0,
            "syntax_valid": True,
            "generation_latency_ms": 10.0,
        },
        {
            "model_name": "semantic",
            "source": "cosql",
            "execution_score": 0.0,
            "strict_execution_score": 0.0,
            "value_execution_score": 0.0,
            "syntax_valid": True,
            "generation_latency_ms": 20.0,
        },
    ]
    (results_dir / "semantic.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = summarize_results(load_results(results_dir))

    assert summary.loc[0, "accuracy"] == 0.5
    assert summary.loc[0, "strict_accuracy"] == 0.0
    assert summary.loc[0, "value_accuracy"] == 0.5
