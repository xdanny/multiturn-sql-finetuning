from __future__ import annotations

from pathlib import Path

import pytest

from eval.local_benchmark_pair import (
    LocalBenchmarkPairSpec,
    run_local_benchmark_pair,
)


def test_run_local_benchmark_pair_runs_both_local_benchmarks(tmp_path: Path) -> None:
    calls: list[dict] = []

    def fake_run_local_benchmark(**kwargs) -> int:
        calls.append(kwargs)
        kwargs["output"].write_text("{}\n")
        kwargs["manifest_output"].write_text("{}\n")
        return 0

    outputs = run_local_benchmark_pair(
        spec=LocalBenchmarkPairSpec(
            method_output_stem="predicted_planner",
            method_prompt_variant="predicted_planner",
        ),
        output_dir=tmp_path / "results",
        run_id="planner-local",
        model_name="local-9b",
        method_input_path=tmp_path / "predicted.jsonl",
        direct_input_path=tmp_path / "direct.jsonl",
        method_adapter_path=tmp_path / "predicted_adapter",
        direct_adapter_path=tmp_path / "direct_adapter",
        database_root=tmp_path / "db",
        max_new_tokens=64,
        max_memory_gb=24,
        method_command=["python", "-m", "eval.run_local_predicted_planner_comparison"],
        direct_command=["python", "-m", "eval.run_local_predicted_planner_comparison"],
        benchmark_runner=fake_run_local_benchmark,
    )

    assert [call["input_path"] for call in calls] == [
        tmp_path / "predicted.jsonl",
        tmp_path / "direct.jsonl",
    ]
    assert calls[0]["adapter_path"] == tmp_path / "predicted_adapter"
    assert calls[1]["adapter_path"] == tmp_path / "direct_adapter"
    assert calls[0]["prompt_variant"] == "predicted_planner"
    assert calls[1]["prompt_variant"] == "direct_sql_control"
    assert outputs["method_output"] == tmp_path / "results" / "planner-local.predicted_planner.jsonl"
    assert outputs["method_manifest_output"] == tmp_path / "results" / "planner-local.predicted_planner.manifest.json"
    assert outputs["direct_output"] == tmp_path / "results" / "planner-local.direct.jsonl"
    assert outputs["direct_manifest_output"] == tmp_path / "results" / "planner-local.direct.manifest.json"
    assert outputs["compared_output"] == tmp_path / "results" / "planner-local.compared.manifest.json"


def test_run_local_benchmark_pair_raises_on_benchmark_failure(tmp_path: Path) -> None:
    def fail_run_local_benchmark(**kwargs) -> int:
        return 9

    with pytest.raises(RuntimeError, match="benchmark failed"):
        run_local_benchmark_pair(
            spec=LocalBenchmarkPairSpec(
                method_output_stem="semantic_proxy",
                method_prompt_variant="semantic_proxy",
            ),
            output_dir=tmp_path / "results",
            run_id="semantic-proxy-local",
            model_name="local-9b",
            method_input_path=tmp_path / "semantic.jsonl",
            direct_input_path=tmp_path / "direct.jsonl",
            method_adapter_path=None,
            direct_adapter_path=None,
            database_root=None,
            max_new_tokens=64,
            max_memory_gb=24,
            method_command=["python", "-m", "eval.run_local_semantic_proxy_comparison"],
            direct_command=["python", "-m", "eval.run_local_semantic_proxy_comparison"],
            benchmark_runner=fail_run_local_benchmark,
        )
