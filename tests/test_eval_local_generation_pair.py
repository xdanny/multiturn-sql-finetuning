from __future__ import annotations

from pathlib import Path

import pytest

from eval.local_generation_pair import (
    LocalGenerationPairSpec,
    run_local_generation_pair,
)


def test_run_local_generation_pair_runs_both_generators_and_evaluators(tmp_path: Path) -> None:
    calls: list[tuple[str, Path, Path | None]] = []

    def method_generate(predictions_path: Path) -> int:
        calls.append(("method_generate", predictions_path, None))
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text("{}\n")
        return 0

    def direct_generate(predictions_path: Path) -> int:
        calls.append(("direct_generate", predictions_path, None))
        predictions_path.write_text("{}\n")
        return 0

    def method_eval(predictions_path: Path, results_path: Path, manifest_path: Path) -> int:
        calls.append(("method_eval", predictions_path, manifest_path))
        results_path.write_text("{}\n")
        manifest_path.write_text("{}\n")
        return 0

    def direct_eval(predictions_path: Path, results_path: Path, manifest_path: Path) -> int:
        calls.append(("direct_eval", predictions_path, manifest_path))
        results_path.write_text("{}\n")
        manifest_path.write_text("{}\n")
        return 0

    outputs = run_local_generation_pair(
        spec=LocalGenerationPairSpec(method_output_stem="metric_dsl"),
        output_dir=tmp_path / "results",
        run_id="metric-local",
        method_generate=method_generate,
        direct_generate=direct_generate,
        method_eval=method_eval,
        direct_eval=direct_eval,
    )

    assert outputs["method_predictions"] == tmp_path / "results" / "metric-local.metric_dsl.predictions.jsonl"
    assert outputs["method_manifest_output"] == tmp_path / "results" / "metric-local.metric_dsl.manifest.json"
    assert outputs["direct_predictions"] == tmp_path / "results" / "metric-local.direct_sql.predictions.jsonl"
    assert outputs["direct_manifest_output"] == tmp_path / "results" / "metric-local.direct_sql.manifest.json"
    assert outputs["compared_output"] == tmp_path / "results" / "metric-local.compared.manifest.json"
    assert [name for name, _, _ in calls] == [
        "method_generate",
        "direct_generate",
        "method_eval",
        "direct_eval",
    ]


def test_run_local_generation_pair_raises_on_generator_failure(tmp_path: Path) -> None:
    def fail_generate(predictions_path: Path) -> int:
        return 7

    with pytest.raises(RuntimeError, match="generation failed"):
        run_local_generation_pair(
            spec=LocalGenerationPairSpec(method_output_stem="metric_dsl"),
            output_dir=tmp_path / "results",
            run_id="metric-local",
            method_generate=fail_generate,
            direct_generate=lambda _: 0,
            method_eval=lambda *_: 0,
            direct_eval=lambda *_: 0,
        )
