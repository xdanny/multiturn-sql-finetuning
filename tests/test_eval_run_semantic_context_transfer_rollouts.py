from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.run_semantic_context_transfer_rollouts import (
    _api_key_from_env,
    run_semantic_context_transfer_rollouts,
)


def test_api_key_from_env_prefers_named_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    value, source = _api_key_from_env("OPENROUTER_API_KEY")

    assert value == "openrouter-key"
    assert source == "OPENROUTER_API_KEY"


def test_api_key_from_env_falls_back_to_openai_env(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    value, source = _api_key_from_env("OPENROUTER_API_KEY")

    assert value == "openai-key"
    assert source == "OPENAI_API_KEY"


def test_api_key_from_env_requires_env_secret(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="Set OPENROUTER_API_KEY"):
        _api_key_from_env("OPENROUTER_API_KEY")


def test_run_semantic_context_transfer_rollouts_runs_both_arms_then_compares(
    tmp_path: Path,
    monkeypatch,
) -> None:
    normal_input = tmp_path / "normal.jsonl"
    semantic_input = tmp_path / "semantic.jsonl"
    preflight = tmp_path / "preflight.json"
    normal_input.write_text("{}\n")
    semantic_input.write_text("{}\n")
    preflight.write_text(json.dumps({"artifact_type": "semantic_context_transfer_preflight"}))
    output_dir = tmp_path / "results"
    run_calls: list[dict] = []

    def fake_run_rollout_eval(**kwargs):
        run_calls.append(kwargs)
        kwargs["manifest_output"].write_text(
            json.dumps(
                {
                    "run_id": kwargs["output"].stem,
                    "input_sha256": kwargs["input_path"].name,
                    "output_path": str(kwargs["output"]),
                    "row_count": 1,
                    "metrics": {
                        "value_execution_accuracy": 0.5,
                        "strict_execution_accuracy": 0.5,
                        "syntax_accuracy": 1.0,
                        "history_policy": "model_generated_sql_rollout",
                    },
                }
            )
            + "\n"
        )
        return 0

    compare_calls: list[dict] = []

    def fake_compare(**kwargs):
        compare_calls.append(kwargs)
        kwargs["output_path"].write_text(json.dumps({"compared": True}) + "\n")
        return {
            "metrics": {
                "semantic_context_value_delta_vs_normal": 0.25,
            }
        }

    monkeypatch.setattr(
        "eval.run_semantic_context_transfer_rollouts.run_rollout_eval",
        fake_run_rollout_eval,
    )
    monkeypatch.setattr(
        "eval.run_semantic_context_transfer_rollouts.compare_semantic_context_transfer_manifest_files",
        fake_compare,
    )

    compared = run_semantic_context_transfer_rollouts(
        normal_input=normal_input,
        semantic_input=semantic_input,
        preflight_manifest=preflight,
        output_dir=output_dir,
        run_id="cp10",
        model_name="anthropic/claude-sonnet-4.6",
        endpoint="https://openrouter.ai/api/v1",
        api_key="secret-key",
        api_key_source="OPENROUTER_API_KEY",
        database_root=tmp_path / "db",
        command=["run-cp10"],
    )

    assert compared["metrics"]["semantic_context_value_delta_vs_normal"] == 0.25
    assert [call["input_path"] for call in run_calls] == [normal_input, semantic_input]
    assert [call["prompt_variant"] for call in run_calls] == [
        "normal_schema_context",
        "schema_context_plus_database_value_retrieval",
    ]
    assert all(call["api_key"] == "secret-key" for call in run_calls)
    assert all("secret-key" not in " ".join(call["command"]) for call in run_calls)
    assert all("# secret-policy" in call["command"] for call in run_calls)
    assert compare_calls == [
        {
            "normal_manifest_path": output_dir / "cp10.normal.rollout.manifest.json",
            "semantic_manifest_path": output_dir / "cp10.semantic.rollout.manifest.json",
            "preflight_manifest_path": preflight,
            "output_path": output_dir / "cp10.semantic_context_comparison.manifest.json",
            "comparison_role": "hosted_sonnet",
            "repo_root": Path("."),
        }
    ]


def test_run_semantic_context_transfer_rollouts_stops_after_failed_normal_arm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_calls = []

    def fake_run_rollout_eval(**kwargs):
        run_calls.append(kwargs)
        return 1

    monkeypatch.setattr(
        "eval.run_semantic_context_transfer_rollouts.run_rollout_eval",
        fake_run_rollout_eval,
    )

    with pytest.raises(RuntimeError, match="normal-context rollout failed"):
        run_semantic_context_transfer_rollouts(
            normal_input=tmp_path / "normal.jsonl",
            semantic_input=tmp_path / "semantic.jsonl",
            preflight_manifest=tmp_path / "preflight.json",
            output_dir=tmp_path / "results",
            run_id="cp10",
            model_name="model",
            endpoint="endpoint",
            api_key="secret-key",
            api_key_source="OPENROUTER_API_KEY",
            database_root=None,
            command=["run-cp10"],
        )

    assert len(run_calls) == 1
