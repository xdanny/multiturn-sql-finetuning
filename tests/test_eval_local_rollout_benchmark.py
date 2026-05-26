from __future__ import annotations

import json
from types import SimpleNamespace

from eval.local_rollout_benchmark import run_local_rollout_benchmark


def _dialog_record() -> dict:
    return {
        "id": "dialog-a",
        "source": "unit",
        "database_id": "db1",
        "history_policy": "gold_sql_teacher_forced",
        "evaluation_mode": "non_oracle_generation",
        "gold_plans": [{}, {}],
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Question:\nfirst"},
            {"role": "assistant", "content": "SELECT gold_first;"},
            {"role": "user", "content": "Question:\nsecond"},
            {"role": "assistant", "content": "SELECT gold_second;"},
        ],
    }


def test_run_local_rollout_benchmark_writes_manifest_and_generated_history(
    tmp_path, monkeypatch
) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "rollout.jsonl"
    manifest_path = tmp_path / "rollout.manifest.json"
    input_path.write_text(json.dumps(_dialog_record()) + "\n")
    prompts: list[list[dict[str, str]]] = []
    generations = iter(["SELECT generated_first;", "SELECT generated_second;"])

    monkeypatch.setattr(
        "eval.local_rollout_benchmark.load_model_and_tokenizer",
        lambda **kwargs: (object(), object()),
    )

    def fake_generate(model, tokenizer, *, messages, max_new_tokens):
        prompts.append(messages)
        return next(generations), 1.0

    monkeypatch.setattr("eval.local_rollout_benchmark.generate_local_sql", fake_generate)
    monkeypatch.setattr(
        "eval.rollout_eval.score_single_turn",
        lambda reference_sql, generated_sql, database_path=None: SimpleNamespace(
            execution_score=1.0,
            strict_execution_score=1.0,
            value_execution_score=1.0,
            order_sensitive=False,
            normalized_match=True,
            syntax_valid=True,
            error=None,
        ),
    )

    exit_code = run_local_rollout_benchmark(
        model_name="local-9b",
        adapter_path=None,
        input_path=input_path,
        output_path=output_path,
        manifest_output=manifest_path,
        database_root=None,
        max_new_tokens=64,
        max_memory_gb=None,
        allow_oracle_plan=False,
        prompt_variant="behavior_recovery_rollout",
        command=["python", "-m", "eval.local_rollout_benchmark"],
    )

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())

    assert exit_code == 0
    assert len(rows) == 2
    assert rows[0]["generated_sql"] == "SELECT generated_first;"
    assert rows[1]["messages"][2] == {
        "role": "assistant",
        "content": "SELECT generated_first;",
    }
    assert "SELECT gold_first" not in json.dumps(rows[1]["messages"])
    assert prompts[1][2]["content"] == "SELECT generated_first;"
    assert manifest["benchmark"] == "prepared_rollout"
    assert manifest["evaluation_mode"] == "non_oracle_generation"
    assert manifest["endpoint"] == "local"
    assert manifest["prompt_variant"] == "behavior_recovery_rollout"
    assert manifest["metrics"]["history_policy"] == "model_generated_sql_rollout"
