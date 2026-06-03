from __future__ import annotations

import json

from eval.rollout_eval import (
    MODEL_GENERATED_SQL_ROLLOUT,
    evaluate_rollout_records,
    load_rollout_prepared_records,
)


def _dialog_record() -> dict:
    return {
        "id": "dialog-a",
        "source": "unit",
        "database_id": "db1",
        "history_policy": "gold_sql_teacher_forced",
        "evaluation_mode": "non_oracle_generation",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Question:\nfirst"},
            {"role": "assistant", "content": "SELECT gold_first;"},
            {"role": "user", "content": "Question:\nsecond"},
            {"role": "assistant", "content": "SELECT gold_second;"},
        ],
    }


def test_rollout_uses_generated_sql_instead_of_prior_gold_sql() -> None:
    prompts: list[list[dict[str, str]]] = []
    generations = iter(["SELECT generated_first;", "SELECT generated_second;"])

    def fake_generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        prompts.append(messages)
        return next(generations), 1.0

    rows = evaluate_rollout_records(
        [_dialog_record()],
        generate_fn=fake_generate,
        model_name="local-9b",
        database_root=None,
    )

    assert [row["reference_sql"] for row in rows] == [
        "SELECT gold_first;",
        "SELECT gold_second;",
    ]
    assert [row["generated_sql"] for row in rows] == [
        "SELECT generated_first;",
        "SELECT generated_second;",
    ]
    assert rows[0]["history_policy"] == MODEL_GENERATED_SQL_ROLLOUT
    assert rows[0]["original_history_policy"] == "gold_sql_teacher_forced"
    assert rows[1]["messages"][2] == {
        "role": "assistant",
        "content": "SELECT generated_first;",
    }
    assert "SELECT gold_first" not in json.dumps(rows[1]["messages"])
    assert "SELECT gold_second" not in json.dumps(rows[1]["messages"])
    assert prompts[1][2] == {"role": "assistant", "content": "SELECT generated_first;"}
    assert "SELECT gold_first" not in json.dumps(prompts[1])
    assert "SELECT gold_second" not in json.dumps(prompts[1])


def test_rollout_continues_after_invalid_generated_sql() -> None:
    prompts: list[list[dict[str, str]]] = []
    generations = iter(["not sql at all", "SELECT generated_second;"])

    def fake_generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        prompts.append(messages)
        return next(generations), 1.0

    rows = evaluate_rollout_records(
        [_dialog_record()],
        generate_fn=fake_generate,
        model_name="local-9b",
        database_root=None,
    )

    assert rows[0]["syntax_valid"] is False
    assert rows[1]["messages"][2] == {"role": "assistant", "content": "not sql at all"}
    assert prompts[1][2] == {"role": "assistant", "content": "not sql at all"}


def test_rollout_treats_seeded_failure_turn_as_history_only() -> None:
    record = _dialog_record()
    record["history_policy"] = "seeded_generated_failure_then_rollout"
    record["seeded_failure_turn_index"] = 0
    prompts: list[list[dict[str, str]]] = []

    def fake_generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        prompts.append(messages)
        return "SELECT repaired;", 1.0

    rows = evaluate_rollout_records(
        [record],
        generate_fn=fake_generate,
        model_name="local-9b",
        database_root=None,
    )

    assert len(rows) == 1
    assert rows[0]["id"] == "dialog-a:1"
    assert rows[0]["turn_index"] == 1
    assert rows[0]["reference_sql"] == "SELECT gold_second;"
    assert rows[0]["generated_sql"] == "SELECT repaired;"
    assert rows[0]["messages"][2] == {
        "role": "assistant",
        "content": "SELECT gold_first;",
    }
    assert prompts[0][2] == {"role": "assistant", "content": "SELECT gold_first;"}


def test_load_rollout_prepared_records_loads_dialogs(tmp_path) -> None:
    path = tmp_path / "prepared.jsonl"
    record = _dialog_record()
    path.write_text(json.dumps(record) + "\n")

    assert load_rollout_prepared_records(path)[0]["id"] == "dialog-a"
