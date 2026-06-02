from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.structured_brief_training_rows import (
    NON_ORACLE_GENERATION_POLICY,
    STRUCTURED_BRIEF_PROMPT_POLICY,
    STRUCTURED_BRIEF_SUPERVISION_POLICY,
    build_structured_brief_training_records,
    structured_brief_record_from_turn,
    write_structured_brief_training_dataset,
)


def _turn(*, split_role: str = "train") -> dict:
    return {
        "id": "dialog-a:1",
        "dialog_id": "dialog-a",
        "turn_index": 1,
        "turn_count": 2,
        "database_id": "store",
        "source": "cosql_train_v1",
        "split_id": "cosql_train_v1",
        "split_role": split_role,
        "split_row_id": "cosql_train:0000:abc",
        "history_policy": "gold_sql_teacher_forced",
        "messages": [
            {"role": "system", "content": "Return only SQL."},
            {
                "role": "user",
                "content": (
                    "Schema/context:\n"
                    "customers(id int, country_code text)\n"
                    "orders(id int, customer_id int, amount real)\n\n"
                    "Question:\nShow revenue by country."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "SELECT customers.country_code, SUM(orders.amount) FROM orders "
                    "JOIN customers ON orders.customer_id = customers.id "
                    "GROUP BY customers.country_code;"
                ),
            },
            {"role": "user", "content": "Question:\nOnly France."},
        ],
        "reference_sql": (
            "SELECT SUM(orders.amount) FROM orders "
            "JOIN customers ON orders.customer_id = customers.id "
            "WHERE customers.country_code = 'FR';"
        ),
        "gold_plan": {
            "relevant_tables": ["orders", "customers"],
            "relevant_columns": ["orders.amount", "customers.country_code"],
            "query_skeleton": {"select": True, "join": True, "where": True},
            "projection_shape": {
                "selected_count": 1,
                "selected_expressions": ["SUM(orders.amount)"],
                "aggregations": ["sum"],
                "preserve_duplicates": True,
            },
        },
    }


def test_structured_brief_record_uses_train_answer_key_as_target_not_prompt() -> None:
    row = structured_brief_record_from_turn(_turn())

    prompt_messages = row["messages"][:-1]
    prompt_text = json.dumps(prompt_messages).lower()
    target = row["messages"][-1]["content"]

    assert "where customers.country_code = 'fr'" not in prompt_text
    assert "gold_plan" not in prompt_text
    assert "expected_rows" not in prompt_text
    assert len([message for message in prompt_messages if message["role"] == "system"]) == 1
    assert "return only sql" not in prompt_messages[0]["content"].lower()
    assert "QUERY_BRIEF:" in target
    assert "intent: Only France." in target
    assert "entities_and_values: 'FR'" in target
    assert "metrics_or_measures: sum" in target
    assert "joins_or_table_families: customers, orders" in target
    assert "SQL:" in target
    assert "WHERE customers.country_code = 'FR'" in target
    assert row["oracle_policy"] == NON_ORACLE_GENERATION_POLICY
    assert row["supervision_policy"] == STRUCTURED_BRIEF_SUPERVISION_POLICY
    assert row["structured_brief_prompt_policy"] == STRUCTURED_BRIEF_PROMPT_POLICY
    assert row["reference_sql_visible_to_model_prompt"] is False
    assert row["structured_brief_visible_as_assistant_label"] is True


def test_structured_brief_record_rejects_non_train_split() -> None:
    with pytest.raises(ValueError, match="split_role=train"):
        structured_brief_record_from_turn(_turn(split_role="clean_local_holdout"))


def test_structured_brief_record_allows_previous_assistant_sql_history() -> None:
    turn = _turn()
    turn["reference_sql"] = "SELECT max ( capacity )  FROM classroom"
    turn["messages"][2]["content"] = (
        "SELECT max ( capacity )  FROM classroom Where building  =  \"Lamberton\""
    )

    row = structured_brief_record_from_turn(turn)

    assert row["messages"][-1]["content"].endswith("SELECT max ( capacity )  FROM classroom")


def test_structured_brief_record_rejects_reference_sql_in_user_prompt() -> None:
    turn = _turn()
    turn["messages"][-1]["content"] += f"\n{turn['reference_sql']}"

    with pytest.raises(ValueError, match="current reference SQL leaked"):
        structured_brief_record_from_turn(turn)


def test_write_structured_brief_training_dataset_writes_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "structured_brief_training_rows.jsonl"
    manifest_path = tmp_path / "structured_brief_training_rows.manifest.json"
    input_path.write_text(json.dumps(_turn()) + "\n", encoding="utf-8")

    manifest = write_structured_brief_training_dataset(
        input_path=input_path,
        output_path=output_path,
        manifest_output_path=manifest_path,
        command=[
            "uv",
            "run",
            "--active",
            "--no-sync",
            "python",
            "-m",
            "data.structured_brief_training_rows",
        ],
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest == written_manifest
    assert manifest["artifact_type"] == "structured_brief_training_dataset"
    assert manifest["row_count"] == 1
    assert manifest["split_roles"] == {"train": 1}
    assert manifest["oracle_policy"] == NON_ORACLE_GENERATION_POLICY
    assert manifest["supervision_policy"] == STRUCTURED_BRIEF_SUPERVISION_POLICY
    assert manifest["reference_sql_visible_to_model_prompt"] is False
    assert manifest["structured_brief_visible_as_assistant_label"] is True
    assert manifest["output_sha256"]
    assert rows[0]["training_target"] == "structured_brief_sql"


def test_build_structured_brief_training_records_rejects_expanded_scorer_hints(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "expanded.jsonl"
    row = _turn()
    row["messages"][-1]["content"] += "\n\nOracle SQL planning hints:\n- use orders.amount"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="scorer-derived planning hints"):
        build_structured_brief_training_records(input_path)
