from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.planner_sft import (
    PLANNER_SFT_LABEL_SOURCE,
    PLANNER_SFT_ORACLE_POLICY,
    PLANNER_SFT_PROMPT_POLICY,
    build_planner_sft_records,
    planner_sft_record_from_turn,
    write_planner_sft_dataset,
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
            {"role": "system", "content": "You are a SQL expert."},
            {
                "role": "user",
                "content": (
                    "Schema/context:\n"
                    "customers(id int, name text)\n"
                    "orders(id int, customer_id int, amount real)\n\n"
                    "Question:\nShow customer names."
                ),
            },
            {"role": "assistant", "content": "SELECT name FROM customers;"},
            {"role": "user", "content": "Question:\nNow show order totals."},
        ],
        "reference_sql": "SELECT SUM(amount) FROM orders;",
        "gold_plan": {
            "relevant_tables": ["orders"],
            "relevant_columns": ["orders.amount"],
            "query_skeleton": {"select": True},
            "projection_shape": {
                "selected_expressions": ["sum"],
                "selected_count": 1,
                "aggregations": ["sum"],
                "preserve_duplicates": True,
            },
        },
    }


def test_planner_sft_record_uses_train_gold_plan_as_target_not_prompt() -> None:
    row = planner_sft_record_from_turn(_turn())

    prompt_messages = row["messages"][:-1]
    prompt_text = json.dumps(prompt_messages).lower()
    target = json.loads(row["messages"][-1]["content"])

    assert all(message["role"] != "assistant" for message in prompt_messages)
    assert "select name from customers" not in prompt_text
    assert "select sum(amount)" not in prompt_text
    assert "gold_plan" not in prompt_text
    assert target["relevant_tables"] == ["orders"]
    assert target["projection_shape"]["aggregations"] == ["sum"]
    assert row["planner_label_source"] == PLANNER_SFT_LABEL_SOURCE
    assert row["oracle_policy"] == PLANNER_SFT_ORACLE_POLICY
    assert row["planner_prompt_policy"] == PLANNER_SFT_PROMPT_POLICY
    assert row["reference_sql_visible_to_model"] is False
    assert row["gold_plan_visible_to_model_prompt"] is False
    assert row["target_plan_visible_as_assistant_label"] is True


def test_planner_sft_record_preserves_target_projection_order() -> None:
    turn = _turn()
    turn["gold_plan"]["projection_shape"]["selected_expressions"] = [
        "orders.total",
        "orders.id",
    ]
    turn["gold_plan"]["projection_shape"]["selected_count"] = 2

    row = planner_sft_record_from_turn(turn)
    target = json.loads(row["messages"][-1]["content"])

    assert target["projection_shape"]["selected_expressions"] == [
        "orders.total",
        "orders.id",
    ]


def test_planner_sft_record_rejects_non_train_split() -> None:
    with pytest.raises(ValueError, match="split_role=train"):
        planner_sft_record_from_turn(_turn(split_role="clean_local_holdout"))


def _prepared_dialog() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": (
                    "Schema/context:\n"
                    "customers(id int, name text)\n\n"
                    "Question:\nShow customer names."
                ),
            },
            {"role": "assistant", "content": "SELECT name FROM customers;"},
        ],
        "source": "cosql_train_v1",
        "database_id": "store",
        "split_id": "cosql_train_v1",
        "split_role": "train",
        "split_row_id": "cosql_train:0000:abc",
        "evaluation_mode": "non_oracle_generation",
        "history_policy": "single_turn",
        "uses_oracle_planning_hints": False,
        "semantic_context_pruned_by_oracle_labels": False,
        "gold_plans": [
            {
                "relevant_tables": ["customers"],
                "relevant_columns": ["customers.name"],
                "query_skeleton": {"select": True},
                "projection_shape": {"selected_count": 1, "preserve_duplicates": True},
            }
        ],
        "schema_link_labels": [
            {
                "relevant_tables": ["customers"],
                "relevant_columns": ["customers.name"],
                "query_skeleton": {"select": True},
                "projection_shape": {"selected_count": 1, "preserve_duplicates": True},
            }
        ],
    }


def test_write_planner_sft_dataset_writes_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    output_path = tmp_path / "planner_sft.jsonl"
    manifest_path = tmp_path / "planner_sft.manifest.json"
    input_path.write_text(json.dumps(_prepared_dialog()) + "\n", encoding="utf-8")

    manifest = write_planner_sft_dataset(
        input_path=input_path,
        output_path=output_path,
        manifest_output_path=manifest_path,
        command=["uv", "run", "--active", "--no-sync", "python", "-m", "data.planner_sft"],
    )

    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    written_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == written_manifest
    assert manifest["artifact_type"] == "planner_sft_dataset"
    assert manifest["row_count"] == 1
    assert manifest["split_roles"] == {"train": 1}
    assert manifest["reference_sql_visible_to_model"] is False
    assert manifest["gold_plan_visible_to_model_prompt"] is False
    assert manifest["target_plan_visible_as_assistant_label"] is True
    assert manifest["planner_prompt_policy"] == PLANNER_SFT_PROMPT_POLICY
    assert manifest["output_sha256"]
    assert rows[0]["split_role"] == "train"
    assert json.loads(rows[0]["messages"][-1]["content"])["relevant_tables"] == ["customers"]


def test_build_planner_sft_records_rejects_non_train_prepared_split(tmp_path: Path) -> None:
    input_path = tmp_path / "prepared.jsonl"
    row = _prepared_dialog()
    row["split_role"] = "clean_local_holdout"
    input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="split_role=train"):
        build_planner_sft_records(input_path)
