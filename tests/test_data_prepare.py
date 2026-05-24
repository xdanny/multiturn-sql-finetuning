from __future__ import annotations

import json

import pytest

from data.prepare import (
    DatasetSpec,
    FormatterError,
    build_dataset_manifest,
    format_bird,
    format_cosql,
    format_gretelai,
    format_sparc,
    iter_formatted_records,
    load_source_dataset,
    parse_dataset_specs,
    schema_from_spider_table,
    semantic_model_from_spider_table,
    write_dataset_manifest,
    write_jsonl,
)


def test_format_gretelai_includes_schema_and_sql() -> None:
    messages = format_gretelai(
        {
            "sql_prompt": "List customers.",
            "sql_context": "CREATE TABLE customers (id INT);",
            "sql": "SELECT id FROM customers;",
        }
    )

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert "CREATE TABLE customers" in messages[1]["content"]
    assert messages[2]["content"] == "SELECT id FROM customers;"


def test_format_sparc_uses_database_id() -> None:
    messages = format_sparc(
        {
            "database_id": "hospital_1",
            "question": "Find the largest department.",
            "query": "SELECT name FROM department LIMIT 1;",
        }
    )

    assert "Database: hospital_1" in messages[1]["content"]
    assert messages[2]["content"].startswith("SELECT")


def test_format_bird_uses_evidence() -> None:
    messages = format_bird(
        {
            "db_id": "cards",
            "question": "What is the ratio?",
            "evidence": "ratio = a / b",
            "SQL": "SELECT 1.0;",
        }
    )

    assert "Database: cards" in messages[1]["content"]
    assert "Evidence:" in messages[1]["content"]
    assert messages[2]["content"] == "SELECT 1.0;"


def test_format_single_turn_can_include_semantic_model_context() -> None:
    messages = format_sparc(
        {
            "database_id": "store",
            "question": "How many orders are completed?",
            "query": "SELECT COUNT(*) FROM orders WHERE status = 'completed';",
            "semantic_model_context": "- Cube orders (grain: one row per order)",
        }
    )

    assert "Semantic model:" in messages[1]["content"]
    assert "Cube orders" in messages[1]["content"]


def test_format_single_turn_can_include_sql_planning_hints() -> None:
    messages = format_sparc(
        {
            "database_id": "store",
            "question": "How many orders are completed?",
            "query": "SELECT COUNT(*) FROM orders WHERE status = 'completed';",
            "include_sql_labels": True,
        }
    )

    assert "Oracle SQL planning hints" in messages[1]["content"]
    assert "derived from reference SQL" in messages[1]["content"]
    assert "Relevant tables: orders" in messages[1]["content"]
    assert "Projection shape: 1 selected expression" in messages[1]["content"]


def test_format_cosql_keeps_multi_turn_pairs() -> None:
    messages = format_cosql(
        {
            "database_id": "concert_singer",
            "schema": "CREATE TABLE singer (name TEXT);",
            "interaction": [
                {"utterance": "List singers.", "query": "SELECT name FROM singer;"},
                {"utterance": "Only distinct names.", "query": "SELECT DISTINCT name FROM singer;"},
            ],
        }
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert "CREATE TABLE singer" in messages[1]["content"]
    assert messages[-1]["content"] == "SELECT DISTINCT name FROM singer;"


def test_format_cosql_includes_semantic_model_on_first_turn_only() -> None:
    messages = format_cosql(
        {
            "database_id": "concert_singer",
            "schema": "CREATE TABLE singer (name TEXT);",
            "semantic_model_context": "- Cube singer (grain: one row per singer)",
            "interaction": [
                {"utterance": "List singers.", "query": "SELECT name FROM singer;"},
                {"utterance": "Only distinct names.", "query": "SELECT DISTINCT name FROM singer;"},
            ],
        }
    )

    assert "Semantic model:" in messages[1]["content"]
    assert "Semantic model:" not in messages[3]["content"]


def test_format_cosql_can_prune_semantic_model_per_turn_and_add_hints() -> None:
    messages = format_cosql(
        {
            "database_id": "store",
            "schema": "CREATE TABLE customers (id INT);",
            "semantic_model_context": "\n".join(
                [
                    "- Cube customers (grain: one row per customers)",
                    "  Dimensions: id [number], name [string]",
                    "  Measures: count",
                    "- Cube orders (grain: one row per orders)",
                    "  Dimensions: id [number], customer_id [number]",
                    "  Measures: count",
                ]
            ),
            "include_sql_labels": True,
            "prune_semantic_model": True,
            "interaction": [
                {"utterance": "List customers.", "query": "SELECT name FROM customers;"},
                {"utterance": "How many orders?", "query": "SELECT COUNT(*) FROM orders;"},
            ],
        }
    )

    assert "Cube customers" in messages[1]["content"]
    assert "Cube orders" not in messages[1]["content"]
    assert "Oracle SQL planning hints" in messages[1]["content"]
    assert "Cube orders" in messages[3]["content"]
    assert "Cube customers" not in messages[3]["content"]
    assert "Relevant tables: orders" in messages[3]["content"]


def test_format_cosql_rejects_empty_interaction() -> None:
    with pytest.raises(FormatterError):
        format_cosql({"interaction": []})


def test_write_jsonl_round_trips_records(tmp_path) -> None:
    output = tmp_path / "train.jsonl"
    records = iter_formatted_records(
        [
            {
                "database_id": "hospital_1",
                "question": "Find the largest department.",
                "query": "SELECT name FROM department LIMIT 1;",
            }
        ],
        spec=DatasetSpec("jellyChiru/SParC", "train[:1]", "sparc"),
        limit=None,
    )

    assert write_jsonl(records, output) == 1
    row = json.loads(output.read_text().strip())
    assert row["source"] == "jellyChiru/SParC"
    assert row["database_id"] == "hospital_1"
    assert row["messages"][0]["role"] == "system"
    assert row["schema_link_labels"][0]["relevant_tables"] == ["department"]


def test_build_dataset_manifest_summarizes_composition(tmp_path) -> None:
    records = [
        {
            "source": "cosql",
            "evaluation_mode": "non_oracle_generation",
            "turn_format": "multi_turn_dialog",
            "history_policy": "gold_sql_teacher_forced",
            "assistant_turn_count": 2,
        },
        {
            "source": "sparc",
            "evaluation_mode": "non_oracle_generation",
            "turn_format": "single_turn",
            "history_policy": "single_turn",
            "assistant_turn_count": 1,
        },
    ]
    specs = [
        DatasetSpec("cosql", "train", "cosql", weight=0.5, include_sql_labels=False),
        DatasetSpec("sparc", "train", "sparc", weight=0.25, include_sql_labels=True),
    ]

    manifest = build_dataset_manifest(
        records=records,
        specs=specs,
        source_counts={"cosql": 1, "sparc": 1},
    )

    assert manifest["total_records"] == 2
    assert manifest["source_counts"] == {"cosql": 1, "sparc": 1}
    assert manifest["evaluation_modes"] == {"non_oracle_generation": 2}
    assert manifest["turn_formats"] == {"multi_turn_dialog": 1, "single_turn": 1}
    assert manifest["history_policies"] == {"gold_sql_teacher_forced": 1, "single_turn": 1}
    assert manifest["assistant_turns"] == {"total": 3, "max_per_record": 2}
    assert manifest["dataset_specs"][0]["configured_weight"] == 0.5
    assert manifest["dataset_specs"][1]["include_sql_labels"] is True

    output = tmp_path / "manifest.json"
    write_dataset_manifest(manifest, output)
    assert json.loads(output.read_text()) == manifest


def test_schema_from_spider_table_groups_columns() -> None:
    schema = schema_from_spider_table(
        {
            "db_id": "school",
            "table_names_original": ["students"],
            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
            "column_types": ["text", "number", "text"],
        }
    )

    assert schema == "students(id number, name text)"


def test_semantic_model_from_spider_table_summarizes_cube_members_and_joins() -> None:
    semantic_model = semantic_model_from_spider_table(
        {
            "db_id": "store",
            "table_names_original": ["customers", "orders"],
            "column_names_original": [
                [-1, "*"],
                [0, "customer_id"],
                [0, "name"],
                [1, "order_id"],
                [1, "customer_id"],
                [1, "amount"],
                [1, "created_at"],
            ],
            "column_types": ["text", "number", "text", "number", "number", "number", "time"],
            "primary_keys": [1, 3],
            "foreign_keys": [[4, 1]],
        }
    )

    assert "Cube customers" in semantic_model
    assert "customer_id [number, primary_key]" in semantic_model
    assert "sum_amount=sum(amount)" in semantic_model
    assert "orders.customer_id -> customers.customer_id (many_to_one)" in semantic_model


def test_iter_formatted_records_adds_semantic_model_from_tables_path(tmp_path) -> None:
    tables_path = tmp_path / "tables.json"
    tables_path.write_text(
        json.dumps(
            [
                {
                    "db_id": "store",
                    "table_names_original": ["orders"],
                    "column_names_original": [[-1, "*"], [0, "order_id"], [0, "amount"]],
                    "column_types": ["text", "number", "number"],
                    "primary_keys": [1],
                    "foreign_keys": [],
                }
            ]
        )
    )
    records = list(
        iter_formatted_records(
            [
                {
                    "database_id": "store",
                    "interaction": [
                        {"utterance": "Total amount?", "query": "SELECT SUM(amount) FROM orders;"},
                        {"utterance": "Now list order ids.", "query": "SELECT order_id FROM orders;"},
                    ],
                }
            ],
            spec=DatasetSpec(str(tmp_path / "cosql.json"), "train", "cosql", tables_path=str(tables_path)),
            limit=None,
        )
    )

    assert len(records) == 1
    assert "Semantic model:" in records[0]["messages"][1]["content"]
    assert "sum_amount=sum(amount)" in records[0]["messages"][1]["content"]
    assert records[0]["assistant_turn_count"] == 2
    assert records[0]["turn_format"] == "multi_turn_dialog"
    assert records[0]["history_policy"] == "gold_sql_teacher_forced"


def test_iter_formatted_records_respects_sql_label_and_pruning_spec(tmp_path) -> None:
    tables_path = tmp_path / "tables.json"
    tables_path.write_text(
        json.dumps(
            [
                {
                    "db_id": "store",
                    "table_names_original": ["customers", "orders"],
                    "column_names_original": [
                        [-1, "*"],
                        [0, "customer_id"],
                        [0, "name"],
                        [1, "order_id"],
                        [1, "amount"],
                    ],
                    "column_types": ["text", "number", "text", "number", "number"],
                    "primary_keys": [1, 3],
                    "foreign_keys": [],
                }
            ]
        )
    )
    records = list(
        iter_formatted_records(
            [
                {
                    "database_id": "store",
                    "interaction": [
                        {"utterance": "Total amount?", "query": "SELECT SUM(amount) FROM orders;"}
                    ],
                }
            ],
            spec=DatasetSpec(
                str(tmp_path / "cosql.json"),
                "train",
                "cosql",
                tables_path=str(tables_path),
                include_sql_labels=True,
                prune_semantic_model=True,
            ),
            limit=None,
        )
    )

    content = records[0]["messages"][1]["content"]
    assert "Oracle SQL planning hints" in content
    assert "Cube orders" in content
    assert "Cube customers" not in content
    assert records[0]["evaluation_mode"] == "oracle_planner_diagnostic"
    assert records[0]["uses_oracle_planning_hints"] is True
    assert records[0]["semantic_context_pruned_by_oracle_labels"] is True
    assert records[0]["planning_label_source"] == "gold_reference_sql"
    assert records[0]["gold_plans"][0]["relevant_tables"] == ["orders"]
    assert set(records[0]["gold_plans"][0]["query_skeleton"]) >= {"select", "where", "join"}
    assert "teacher-forced/oracle diagnostics" in records[0]["oracle_diagnostic_warning"]


def test_iter_formatted_records_marks_non_oracle_generation(tmp_path) -> None:
    records = list(
        iter_formatted_records(
            [
                {
                    "database_id": "store",
                    "question": "List customers.",
                    "query": "SELECT name FROM customers;",
                }
            ],
            spec=DatasetSpec(str(tmp_path / "sparc.json"), "train", "sparc"),
            limit=None,
        )
    )

    assert records[0]["evaluation_mode"] == "non_oracle_generation"
    assert records[0]["uses_oracle_planning_hints"] is False
    assert records[0]["semantic_context_pruned_by_oracle_labels"] is False
    assert records[0]["gold_plans"][0]["relevant_tables"] == ["customers"]
    assert records[0]["assistant_turn_count"] == 1
    assert records[0]["turn_format"] == "single_turn"
    assert records[0]["history_policy"] == "single_turn"
    assert "oracle_diagnostic_warning" not in records[0]


def test_local_json_dataset_loading(tmp_path) -> None:
    path = tmp_path / "cosql.json"
    path.write_text('[{"database_id": "db", "interaction": []}]')

    dataset = load_source_dataset(DatasetSpec(str(path), "train", "cosql"))

    assert dataset == [{"database_id": "db", "interaction": []}]


def test_parse_dataset_specs_supports_eval_section(tmp_path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """
data:
  train_datasets:
    - name: train.json
      formatter: cosql
  eval_datasets:
    - name: eval.json
      formatter: cosql
      tables_path: tables.json
"""
    )

    specs = parse_dataset_specs(config, section="eval")

    assert len(specs) == 1
    assert specs[0].name == "eval.json"
    assert specs[0].tables_path == "tables.json"
