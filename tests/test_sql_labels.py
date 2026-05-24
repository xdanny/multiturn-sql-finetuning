from __future__ import annotations

from data.sql_labels import (
    labels_from_sql,
    planning_hint_from_labels,
    prune_semantic_model_context,
    schema_columns_from_context,
)


def test_labels_from_sql_extracts_schema_link_and_projection_shape() -> None:
    labels = labels_from_sql(
        "SELECT count(*), c.name FROM customers c JOIN orders o ON c.id = o.customer_id "
        "WHERE o.status = 'paid' GROUP BY c.name ORDER BY count(*) DESC LIMIT 3"
    )

    assert labels["parseable"]
    assert labels["relevant_tables"] == ["customers", "orders"]
    assert "customers.name" in labels["relevant_columns"]
    assert labels["join_path"] == ["c.id = o.customer_id"]
    assert labels["query_skeleton"]["join"]
    assert labels["query_skeleton"]["group_by"]
    assert labels["projection_shape"]["selected_count"] == 2
    assert labels["projection_shape"]["aggregations"] == ["count(*)"]
    assert labels["projection_shape"]["preserve_duplicates"]


def test_labels_from_sql_repairs_cosql_spaced_operator() -> None:
    labels = labels_from_sql(
        "SELECT name FROM singer GROUP BY name HAVING count(*) > = 3"
    )

    assert labels["parseable"]
    assert labels["query_skeleton"]["having"]


def test_labels_from_sql_repairs_cosql_missing_space_after_where() -> None:
    labels = labels_from_sql(
        "select Claim_ID from Claims_Processing_Stages AS T1 "
        "JOIN Claims_Processing AS T2 ON T1.Claim_Stage_ID = T2.Claim_Stage_ID "
        "WHEREClaim_Status_Name = 'Open'"
    )

    assert labels["parseable"]
    assert "claim_status_name" in labels["relevant_columns"]


def test_labels_from_sql_repairs_unmatched_trailing_parenthesis() -> None:
    labels = labels_from_sql(
        "SELECT countryname FROM countries WHERE countryid = 1 or countryid = 2 )"
    )

    assert labels["parseable"]
    assert labels["relevant_tables"] == ["countries"]


def test_labels_from_sql_repairs_alias_after_where_annotation() -> None:
    labels = labels_from_sql(
        "SELECT avg ( T2.rating ) FROM useracct WHERE T2.u_id = 1 AS T1 "
        "JOIN review AS T2 ON T1.u_id = T2.u_id GROUP BY T2.u_id"
    )

    assert labels["parseable"]
    assert labels["relevant_tables"] == ["review", "useracct"]
    assert labels["query_skeleton"]["join"]
    assert labels["query_skeleton"]["where"]


def test_labels_from_sql_filters_double_quoted_literal_values_from_columns() -> None:
    labels = labels_from_sql('SELECT result FROM battle WHERE date = "31 January 1206"')

    assert labels["relevant_columns"] == ["date", "result"]


def test_labels_from_sql_uses_schema_columns_to_filter_identifier_like_literals() -> None:
    labels = labels_from_sql(
        'SELECT "Name" FROM member WHERE role = "Violin"',
        schema_columns={"member.name", "member.role"},
    )

    assert labels["relevant_columns"] == ["name", "role"]


def test_schema_columns_from_context_extracts_qualified_and_unqualified_columns() -> None:
    columns = schema_columns_from_context("member(Member_ID text, Name text)\nperformance(Date time)")

    assert "member.member_id" in columns
    assert "member_id" in columns
    assert "performance.date" in columns


def test_labels_from_sql_extracts_projection_from_set_operation() -> None:
    labels = labels_from_sql("SELECT Nationality FROM HOST WHERE Age > 45 INTERSECT SELECT Nationality FROM HOST WHERE Age < 35")

    assert labels["relevant_columns"] == ["age", "nationality"]
    assert labels["query_skeleton"]["where"]
    assert labels["projection_shape"]["selected_expressions"] == ["nationality"]
    assert labels["projection_shape"]["selected_count"] == 1


def test_labels_from_sql_detects_nested_distinct_duplicate_policy() -> None:
    labels = labels_from_sql("SELECT count(*) FROM (SELECT DISTINCT treatment_type_code FROM treatments)")

    assert labels["query_skeleton"]["distinct"]
    assert labels["projection_shape"]["distinct"]
    assert not labels["projection_shape"]["preserve_duplicates"]


def test_prune_semantic_model_context_keeps_relevant_cube_blocks() -> None:
    semantic_model = "\n".join(
        [
            "- Cube customers (grain: one row per customers)",
            "  Dimensions: id [number], name [string]",
            "  Measures: count",
            "- Cube orders (grain: one row per orders)",
            "  Dimensions: id [number], customer_id [number]",
            "  Measures: count",
        ]
    )

    pruned = prune_semantic_model_context(semantic_model, ["orders"])

    assert "Cube orders" in pruned
    assert "Cube customers" not in pruned


def test_planning_hint_from_labels_includes_projection_and_duplicate_policy() -> None:
    labels = labels_from_sql("SELECT DISTINCT name FROM singer")

    hint = planning_hint_from_labels(labels)

    assert "Relevant tables: singer" in hint
    assert "Projection shape: 1 selected expression" in hint
    assert "Duplicate policy: deduplicate rows" in hint
