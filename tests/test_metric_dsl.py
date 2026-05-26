from __future__ import annotations

from data.metric_dsl import (
    compile_metric_query,
    parse_metric_query,
    score_metric_query,
)

SEMANTIC_MODEL = {
    "base_table": "orders",
    "measures": {
        "revenue": {"sql": "SUM(orders.amount)"},
        "orders_count": {"sql": "COUNT(DISTINCT orders.id)"},
    },
    "dimensions": {
        "customer_country": {"sql": "customers.country"},
        "order_month": {"sql": "strftime('%Y-%m', orders.created_at)"},
    },
    "joins": [
        {
            "table": "customers",
            "sql_on": "orders.customer_id = customers.id",
            "required_by": ["customer_country"],
        }
    ],
}


def test_parse_metric_query_preserves_measure_tokens() -> None:
    query = parse_metric_query(
        "MEASURE(revenue), MEASURE(orders_count) BY customer_country, order_month "
        "WHERE customer_country = 'FR' ORDER BY MEASURE(revenue) DESC LIMIT 5"
    )

    assert query.measures == ("revenue", "orders_count")
    assert query.dimensions == ("customer_country", "order_month")
    assert query.filters == ("customer_country = 'FR'",)
    assert query.order_by == "MEASURE(revenue) DESC"
    assert query.limit == 5


def test_compile_metric_query_expands_governed_measures_at_the_boundary() -> None:
    query = parse_metric_query(
        "MEASURE(revenue) BY customer_country WHERE customer_country = 'FR' "
        "ORDER BY MEASURE(revenue) DESC LIMIT 5"
    )

    sql = compile_metric_query(query, SEMANTIC_MODEL)

    assert sql == (
        "SELECT customers.country AS customer_country, SUM(orders.amount) AS revenue "
        "FROM orders "
        "JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country = 'FR' "
        "GROUP BY customers.country "
        "ORDER BY revenue DESC "
        "LIMIT 5"
    )


def test_score_metric_query_separates_measure_preservation_from_dimensions() -> None:
    gold = parse_metric_query("MEASURE(revenue) BY customer_country")
    predicted = parse_metric_query("MEASURE(revenue) BY order_month")
    raw_sql_prediction = parse_metric_query("SUM(orders.amount) BY customer_country")

    score = score_metric_query(predicted, gold)
    raw_score = score_metric_query(raw_sql_prediction, gold)

    assert score["measure_f1"] == 1.0
    assert score["dimension_f1"] == 0.0
    assert score["measure_preservation"] == 1.0
    assert raw_score["measure_f1"] == 0.0
    assert raw_score["dimension_f1"] == 1.0
    assert raw_score["measure_preservation"] == 0.0
