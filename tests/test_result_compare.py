from __future__ import annotations

import pandas as pd

from eval.result_compare import compare_dataframes, is_order_sensitive_sql


def test_value_match_ignores_harmless_alias_differences() -> None:
    reference = pd.DataFrame({"count(*)": [2]})
    generated = pd.DataFrame({"count ( * )": [2]})

    comparison = compare_dataframes(reference, generated, order_sensitive=False)

    assert comparison.value_match
    assert not comparison.strict_match


def test_order_insensitive_comparison_preserves_duplicates() -> None:
    reference = pd.DataFrame({"name": ["Ada", "Ada", "Grace"]})
    generated = pd.DataFrame({"name": ["Grace", "Ada", "Ada"]})

    comparison = compare_dataframes(reference, generated, order_sensitive=False)

    assert comparison.value_match


def test_order_sensitive_comparison_preserves_topk_order() -> None:
    reference = pd.DataFrame({"name": ["Ada", "Grace"]})
    generated = pd.DataFrame({"name": ["Grace", "Ada"]})

    comparison = compare_dataframes(reference, generated, order_sensitive=True)

    assert not comparison.value_match


def test_float_values_compare_with_tolerance() -> None:
    reference = pd.DataFrame({"ratio": [0.3333333]})
    generated = pd.DataFrame({"ratio": [0.3333334]})

    comparison = compare_dataframes(reference, generated, order_sensitive=False)

    assert comparison.value_match


def test_null_values_match() -> None:
    reference = pd.DataFrame({"value": [None]})
    generated = pd.DataFrame({"value": [None]})

    comparison = compare_dataframes(reference, generated, order_sensitive=False)

    assert comparison.value_match


def test_order_sensitive_sql_detects_order_or_limit() -> None:
    assert is_order_sensitive_sql("SELECT name FROM users ORDER BY name")
    assert is_order_sensitive_sql("SELECT name FROM users LIMIT 1")
    assert not is_order_sensitive_sql("SELECT name FROM users")
