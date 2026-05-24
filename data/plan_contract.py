"""
Stable JSONL contract for planning-aware SQL evaluation.

Gold plans are answer-key labels extracted from reference SQL. They are valid
as scorer targets and supervised labels, but not as production prompt input.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

NON_ORACLE_GENERATION = "non_oracle_generation"
ORACLE_PLANNER_DIAGNOSTIC = "oracle_planner_diagnostic"
PREDICTED_PLANNER = "predicted_planner"

EVALUATION_MODES = {
    NON_ORACLE_GENERATION,
    ORACLE_PLANNER_DIAGNOSTIC,
    PREDICTED_PLANNER,
}

SKELETON_FIELDS = (
    "select",
    "join",
    "where",
    "group_by",
    "having",
    "order_by",
    "limit",
    "nested",
    "distinct",
)


def _normalize_identifier(value: Any) -> str:
    return " ".join(str(value).strip().strip("`\"[]").lower().split())


def _normalized_list(values: Iterable[Any] | None) -> list[str]:
    return sorted({_normalize_identifier(value) for value in values or [] if value not in (None, "")})


def normalize_plan(plan: dict[str, Any] | None) -> dict[str, Any]:
    """Return the stable planner schema used in JSONL outputs and summaries."""

    plan = plan or {}
    projection = plan.get("projection_shape") or {}
    skeleton = plan.get("query_skeleton") or {}
    return {
        "parseable": bool(plan.get("parseable", True)),
        "relevant_tables": _normalized_list(plan.get("relevant_tables")),
        "relevant_columns": _normalized_list(plan.get("relevant_columns")),
        "join_path": _normalized_list(plan.get("join_path")),
        "query_skeleton": {field: bool(skeleton.get(field, False)) for field in SKELETON_FIELDS},
        "projection_shape": {
            "selected_expressions": _normalized_list(projection.get("selected_expressions")),
            "selected_count": int(projection.get("selected_count") or 0),
            "aggregations": _normalized_list(projection.get("aggregations")),
            "group_by": _normalized_list(projection.get("group_by")),
            "order_by": _normalize_identifier(projection.get("order_by"))
            if projection.get("order_by")
            else None,
            "limit": _normalize_identifier(projection.get("limit")) if projection.get("limit") else None,
            "distinct": bool(projection.get("distinct", False)),
            "preserve_duplicates": bool(projection.get("preserve_duplicates", True)),
        },
    }


def evaluation_mode_from_flags(
    *,
    uses_oracle_planning_hints: bool,
    semantic_context_pruned_by_oracle_labels: bool,
    has_predicted_plan: bool = False,
) -> str:
    """Choose the evaluation mode from explicit data provenance flags."""

    if uses_oracle_planning_hints or semantic_context_pruned_by_oracle_labels:
        return ORACLE_PLANNER_DIAGNOSTIC
    if has_predicted_plan:
        return PREDICTED_PLANNER
    return NON_ORACLE_GENERATION


def validate_evaluation_mode(value: str) -> str:
    if value not in EVALUATION_MODES:
        raise ValueError(f"unknown evaluation_mode: {value}")
    return value


def assistant_turn_count(messages: list[dict[str, str]]) -> int:
    return sum(1 for message in messages if message.get("role") == "assistant")


def validate_prepared_record_contract(record: dict[str, Any]) -> None:
    """Validate the stable fields produced by data preparation."""

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("prepared record must include a non-empty messages list")

    mode = validate_evaluation_mode(str(record.get("evaluation_mode") or ""))
    gold_plans = record.get("gold_plans")
    if not isinstance(gold_plans, list):
        raise ValueError("prepared record must include gold_plans")
    if len(gold_plans) != assistant_turn_count(messages):
        raise ValueError("gold_plans length must match assistant turns")

    predicted_plans = record.get("predicted_plans")
    if predicted_plans is not None and not isinstance(predicted_plans, list):
        raise ValueError("predicted_plans must be a list when present")
    if isinstance(predicted_plans, list) and len(predicted_plans) != assistant_turn_count(messages):
        raise ValueError("predicted_plans length must match assistant turns")

    if mode == ORACLE_PLANNER_DIAGNOSTIC and record.get("planning_label_source") != "gold_reference_sql":
        raise ValueError("oracle planner diagnostic records must declare gold_reference_sql labels")

    if mode == PREDICTED_PLANNER and not predicted_plans:
        raise ValueError("predicted planner records must include predicted_plans")


def predicted_planning_hint_from_plan(plan: dict[str, Any]) -> str:
    """Format non-oracle planner output as prompt context."""

    normalized = normalize_plan(plan)
    skeleton = normalized["query_skeleton"]
    projection = normalized["projection_shape"]
    skeleton_flags = [name for name, enabled in skeleton.items() if enabled]
    lines = [
        "Predicted SQL plan (generated without reference SQL):",
        f"Relevant tables: {', '.join(normalized['relevant_tables']) or 'none'}",
        f"Relevant columns: {', '.join(normalized['relevant_columns']) or 'none'}",
        f"Join path: {'; '.join(normalized['join_path']) or 'none'}",
        f"Query skeleton: {', '.join(skeleton_flags) or 'select'}",
        (
            "Projection shape: "
            f"{projection['selected_count']} selected expression(s) in this order: "
            f"{'; '.join(projection['selected_expressions']) or 'unknown'}"
        ),
    ]
    if projection["aggregations"]:
        lines.append(f"Aggregation outputs: {', '.join(projection['aggregations'])}")
    if projection["group_by"]:
        lines.append(f"Group by: {', '.join(projection['group_by'])}")
    if projection["order_by"]:
        lines.append(f"Order by: {projection['order_by']}")
    if projection["limit"]:
        lines.append(f"Limit: {projection['limit']}")
    duplicate_note = "preserve duplicate rows" if projection["preserve_duplicates"] else "deduplicate rows"
    lines.append(f"Duplicate policy: {duplicate_note}")
    return "\n".join(lines)
