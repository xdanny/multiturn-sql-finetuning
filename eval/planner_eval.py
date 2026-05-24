"""
Evaluate the planning step separately from SQL generation.

The gold plan is still extracted from reference SQL, so it is an evaluation
target, not an inference input. The lexical baseline below predicts a plan from
only the visible prompt context.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from data.plan_contract import PREDICTED_PLANNER, normalize_plan, validate_prepared_record_contract
from eval.run_eval import (
    assistant_turn_indices,
    load_prepared_records,
    record_uses_oracle_plan,
    write_results,
)

PLAN_FIELDS = (
    "table_f1",
    "column_f1",
    "join_f1",
    "skeleton_f1",
    "aggregation_f1",
    "group_by_f1",
    "selected_count_match",
    "duplicate_policy_match",
    "macro_planner_score",
)


def _normalize_identifier(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().strip("`\"[]").lower())


def _identifier_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.split(r"[^a-z0-9]+", value.lower())
        if len(token) > 1 and token not in {"the", "and", "for", "with", "from"}
    }


def _f1(gold: Iterable[Any], predicted: Iterable[Any]) -> float:
    gold_set = {_normalize_identifier(item) for item in gold if item not in (None, "")}
    predicted_set = {_normalize_identifier(item) for item in predicted if item not in (None, "")}
    if not gold_set and not predicted_set:
        return 1.0
    if not gold_set or not predicted_set:
        return 0.0
    overlap = len(gold_set & predicted_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_set)
    recall = overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


def _bool_f1(gold: dict[str, Any], predicted: dict[str, Any]) -> float:
    keys = set(gold) | set(predicted)
    if not keys:
        return 1.0
    gold_enabled = {key for key in keys if bool(gold.get(key))}
    predicted_enabled = {key for key in keys if bool(predicted.get(key))}
    return _f1(gold_enabled, predicted_enabled)


def score_plans(gold_plan: dict[str, Any] | None, predicted_plan: dict[str, Any] | None) -> dict[str, float]:
    """Score predicted planner output against gold SQL-derived planner labels."""

    gold = normalize_plan(gold_plan)
    predicted = normalize_plan(predicted_plan)
    gold_projection = gold["projection_shape"]
    predicted_projection = predicted["projection_shape"]
    scores = {
        "table_f1": _f1(gold["relevant_tables"], predicted["relevant_tables"]),
        "column_f1": _f1(gold["relevant_columns"], predicted["relevant_columns"]),
        "join_f1": _f1(gold["join_path"], predicted["join_path"]),
        "skeleton_f1": _bool_f1(gold["query_skeleton"], predicted["query_skeleton"]),
        "aggregation_f1": _f1(gold_projection["aggregations"], predicted_projection["aggregations"]),
        "group_by_f1": _f1(gold_projection["group_by"], predicted_projection["group_by"]),
        "selected_count_match": float(
            gold_projection["selected_count"] == predicted_projection["selected_count"]
        ),
        "duplicate_policy_match": float(
            gold_projection["preserve_duplicates"] == predicted_projection["preserve_duplicates"]
        ),
    }
    scores["macro_planner_score"] = sum(scores.values()) / len(scores)
    return scores


def _split_columns(raw_columns: str) -> list[str]:
    columns = []
    depth = 0
    current = []
    for char in raw_columns:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            columns.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        columns.append("".join(current).strip())
    return columns


def _column_name(raw_column: str) -> str | None:
    cleaned = raw_column.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered.startswith(("primary key", "foreign key", "constraint", "unique", "check")):
        return None
    return cleaned.split()[0].strip("`\"[]")


def extract_schema_inventory(messages: list[dict[str, str]]) -> dict[str, list[str]]:
    """Extract visible table/column names from prompt schema text."""

    inventory: dict[str, set[str]] = {}
    user_text = "\n".join(
        message.get("content", "") for message in messages if message.get("role") == "user"
    )

    for match in re.finditer(r"(?im)^\s*([A-Za-z_][\w]*)\(([^;\n]+)\)\s*$", user_text):
        table = _normalize_identifier(match.group(1))
        for raw_column in _split_columns(match.group(2)):
            column = _column_name(raw_column)
            if column:
                inventory.setdefault(table, set()).add(_normalize_identifier(column))

    for match in re.finditer(
        r"(?is)\bCREATE\s+TABLE\s+[`\"]?([A-Za-z_][\w]*)[`\"]?\s*\((.*?)\)\s*;",
        user_text,
    ):
        table = _normalize_identifier(match.group(1))
        for raw_column in _split_columns(match.group(2)):
            column = _column_name(raw_column)
            if column:
                inventory.setdefault(table, set()).add(_normalize_identifier(column))

    for match in re.finditer(r"(?im)^-\s+Cube\s+([A-Za-z_][\w]*)", user_text):
        inventory.setdefault(_normalize_identifier(match.group(1)), set())

    return {table: sorted(columns) for table, columns in sorted(inventory.items())}


def _visible_question_text(messages: list[dict[str, str]]) -> str:
    questions = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        match = re.search(r"(?is)\bQuestion:\s*(.*)$", content)
        questions.append(match.group(1) if match else content)
    if not questions:
        return ""
    return "\n".join(questions)


def _value_hint_matches_column(question_text: str, column: str) -> bool:
    lowered = question_text.lower()
    column = _normalize_identifier(column)
    if column in {"country", "nation", "nationality"}:
        return bool(re.search(r"\b(from|in|of)\s+[A-Z][a-z]+", question_text))
    if column in {"city", "town", "location", "place"}:
        return bool(re.search(r"\b(in|at|near|from|to)\s+[A-Z][a-z]+", question_text))
    if column in {"name", "title"}:
        return any(token in lowered for token in ("which", "list", "show"))
    return False


def lexical_planner(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Predict a weak non-oracle plan from prompt-visible schema and question text."""

    inventory = extract_schema_inventory(messages)
    question_text = _visible_question_text(messages)
    question_tokens = _identifier_tokens(question_text)
    selected_tables: set[str] = set()
    selected_columns: set[str] = set()

    for table, columns in inventory.items():
        table_tokens = _identifier_tokens(table)
        column_hits = []
        for column in columns:
            column_tokens = _identifier_tokens(column)
            if column_tokens & question_tokens or _value_hint_matches_column(question_text, column):
                column_hits.append(column)
                selected_columns.add(f"{table}.{column}")
        if table_tokens & question_tokens or column_hits:
            selected_tables.add(table)

    skeleton = {
        "select": True,
        "join": len(selected_tables) > 1,
        "where": any(token in question_tokens for token in {"where", "only", "with", "from", "in"}),
        "group_by": any(token in question_tokens for token in {"by", "per", "each", "group"}),
        "having": any(token in question_tokens for token in {"having"}),
        "order_by": any(token in question_tokens for token in {"top", "highest", "lowest", "most", "least"}),
        "limit": any(token in question_tokens for token in {"top", "first", "last"}),
        "nested": False,
        "distinct": any(token in question_tokens for token in {"distinct", "unique", "different"}),
    }
    aggregation_words = {
        "count": "count(*)",
        "number": "count(*)",
        "many": "count(*)",
        "average": "avg",
        "avg": "avg",
        "total": "sum",
        "sum": "sum",
        "maximum": "max",
        "max": "max",
        "highest": "max",
        "minimum": "min",
        "min": "min",
        "lowest": "min",
    }
    aggregations = sorted({value for token, value in aggregation_words.items() if token in question_tokens})
    return {
        "parseable": True,
        "prediction_source": "lexical_schema_baseline",
        "relevant_tables": sorted(selected_tables),
        "relevant_columns": sorted(selected_columns),
        "join_path": [],
        "query_skeleton": skeleton,
        "projection_shape": {
            "selected_count": max(1, len(selected_columns)) if selected_columns else 1,
            "aggregations": aggregations,
            "group_by": [],
            "preserve_duplicates": not skeleton["distinct"],
        },
    }


def evaluate_planner_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated = []
    for record in records:
        gold_plan = record.get("gold_plan") or record.get("schema_link_labels") or {}
        predicted_plan = record.get("predicted_plan") or lexical_planner(record["messages"])
        planner_scores = score_plans(gold_plan, predicted_plan)
        evaluated.append(
            {
                **record,
                "gold_plan": normalize_plan(gold_plan),
                "predicted_plan": normalize_plan(predicted_plan),
                "predicted_plan_source": predicted_plan.get("prediction_source", "unknown"),
                "planner_scores": planner_scores,
            }
        )
    return evaluated


def summarize_planner_scores(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": len(records),
        "dialog_count": len({record.get("dialog_id") for record in records}),
        "evaluation_modes": dict(Counter(record.get("evaluation_mode", "unknown") for record in records)),
        "sources": dict(Counter(record.get("source", "unknown") for record in records)),
        "prediction_sources": dict(
            Counter(record.get("predicted_plan_source", "unknown") for record in records)
        ),
        "oracle_prompt_rows": sum(
            1
            for record in records
            if any(
                marker in str(message.get("content", ""))
                for message in record.get("messages", [])
                for marker in ("Oracle SQL planning hints", "SQL planning hints:")
            )
        ),
    }
    for field in PLAN_FIELDS:
        summary[field] = (
            sum(record["planner_scores"][field] for record in records) / len(records)
            if records
            else 0.0
        )
    return summary


def run_planner_eval(
    *,
    input_path: Path,
    output: Path,
    summary_output: Path,
    limit: int | None,
    allow_oracle_plan: bool,
) -> int:
    records = load_prepared_records(input_path, limit=limit, allow_oracle_plan=allow_oracle_plan)
    evaluated = evaluate_planner_records(records)
    write_results(evaluated, output)
    summary = summarize_planner_scores(evaluated)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {len(evaluated)} planner rows to {output}")
    print(f"Wrote planner summary to {summary_output}")
    print(f"Mean planner score: {summary['macro_planner_score']:.3f}")
    return 0 if evaluated else 1


def _gold_plans_for_record(record: dict[str, Any], assistant_count: int) -> list[dict[str, Any]]:
    gold_plans = record.get("gold_plans")
    if isinstance(gold_plans, list) and len(gold_plans) == assistant_count:
        return [normalize_plan(plan) for plan in gold_plans]
    schema_link_labels = record.get("schema_link_labels")
    if isinstance(schema_link_labels, list):
        return [
            normalize_plan(schema_link_labels[index] if index < len(schema_link_labels) else {})
            for index in range(assistant_count)
        ]
    return [normalize_plan({}) for _ in range(assistant_count)]


def annotate_prepared_records_with_lexical_plans(
    input_path: Path,
    output_path: Path,
    *,
    limit: int | None,
) -> int:
    """Write dialog records with non-oracle lexical predicted planner output."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    expanded_turns = 0
    with input_path.open() as source, output_path.open("w") as target:
        for line in source:
            if not line.strip():
                continue
            if limit is not None and expanded_turns >= limit:
                break
            record = json.loads(line)
            if record_uses_oracle_plan(record):
                raise ValueError(
                    f"{input_path} contains gold SQL-derived oracle planning hints. "
                    "Predicted-planner artifacts must be generated from non-oracle prompts."
                )
            messages = record["messages"]
            assistant_indices = assistant_turn_indices(messages)
            if not assistant_indices:
                continue

            predicted_plans = []
            for assistant_index in assistant_indices:
                if limit is not None and expanded_turns >= limit:
                    break
                predicted_plans.append(lexical_planner(messages[:assistant_index]))
                expanded_turns += 1

            if not predicted_plans:
                break
            if len(predicted_plans) != len(assistant_indices):
                messages = messages[: assistant_indices[len(predicted_plans) - 1] + 1]
                assistant_indices = assistant_indices[: len(predicted_plans)]

            updated = dict(record)
            updated["messages"] = messages
            updated["gold_plans"] = _gold_plans_for_record(record, len(assistant_indices))
            updated["predicted_plans"] = predicted_plans
            updated["evaluation_mode"] = PREDICTED_PLANNER
            updated["uses_oracle_planning_hints"] = False
            updated["semantic_context_pruned_by_oracle_labels"] = False
            updated["predicted_plan_source"] = "lexical_schema_baseline"
            validate_prepared_record_contract(updated)
            target.write(json.dumps(updated, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/planner_eval.jsonl"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/planner_eval_summary.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow-oracle-plan",
        action="store_true",
        help="Allow inputs whose prompts already contain gold SQL-derived planning hints.",
    )
    parser.add_argument(
        "--predicted-prepared-output",
        type=Path,
        default=None,
        help="Also write prepared JSONL with non-oracle lexical predicted_plans.",
    )
    args = parser.parse_args()
    if args.predicted_prepared_output:
        count = annotate_prepared_records_with_lexical_plans(
            args.input,
            args.predicted_prepared_output,
            limit=args.limit,
        )
        print(f"Wrote {count} predicted-planner prepared records to {args.predicted_prepared_output}")
    return run_planner_eval(
        input_path=args.input,
        output=args.output,
        summary_output=args.summary_output,
        limit=args.limit,
        allow_oracle_plan=args.allow_oracle_plan,
    )


if __name__ == "__main__":
    raise SystemExit(main())
