"""Derive scorer-side Metric DSL gold labels for clean-holdout candidates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "metric_dsl_gold_label_rows"
SUMMARY_ARTIFACT_TYPE = "metric_dsl_gold_label_summary"
MANIFEST_ARTIFACT_TYPE = "metric_dsl_gold_label_manifest"
DEFAULT_INPUT = Path("data/processed/metric_dsl/clean_holdout_candidates.jsonl")
DEFAULT_OUTPUT = Path("data/processed/metric_dsl/clean_holdout_gold_labels.jsonl")
DEFAULT_SUMMARY = Path("docs/training_runs/metric_dsl_gold_labels_20260602.json")
DEFAULT_MANIFEST = Path("docs/data_artifacts/metric_dsl_gold_labels.manifest.json")
INCOMPLETE_LABEL_BLOCKER = "metric DSL gold label derivation incomplete"
NEXT_STEP = (
    "generate same-row Metric DSL predictions and direct-SQL predictions for the "
    "labelled clean-holdout rows, then run eval.run_metric_dsl_comparison"
)
AGGREGATE_RE = re.compile(
    r"\b(count|sum|avg|min|max)\s*\(\s*(distinct\s+)?([^)]+?)\s*\)",
    re.IGNORECASE,
)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip().strip("`\"[]").lower())


def _split_top_level_csv(value: str) -> list[str]:
    parts = []
    current = []
    depth = 0
    for char in value:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts


def _semantic_model_text(row: dict[str, Any]) -> str:
    for message in row.get("messages") or []:
        content = str(message.get("content") or "")
        marker = "Semantic model:\n"
        if marker not in content:
            continue
        semantic_text = content.split(marker, 1)[1]
        if "\n\nQuestion:" in semantic_text:
            semantic_text = semantic_text.split("\n\nQuestion:", 1)[0]
        return semantic_text.strip()
    raise ValueError("semantic model prompt context missing")


def _parse_semantic_model_text(text: str) -> dict[str, Any]:
    cubes: dict[str, dict[str, Any]] = {}
    current_table: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        cube_match = re.match(r"- Cube\s+(.+?)\s+\(", line)
        if cube_match:
            current_table = cube_match.group(1).strip()
            cubes[current_table] = {"dimensions": {}, "measures": {}, "joins": []}
            continue
        if current_table is None:
            continue
        if line.startswith("Dimensions: "):
            for item in _split_top_level_csv(line.removeprefix("Dimensions: ")):
                name = item.split("[", 1)[0].strip()
                if name:
                    key = _normalize(name)
                    cubes[current_table]["dimensions"][key] = f"{current_table}.{name}"
                    cubes[current_table]["dimensions"][f"{_normalize(current_table)}_{key}"] = (
                        f"{current_table}.{name}"
                    )
            continue
        if line.startswith("Measures: "):
            for item in _split_top_level_csv(line.removeprefix("Measures: ")):
                if "=" in item:
                    name, expression = item.split("=", 1)
                    measure_key = _normalize(name)
                    expression = expression.strip()
                    cubes[current_table]["measures"][measure_key] = (
                        _qualify_measure_expression(expression, current_table)
                    )
                elif item.strip().lower() == "count":
                    cubes[current_table]["measures"]["count"] = "COUNT(*)"
            continue
        if line.startswith("Joins: "):
            for item in _split_top_level_csv(line.removeprefix("Joins: ")):
                join = item.split("(", 1)[0].strip()
                if " -> " not in join:
                    continue
                left, right = [part.strip() for part in join.split(" -> ", 1)]
                left_table = left.split(".", 1)[0]
                right_table = right.split(".", 1)[0]
                cubes[current_table]["joins"].append(
                    {
                        "source_table": left_table,
                        "target_table": right_table,
                        "table": right_table,
                        "sql_on": f"{left} = {right}",
                    }
                )
    return {"cubes": cubes}


def _qualify_measure_expression(expression: str, table: str) -> str:
    def replace(match: re.Match[str]) -> str:
        function = match.group(1).upper()
        distinct = match.group(2) or ""
        column = match.group(3).strip()
        if column == "*":
            return f"{function}(*)"
        if "." not in column:
            column = f"{table}.{column}"
        return f"{function}({distinct}{column})"

    return AGGREGATE_RE.sub(replace, expression)


def _alias_map(sql: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+([`\"\[]?\w+[`\"\]]?)(?:\s+(?:AS\s+)?)?([`\"\[]?\w+[`\"\]]?)?",
        sql,
        re.IGNORECASE,
    ):
        table = match.group(1).strip("`\"[]")
        alias = (match.group(2) or table).strip("`\"[]")
        aliases[alias.lower()] = table.lower()
        aliases[table.lower()] = table.lower()
    return aliases


def _column_ref(value: str, aliases: dict[str, str] | None = None) -> tuple[str | None, str]:
    cleaned = value.strip().strip("`\"[]")
    if "." in cleaned:
        table, column = cleaned.rsplit(".", 1)
        table_key = table.strip("`\"[]").lower()
        if aliases and table_key in aliases:
            table_key = aliases[table_key]
        return table_key, column.strip("`\"[]")
    return None, cleaned


def _cube_for_table(semantic: dict[str, Any], table: str) -> tuple[str, dict[str, Any]]:
    normalized = table.lower()
    for table_name, cube in semantic["cubes"].items():
        if table_name.lower() == normalized:
            return table_name, cube
    raise KeyError(table)


def _aggregate_parts(expression: str) -> tuple[str, str | None, str] | None:
    match = AGGREGATE_RE.search(expression)
    if not match:
        return None
    function = match.group(1).lower()
    distinct = "distinct" if match.group(2) else None
    source = match.group(3).strip()
    return function, distinct, source


def _unsupported_shape(row: dict[str, Any]) -> str | None:
    sql = str(row.get("reference_sql") or "")
    if re.search(r"\b(UNION|INTERSECT|EXCEPT)\b", sql, re.IGNORECASE):
        return "unsupported SQL shape: set operation"
    skeleton = (row.get("gold_plan") or {}).get("query_skeleton") or {}
    if skeleton.get("nested"):
        return "unsupported SQL shape: nested query"
    if skeleton.get("having"):
        return "unsupported SQL shape: having"
    if skeleton.get("where"):
        return "unsupported SQL shape: where filter"
    if skeleton.get("order_by"):
        return "unsupported SQL shape: order by"
    return None


def _measure_for_aggregate(
    *,
    semantic: dict[str, Any],
    aggregate: tuple[str, str | None, str],
    relevant_tables: list[str],
    aliases: dict[str, str],
) -> tuple[str, str]:
    function, distinct, source = aggregate
    if function == "count" and source.strip() == "*":
        base_table = relevant_tables[0] if relevant_tables else next(iter(semantic["cubes"]))
        base_table, cube = _cube_for_table(semantic, base_table)
        if "count" not in cube["measures"]:
            raise ValueError("semantic measure not found for count")
        return base_table, "count"
    source_table, source_column = _column_ref(source, aliases)
    for table_name, cube in semantic["cubes"].items():
        if source_table and table_name.lower() != source_table:
            continue
        for measure_key, sql in cube["measures"].items():
            measure_aggregate = _aggregate_parts(str(sql))
            if measure_aggregate is None:
                continue
            measure_function, measure_distinct, measure_source = measure_aggregate
            measure_table, measure_column = _column_ref(measure_source)
            if (
                measure_function == function
                and bool(measure_distinct) == bool(distinct)
                and _normalize(measure_column) == _normalize(source_column)
                and (not source_table or measure_table == source_table)
            ):
                return table_name, measure_key
    raise ValueError("semantic measure not found for aggregate")


def _dimension_for_group_by(
    *,
    semantic: dict[str, Any],
    expression: str,
    aliases: dict[str, str],
) -> tuple[str, str]:
    table, column = _column_ref(expression, aliases)
    candidates = []
    for table_name, cube in semantic["cubes"].items():
        if table and table_name.lower() != table:
            continue
        for dimension_key, sql in cube["dimensions"].items():
            dim_table, dim_column = _column_ref(str(sql))
            if _normalize(dim_column) == _normalize(column) and (
                not table or dim_table == table
            ):
                candidates.append((table_name, dimension_key))
    # Prefer the business-level column key over table-qualified duplicate keys.
    candidates.sort(key=lambda item: (item[1].count("_"), item[1]))
    if not candidates:
        raise ValueError("semantic dimension not found for group-by")
    return candidates[0]


def _join_for_dimension(
    *,
    semantic: dict[str, Any],
    base_table: str,
    dimension_table: str,
    dimension_key: str,
) -> dict[str, Any] | None:
    if base_table == dimension_table:
        return None
    for join in semantic["cubes"].get(base_table, {}).get("joins", []):
        if join["target_table"].lower() == dimension_table.lower():
            return {
                "table": join["table"],
                "sql_on": join["sql_on"],
                "required_by": [dimension_key],
            }
    raise ValueError("semantic join not found for dimension")


def derive_metric_dsl_gold_label(row: dict[str, Any]) -> dict[str, Any]:
    """Return one candidate row with a scorer-side gold Metric DSL label when safe."""

    labelled = dict(row)
    labelled["artifact_type"] = ARTIFACT_TYPE
    labelled["schema_version"] = SCHEMA_VERSION
    labelled["gold_metric_dsl_available"] = False
    labelled.pop("gold_dsl", None)
    labelled.pop("reference_metric_dsl", None)
    labelled.pop("semantic_model", None)
    unsupported = _unsupported_shape(row)
    if unsupported:
        labelled["metric_dsl_label_error"] = unsupported
        labelled["readiness_blockers"] = [INCOMPLETE_LABEL_BLOCKER]
        return labelled
    try:
        semantic = _parse_semantic_model_text(_semantic_model_text(row))
        aliases = _alias_map(str(row.get("reference_sql") or ""))
        plan = row.get("gold_plan") or {}
        projection = plan.get("projection_shape") or {}
        aggregate_expressions = list(projection.get("aggregations") or [])
        if not aggregate_expressions:
            aggregate_expressions = [
                expression
                for expression in projection.get("selected_expressions") or []
                if _aggregate_parts(str(expression))
            ]
        if len(aggregate_expressions) != 1:
            raise ValueError("expected exactly one aggregate expression")
        aggregate = _aggregate_parts(str(aggregate_expressions[0]))
        if aggregate is None:
            raise ValueError("aggregate expression not parseable")
        base_table, measure_key = _measure_for_aggregate(
            semantic=semantic,
            aggregate=aggregate,
            relevant_tables=[str(table) for table in plan.get("relevant_tables") or []],
            aliases=aliases,
        )
        dimensions = []
        semantic_dimensions = {}
        joins = []
        for group_expression in projection.get("group_by") or []:
            dimension_table, dimension_key = _dimension_for_group_by(
                semantic=semantic,
                expression=str(group_expression),
                aliases=aliases,
            )
            dimensions.append(dimension_key)
            semantic_dimensions[dimension_key] = {
                "sql": semantic["cubes"][dimension_table]["dimensions"][dimension_key]
            }
            join = _join_for_dimension(
                semantic=semantic,
                base_table=base_table,
                dimension_table=dimension_table,
                dimension_key=dimension_key,
            )
            if join:
                joins.append(join)
        dsl = f"MEASURE({measure_key})"
        if dimensions:
            dsl = f"{dsl} BY {', '.join(dimensions)}"
        labelled["gold_dsl"] = dsl
        labelled["reference_metric_dsl"] = dsl
        labelled["semantic_model"] = {
            "base_table": base_table,
            "measures": {
                measure_key: {"sql": semantic["cubes"][base_table]["measures"][measure_key]}
            },
            "dimensions": semantic_dimensions,
            "joins": joins,
        }
        labelled["semantic_model_source"] = "prepared_prompt_context_non_oracle"
        labelled["metric_dsl_label_source"] = "clean_holdout_reference_sql_scorer_side"
        labelled["gold_metric_dsl_available"] = True
        labelled["readiness_blockers"] = []
        labelled.pop("metric_dsl_label_error", None)
    except Exception as exc:
        labelled["metric_dsl_label_error"] = str(exc)
        labelled["readiness_blockers"] = [INCOMPLETE_LABEL_BLOCKER]
    return labelled


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def label_metric_dsl_candidates(input_path: Path) -> list[dict[str, Any]]:
    """Return candidate rows augmented with derived gold-label status."""

    return [derive_metric_dsl_gold_label(row) for row in _load_jsonl(input_path)]


def summarize_metric_dsl_gold_labels(
    rows: list[dict[str, Any]], *, input_path: Path
) -> dict[str, Any]:
    labelled = [row for row in rows if row.get("gold_metric_dsl_available")]
    split_roles = Counter(str(row.get("split_role") or "unknown") for row in rows)
    split_ids = Counter(str(row.get("split_id") or "unknown") for row in rows)
    blockers = Counter(blocker for row in rows for blocker in row.get("readiness_blockers") or [])
    label_errors = Counter(
        str(row.get("metric_dsl_label_error"))
        for row in rows
        if row.get("metric_dsl_label_error")
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "checkpoint": 7,
        "run_id": "metric_dsl_gold_labels_20260602",
        "claim_boundary": (
            "Scorer-side clean-holdout Metric DSL gold-label derivation only; no "
            "generated DSL, compiled SQL execution score, value delta, or method win "
            "is claimed."
        ),
        "candidate_count": len(rows),
        "labelled_count": len(labelled),
        "unlabelled_count": len(rows) - len(labelled),
        "dialog_count": len({row.get("dialog_id") for row in rows}),
        "database_count": len({row.get("database_id") for row in rows if row.get("database_id")}),
        "split_ids": dict(split_ids),
        "split_roles": dict(split_roles),
        "readiness_blockers": dict(sorted(blockers.items())),
        "label_error_counts": dict(sorted(label_errors.items())),
        "promotion_status": "not_ready",
        "next_step": NEXT_STEP,
        "oracle_policy": "non_oracle_generation",
        "comparison_contract": "metric_dsl_clean_holdout_gold_labels",
        "reference_sql_visible_to_model_prompt": False,
        "scorer_fields_visible_to_model_prompt": False,
        "input_path": str(input_path),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_metric_dsl_gold_label_artifacts(
    *,
    input_path: Path,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    manifest_path: Path = DEFAULT_MANIFEST,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write labelled Metric DSL rows, summary, and manifest."""

    rows = label_metric_dsl_candidates(input_path)
    summary = summarize_metric_dsl_gold_labels(rows, input_path=input_path)
    output_rows = [row for row in rows if row.get("gold_metric_dsl_available")]
    _write_jsonl(output_path, output_rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "candidate_count": len(rows),
        "labelled_count": int(summary["labelled_count"]),
        "unlabelled_count": int(summary["unlabelled_count"]),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.exists() else None,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_row_count": len(output_rows),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "oracle_policy": "non_oracle_generation",
        "promotion_status": "not_ready",
        "command": list(command or []),
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = write_metric_dsl_gold_label_artifacts(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(
        f"Wrote {manifest['labelled_count']} labelled Metric DSL rows "
        f"from {manifest['candidate_count']} candidates to {args.output}"
    )
    return 0 if manifest["labelled_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
