"""Build train-split structured query-brief SFT rows from prepared SQL turns."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp

from eval.result_manifest import sha256_file
from eval.run_eval import load_prepared_records, record_uses_oracle_plan

NON_ORACLE_GENERATION_POLICY = "non_oracle_generation"
STRUCTURED_BRIEF_LABEL_SOURCE = "train_split_reference_sql_and_schema_labels"
STRUCTURED_BRIEF_SUPERVISION_POLICY = (
    "train_split_reference_sql_and_schema_labels_as_assistant_target"
)
STRUCTURED_BRIEF_PROMPT_POLICY = "visible_structured_brief_before_sql_v1"
STRUCTURED_BRIEF_SYSTEM_PROMPT = (
    "You are a SQL expert. First write a compact visible query brief starting "
    "with QUERY_BRIEF:, then write the SQL after SQL:. Do not include private "
    "chain-of-thought."
)


def _row_uses_scorer_derived_planning_hints(row: dict[str, Any]) -> bool:
    return record_uses_oracle_plan(row)


def _load_turn_records(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load prepared dialogs or already-expanded turn rows."""

    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    if not rows:
        return []
    if all("reference_sql" in row and "turn_index" in row for row in rows):
        for row in rows:
            if _row_uses_scorer_derived_planning_hints(row):
                raise ValueError(
                    f"{path} contains scorer-derived planning hints in expanded rows"
                )
        return rows
    return load_prepared_records(path, limit=limit, allow_oracle_plan=False)


def _model_instruction_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") in {"system", "user"}
    )


def _structured_prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": STRUCTURED_BRIEF_SYSTEM_PROMPT},
        *[dict(message) for message in messages if message.get("role") != "system"],
    ]


def structured_brief_eval_record_from_prepared_dialog(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Return a prepared eval dialog with the structured-brief prompt."""

    if _row_uses_scorer_derived_planning_hints(record):
        raise ValueError(f"{record.get('id') or record.get('dialog_id')}: scorer-derived hints leaked")
    return {
        **record,
        "messages": _structured_prompt_messages(record.get("messages", [])),
        "prompt_variant": "structured_brief_sql",
        "oracle_policy": NON_ORACLE_GENERATION_POLICY,
        "reference_sql_visible_to_model_prompt": False,
        "scorer_labels_visible_to_model_prompt": False,
    }


def build_structured_brief_eval_records(
    input_path: Path, *, limit: int | None = None
) -> list[dict[str, Any]]:
    rows = []
    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(
                structured_brief_eval_record_from_prepared_dialog(json.loads(line))
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _dialog_identity(row: dict[str, Any]) -> str:
    return str(
        row.get("dialog_id")
        or row.get("id")
        or row.get("split_row_id")
        or row.get("split_source_path")
        or "unknown"
    )


def _current_question(turn: dict[str, Any]) -> str:
    for message in reversed(turn.get("messages", [])):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", "")).strip()
        if "Question:" in content:
            content = content.rsplit("Question:", 1)[-1].strip()
        return " ".join(content.split())
    return "unknown"


def _parse_sql(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return None


def _ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _csv_or_none(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _tables(plan: dict[str, Any], parsed: exp.Expression | None) -> list[str]:
    planned = [str(table) for table in plan.get("relevant_tables") or []]
    if planned:
        return sorted(_ordered_unique(planned))
    if parsed is None:
        return []
    return sorted(_ordered_unique([table.name for table in parsed.find_all(exp.Table)]))


def _values(parsed: exp.Expression | None) -> list[str]:
    if parsed is None:
        return []
    values = []
    for literal in parsed.find_all(exp.Literal):
        values.append(literal.sql(dialect="sqlite"))
    return _ordered_unique(values)


def _metrics(plan: dict[str, Any], parsed: exp.Expression | None) -> list[str]:
    projection = plan.get("projection_shape") or {}
    planned = [str(value).lower() for value in projection.get("aggregations") or []]
    if planned:
        return _ordered_unique(planned)
    if parsed is None:
        return []
    aggregates = []
    for node in parsed.walk():
        if isinstance(node, exp.AggFunc):
            aggregates.append(node.key.lower())
    return _ordered_unique(aggregates)


def _filters(parsed: exp.Expression | None) -> str:
    if parsed is None:
        return "none"
    where = parsed.args.get("where")
    return where.this.sql(dialect="sqlite") if where is not None else "none"


def _grouping(parsed: exp.Expression | None) -> str:
    if parsed is None:
        return "not parsed"
    group = parsed.args.get("group")
    if group and group.expressions:
        return _csv_or_none([item.sql(dialect="sqlite") for item in group.expressions])
    if any(isinstance(node, exp.AggFunc) for node in parsed.walk()):
        return "single aggregate result"
    return "one row per selected record"


def _final_answer_shape(plan: dict[str, Any], parsed: exp.Expression | None) -> str:
    projection = plan.get("projection_shape") or {}
    selected_count = projection.get("selected_count")
    if selected_count is None and parsed is not None:
        selected_count = len(parsed.expressions)
    preserve_duplicates = projection.get("preserve_duplicates")
    parts = []
    if selected_count is not None:
        noun = "expression" if int(selected_count) == 1 else "expressions"
        parts.append(f"{int(selected_count)} selected {noun}")
    if preserve_duplicates is not None:
        parts.append(f"preserve_duplicates={bool(preserve_duplicates)}")
    if parsed is not None and parsed.args.get("order"):
        parts.append("ordered")
    if parsed is not None and parsed.args.get("limit"):
        parts.append("limited")
    return "; ".join(parts) if parts else "unspecified"


def structured_brief_for_turn(turn: dict[str, Any]) -> str:
    """Return a deterministic visible brief target for a train turn."""

    reference_sql = str(turn.get("reference_sql") or "").strip()
    plan = turn.get("gold_plan") or {}
    parsed = _parse_sql(reference_sql)
    lines = [
        "QUERY_BRIEF:",
        f"intent: {_current_question(turn)}",
        f"entities_and_values: {_csv_or_none(_values(parsed))}",
        f"metrics_or_measures: {_csv_or_none(_metrics(plan, parsed))}",
        f"filters: {_filters(parsed)}",
        f"grouping_and_grain: {_grouping(parsed)}",
        f"joins_or_table_families: {_csv_or_none(_tables(plan, parsed))}",
        f"final_answer_shape: {_final_answer_shape(plan, parsed)}",
        "SQL:",
        reference_sql,
    ]
    return "\n".join(lines)


def structured_brief_record_from_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Return one structured-brief SFT row for an expanded train-split turn."""

    if turn.get("split_role") != "train":
        raise ValueError(
            f"{turn.get('id')}: structured brief rows must come from split_role=train"
    )
    messages = _structured_prompt_messages(turn.get("messages", []))
    current_reference_sql = str(turn.get("reference_sql") or "")
    prompt_text = _model_instruction_text(messages)
    if current_reference_sql and current_reference_sql in prompt_text:
        raise ValueError(f"{turn.get('id')}: current reference SQL leaked into prompt")

    target = structured_brief_for_turn(turn)
    return {
        "id": str(turn["id"]),
        "dialog_id": str(turn.get("dialog_id") or ""),
        "turn_index": int(turn.get("turn_index") or 0),
        "turn_count": int(turn.get("turn_count") or 0),
        "database_id": turn.get("database_id"),
        "source": turn.get("source"),
        "split_id": turn.get("split_id"),
        "split_role": turn.get("split_role"),
        "split_row_id": turn.get("split_row_id"),
        "history_policy": turn.get("history_policy"),
        "evaluation_mode": turn.get("evaluation_mode") or NON_ORACLE_GENERATION_POLICY,
        "training_target": "structured_brief_sql",
        "messages": [*messages, {"role": "assistant", "content": target}],
        "structured_brief": target.split("SQL:", 1)[0].strip(),
        "structured_brief_label_source": STRUCTURED_BRIEF_LABEL_SOURCE,
        "oracle_policy": NON_ORACLE_GENERATION_POLICY,
        "supervision_policy": STRUCTURED_BRIEF_SUPERVISION_POLICY,
        "structured_brief_prompt_policy": STRUCTURED_BRIEF_PROMPT_POLICY,
        "reference_sql_visible_to_model_prompt": False,
        "scorer_labels_visible_to_model_prompt": False,
        "structured_brief_visible_as_assistant_label": True,
        "target_sql_visible_as_assistant_label": True,
    }


def build_structured_brief_training_records(
    input_path: Path, *, limit: int | None = None
) -> list[dict[str, Any]]:
    turns = _load_turn_records(input_path, limit=limit)
    return [structured_brief_record_from_turn(turn) for turn in turns]


def write_jsonl(rows: list[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def build_structured_brief_training_manifest(
    *,
    rows: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    split_ids = Counter(str(row.get("split_id") or "unknown") for row in rows)
    split_roles = Counter(str(row.get("split_role") or "unknown") for row in rows)
    database_ids = {str(row.get("database_id")) for row in rows if row.get("database_id")}
    return {
        "schema_version": 1,
        "artifact_type": "structured_brief_training_dataset",
        "row_count": len(rows),
        "dialog_count": len({row.get("dialog_id") for row in rows}),
        "database_count": len(database_ids),
        "split_ids": dict(split_ids),
        "split_roles": dict(split_roles),
        "training_target": "structured_brief_sql",
        "structured_brief_label_source": STRUCTURED_BRIEF_LABEL_SOURCE,
        "oracle_policy": NON_ORACLE_GENERATION_POLICY,
        "supervision_policy": STRUCTURED_BRIEF_SUPERVISION_POLICY,
        "structured_brief_prompt_policy": STRUCTURED_BRIEF_PROMPT_POLICY,
        "reference_sql_visible_to_model_prompt": False,
        "scorer_labels_visible_to_model_prompt": False,
        "structured_brief_visible_as_assistant_label": True,
        "target_sql_visible_as_assistant_label": True,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.exists() else None,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "command": list(command or []),
    }


def build_structured_brief_eval_manifest(
    *,
    rows: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    split_ids = Counter(str(row.get("split_id") or "unknown") for row in rows)
    split_roles = Counter(str(row.get("split_role") or "unknown") for row in rows)
    database_ids = {str(row.get("database_id")) for row in rows if row.get("database_id")}
    return {
        "schema_version": 1,
        "artifact_type": "structured_brief_eval_prepared_dataset",
        "row_count": len(rows),
        "dialog_count": len({_dialog_identity(row) for row in rows}),
        "database_count": len(database_ids),
        "split_ids": dict(split_ids),
        "split_roles": dict(split_roles),
        "prompt_variant": "structured_brief_sql",
        "oracle_policy": NON_ORACLE_GENERATION_POLICY,
        "structured_brief_prompt_policy": STRUCTURED_BRIEF_PROMPT_POLICY,
        "reference_sql_visible_to_model_prompt": False,
        "scorer_labels_visible_to_model_prompt": False,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.exists() else None,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path) if output_path.exists() else None,
        "command": list(command or []),
    }


def write_structured_brief_training_dataset(
    *,
    input_path: Path,
    output_path: Path,
    manifest_output_path: Path,
    limit: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = build_structured_brief_training_records(input_path, limit=limit)
    write_jsonl(rows, output_path)
    manifest = build_structured_brief_training_manifest(
        rows=rows,
        input_path=input_path,
        output_path=output_path,
        command=command,
    )
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def write_structured_brief_eval_dataset(
    *,
    input_path: Path,
    output_path: Path,
    manifest_output_path: Path,
    limit: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = build_structured_brief_eval_records(input_path, limit=limit)
    write_jsonl(rows, output_path)
    manifest = build_structured_brief_eval_manifest(
        rows=rows,
        input_path=input_path,
        output_path=output_path,
        command=command,
    )
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("training", "eval"),
        default="training",
        help="Write training SFT rows or prepared eval dialogs with structured-brief prompt.",
    )
    args = parser.parse_args()

    if args.mode == "eval":
        manifest = write_structured_brief_eval_dataset(
            input_path=args.input,
            output_path=args.output,
            manifest_output_path=args.manifest_output,
            limit=args.limit,
            command=sys.argv,
        )
    else:
        manifest = write_structured_brief_training_dataset(
            input_path=args.input,
            output_path=args.output,
            manifest_output_path=args.manifest_output,
            limit=args.limit,
            command=sys.argv,
        )
    print(
        f"Wrote {args.output} and {args.manifest_output} "
        f"({manifest['row_count']} structured-brief rows)"
    )
    return 0 if manifest["row_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
