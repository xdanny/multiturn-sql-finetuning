"""
Data preparation for SQL fine-tuning.

Downloads accessible SQL datasets, formats each example as chat messages, and
writes JSONL records that TRL's SFTTrainer can consume directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from datasets import Dataset, load_dataset
from huggingface_hub.errors import HfHubHTTPError

from data.plan_contract import (
    assistant_turn_count,
    evaluation_mode_from_flags,
    normalize_plan,
    validate_prepared_record_contract,
)
from data.sql_labels import (
    labels_from_sql,
    planning_hint_from_labels,
    prune_semantic_model_context,
    schema_columns_from_context,
)

ORACLE_DIAGNOSTIC_WARNING = (
    "This record uses planning labels derived from reference SQL. Treat results as "
    "teacher-forced/oracle diagnostics, not production text-to-SQL accuracy."
)

SYSTEM_PROMPT = (
    "You are a SQL expert. Given database context and a user question, generate only the "
    "correct SQL query. Resolve follow-up questions against the conversation history. Use "
    "semantic model hints to choose entities, dimensions, measures, and joins, but write SQL "
    "against the physical table and column names. If the user request is ambiguous, ask one "
    "concise clarifying question."
)


class FormatterError(ValueError):
    """Raised when an example cannot be converted to training messages."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    split: str
    formatter: str
    weight: float = 1.0
    tables_path: str | None = None
    include_sql_labels: bool = False
    prune_semantic_model: bool = False


Formatter = Callable[[dict[str, Any]], list[dict[str, str]]]


def _required(example: dict[str, Any], key: str) -> Any:
    value = example.get(key)
    if value in (None, ""):
        raise FormatterError(f"missing required field: {key}")
    return value


def _user_content(
    question: str,
    *,
    schema: str | None = None,
    semantic_model: str | None = None,
    database_id: str | None = None,
    planning_hint: str | None = None,
) -> str:
    parts = []
    if database_id:
        parts.append(f"Database: {database_id}")
    if schema:
        parts.append(f"Schema/context:\n{schema}")
    if semantic_model:
        parts.append(f"Semantic model:\n{semantic_model}")
    if planning_hint:
        parts.append(planning_hint)
    parts.append(f"Question:\n{question}")
    return "\n\n".join(parts)


def format_sparc(example: dict[str, Any]) -> list[dict[str, str]]:
    """Format accessible SParC HF rows.

    The currently accessible `jellyChiru/SParC` dataset is flattened to
    database_id/question/query rows, not full interaction objects.
    """

    question = str(_required(example, "question"))
    query = str(_required(example, "query"))
    database_id = str(example.get("database_id") or example.get("db_id") or "")
    semantic_model = example.get("semantic_model_context")
    include_sql_labels = bool(example.get("include_sql_labels"))
    schema_columns = example.get("schema_columns")
    labels = labels_from_sql(query, schema_columns=schema_columns) if include_sql_labels else None
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _user_content(
                question,
                semantic_model=str(semantic_model) if semantic_model else None,
                database_id=database_id,
                planning_hint=planning_hint_from_labels(labels) if labels else None,
            ),
        },
        {"role": "assistant", "content": query},
    ]


def format_gretelai(example: dict[str, Any]) -> list[dict[str, str]]:
    """Format `gretelai/synthetic_text_to_sql` single-turn rows."""

    question = str(_required(example, "sql_prompt"))
    schema = str(_required(example, "sql_context"))
    query = str(_required(example, "sql"))
    semantic_model = example.get("semantic_model_context")
    include_sql_labels = bool(example.get("include_sql_labels"))
    schema_columns = example.get("schema_columns") or schema_columns_from_context(schema)
    labels = labels_from_sql(query, schema_columns=schema_columns) if include_sql_labels else None
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _user_content(
                question,
                schema=schema,
                semantic_model=str(semantic_model) if semantic_model else None,
                planning_hint=planning_hint_from_labels(labels) if labels else None,
            ),
        },
        {"role": "assistant", "content": query},
    ]


def format_bird(example: dict[str, Any]) -> list[dict[str, str]]:
    """Format BIRD mini-dev rows for evaluation-oriented SFT/debug data."""

    question = str(_required(example, "question"))
    query = str(_required(example, "SQL"))
    evidence = example.get("evidence")
    schema = f"Evidence:\n{evidence}" if evidence else None
    database_id = str(example.get("db_id") or "")
    semantic_model = example.get("semantic_model_context")
    include_sql_labels = bool(example.get("include_sql_labels"))
    schema_columns = example.get("schema_columns")
    labels = labels_from_sql(query, schema_columns=schema_columns) if include_sql_labels else None
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _user_content(
                question,
                schema=schema,
                semantic_model=str(semantic_model) if semantic_model else None,
                database_id=database_id,
                planning_hint=planning_hint_from_labels(labels) if labels else None,
            ),
        },
        {"role": "assistant", "content": query},
    ]


def _turn_text(turn: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = turn.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def format_cosql(example: dict[str, Any]) -> list[dict[str, str]]:
    """Format raw CoSQL-shaped interaction rows.

    CoSQL is not currently available at the repo's old HF ID. This formatter is
    intentionally schema-tolerant so local/raw CoSQL exports can still be used.
    """

    interaction = example.get("interaction") or example.get("interactions") or example.get("turns")
    if not isinstance(interaction, list) or not interaction:
        raise FormatterError("missing CoSQL interaction turns")

    schema = example.get("schema") or example.get("database_schema") or example.get("db_schema")
    semantic_model = example.get("semantic_model_context")
    include_sql_labels = bool(example.get("include_sql_labels"))
    prune_semantic_model = bool(example.get("prune_semantic_model"))
    database_id = example.get("database_id") or example.get("db_id")
    schema_columns = example.get("schema_columns") or schema_columns_from_context(str(schema) if schema else None)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    usable_turns = 0
    for index, turn in enumerate(interaction):
        if not isinstance(turn, dict):
            continue
        question = _turn_text(turn, "utterance", "question", "user", "text")
        query = _turn_text(turn, "query", "sql", "SQL")
        if not question or not query:
            continue
        labels = labels_from_sql(query, schema_columns=schema_columns)
        planning_hint = planning_hint_from_labels(labels) if include_sql_labels else None
        semantic_model_for_turn = str(semantic_model) if semantic_model else None
        if semantic_model_for_turn and prune_semantic_model:
            semantic_model_for_turn = prune_semantic_model_context(
                semantic_model_for_turn,
                labels.get("relevant_tables") or [],
            )
        messages.append(
            {
                "role": "user",
                "content": _user_content(
                    question,
                    schema=str(schema) if index == 0 and schema else None,
                    semantic_model=semantic_model_for_turn
                    if semantic_model_for_turn and (index == 0 or prune_semantic_model)
                    else None,
                    database_id=str(database_id) if index == 0 and database_id else None,
                    planning_hint=planning_hint,
                ),
            }
        )
        messages.append({"role": "assistant", "content": query})
        usable_turns += 1

    if usable_turns == 0:
        raise FormatterError("CoSQL example has no question/query turns")
    return messages


def schema_from_spider_table(table: dict[str, Any]) -> str:
    table_names = table.get("table_names_original") or table.get("table_names") or []
    columns = table.get("column_names_original") or table.get("column_names") or []
    column_types = table.get("column_types") or []
    grouped: dict[int, list[str]] = {index: [] for index, _ in enumerate(table_names)}
    for index, column in enumerate(columns):
        table_index, column_name = column
        if table_index == -1:
            continue
        column_type = column_types[index] if index < len(column_types) else "text"
        grouped.setdefault(table_index, []).append(f"{column_name} {column_type}")
    lines = []
    for table_index, table_name in enumerate(table_names):
        lines.append(f"{table_name}({', '.join(grouped.get(table_index, []))})")
    return "\n".join(lines)


def _identifier_like(column_name: str) -> bool:
    normalized = column_name.lower()
    return normalized in {"id", "uid"} or normalized.endswith("_id") or normalized.endswith(" id")


def _semantic_type(column_type: str) -> str:
    normalized = column_type.lower()
    if normalized in {"number", "integer", "int", "float", "double", "real", "decimal"}:
        return "number"
    if normalized in {"time", "date", "datetime", "timestamp"}:
        return "time"
    if normalized in {"boolean", "bool"}:
        return "boolean"
    return "string"


def _semantic_measure_name(aggregation: str, column_name: str) -> str:
    safe_name = column_name.lower().replace(" ", "_")
    return f"{aggregation}_{safe_name}"


def semantic_model_from_spider_table(table: dict[str, Any]) -> str:
    """Summarize Spider/CoSQL schema as Cube-inspired semantic model hints."""

    table_names = table.get("table_names_original") or table.get("table_names") or []
    columns = table.get("column_names_original") or table.get("column_names") or []
    column_types = table.get("column_types") or []
    primary_keys = set(table.get("primary_keys") or [])
    foreign_keys = table.get("foreign_keys") or []

    grouped: dict[int, list[tuple[int, str, str]]] = {index: [] for index, _ in enumerate(table_names)}
    for index, column in enumerate(columns):
        table_index, column_name = column
        if table_index == -1:
            continue
        column_type = column_types[index] if index < len(column_types) else "text"
        grouped.setdefault(table_index, []).append((index, str(column_name), _semantic_type(str(column_type))))

    joins_by_table: dict[int, list[str]] = {index: [] for index, _ in enumerate(table_names)}
    for source_column_index, target_column_index in foreign_keys:
        if source_column_index >= len(columns) or target_column_index >= len(columns):
            continue
        source_table_index, source_column = columns[source_column_index]
        target_table_index, target_column = columns[target_column_index]
        if source_table_index == -1 or target_table_index == -1:
            continue
        source_table = table_names[source_table_index]
        target_table = table_names[target_table_index]
        joins_by_table.setdefault(source_table_index, []).append(
            f"{source_table}.{source_column} -> {target_table}.{target_column} (many_to_one)"
        )

    lines = []
    for table_index, table_name in enumerate(table_names):
        columns_for_table = grouped.get(table_index, [])
        pk_names = [name for column_index, name, _ in columns_for_table if column_index in primary_keys]
        grain = f"one row per {table_name}"
        if pk_names:
            grain = f"{grain}; primary key: {', '.join(pk_names)}"
        lines.append(f"- Cube {table_name} (grain: {grain})")

        dimensions = []
        measures = ["count"]
        for column_index, column_name, semantic_type in columns_for_table:
            flags = []
            if column_index in primary_keys:
                flags.append("primary_key")
            dimension_suffix = f" [{semantic_type}{', ' + ', '.join(flags) if flags else ''}]"
            dimensions.append(f"{column_name}{dimension_suffix}")
            if semantic_type == "number" and column_index not in primary_keys and not _identifier_like(column_name):
                measures.append(f"{_semantic_measure_name('sum', column_name)}=sum({column_name})")
                measures.append(f"{_semantic_measure_name('avg', column_name)}=avg({column_name})")

        if dimensions:
            lines.append(f"  Dimensions: {', '.join(dimensions)}")
        lines.append(f"  Measures: {', '.join(measures)}")
        joins = joins_by_table.get(table_index) or []
        if joins:
            lines.append(f"  Joins: {', '.join(joins)}")

    return "\n".join(lines)


def load_schema_map(tables_path: str | None) -> dict[str, str]:
    if not tables_path:
        return {}
    path = Path(tables_path)
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {row["db_id"]: schema_from_spider_table(row) for row in rows}


def load_semantic_model_map(tables_path: str | None) -> dict[str, str]:
    if not tables_path:
        return {}
    path = Path(tables_path)
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    return {row["db_id"]: semantic_model_from_spider_table(row) for row in rows}


FORMATTERS: dict[str, Formatter] = {
    "sparc": format_sparc,
    "gretelai": format_gretelai,
    "bird": format_bird,
    "cosql": format_cosql,
}

DEFAULT_TRAIN_DATASETS = [
    DatasetSpec("jellyChiru/SParC", "train", "sparc", 0.50),
    DatasetSpec("gretelai/synthetic_text_to_sql", "train", "gretelai", 0.50),
]


def build_conversation(
    messages: list[dict[str, str]],
    source: str,
    *,
    database_id: str | None = None,
    schema_columns: Iterable[str] | None = None,
    uses_oracle_planning_hints: bool = False,
    semantic_context_pruned_by_oracle_labels: bool = False,
) -> dict[str, Any]:
    """Wrap chat messages with lightweight metadata for training/evaluation."""

    if len(messages) < 3:
        raise FormatterError("conversation must include system, user, and assistant messages")
    assistant_turns = assistant_turn_count(messages)
    record: dict[str, Any] = {
        "messages": messages,
        "source": source,
        "assistant_turn_count": assistant_turns,
        "turn_format": "multi_turn_dialog" if assistant_turns > 1 else "single_turn",
        "history_policy": "gold_sql_teacher_forced" if assistant_turns > 1 else "single_turn",
    }
    oracle_labels = [
        labels_from_sql(message["content"], schema_columns=schema_columns)
        for message in messages
        if message.get("role") == "assistant"
    ]
    gold_plans = [normalize_plan(labels) for labels in oracle_labels]
    record["schema_link_labels"] = oracle_labels
    record["gold_plans"] = gold_plans
    record["planning_label_source"] = "gold_reference_sql"
    record["uses_oracle_planning_hints"] = uses_oracle_planning_hints
    record["semantic_context_pruned_by_oracle_labels"] = semantic_context_pruned_by_oracle_labels
    record["evaluation_mode"] = evaluation_mode_from_flags(
        uses_oracle_planning_hints=uses_oracle_planning_hints,
        semantic_context_pruned_by_oracle_labels=semantic_context_pruned_by_oracle_labels,
    )
    if record["evaluation_mode"] == "oracle_planner_diagnostic":
        record["oracle_diagnostic_warning"] = ORACLE_DIAGNOSTIC_WARNING
    if database_id:
        record["database_id"] = database_id
    validate_prepared_record_contract(record)
    return record


def parse_dataset_specs(config_path: Path | None, *, section: str = "train") -> list[DatasetSpec]:
    if config_path is None:
        return DEFAULT_TRAIN_DATASETS

    with config_path.open() as f:
        config = yaml.safe_load(f)

    specs = []
    key = f"{section}_datasets"
    for item in config.get("data", {}).get(key, []):
        name = item["name"]
        formatter = item.get("formatter") or infer_formatter(name)
        specs.append(
            DatasetSpec(
                name=name,
                split=item.get("split", "train"),
                formatter=formatter,
                weight=float(item.get("weight", 1.0)),
                tables_path=item.get("tables_path"),
                include_sql_labels=bool(item.get("include_sql_labels", False)),
                prune_semantic_model=bool(item.get("prune_semantic_model", False)),
            )
        )
    return specs


def infer_formatter(dataset_name: str) -> str:
    lowered = dataset_name.lower()
    if "gretelai" in lowered:
        return "gretelai"
    if "sparc" in lowered:
        return "sparc"
    if "bird" in lowered:
        return "bird"
    if "cosql" in lowered:
        return "cosql"
    raise FormatterError(f"cannot infer formatter for dataset: {dataset_name}")


def load_source_dataset(spec: DatasetSpec) -> Dataset:
    path = Path(spec.name)
    if path.exists():
        return json.loads(path.read_text())
    return load_dataset(spec.name, split=spec.split)


def iter_formatted_records(
    dataset: Iterable[dict[str, Any]],
    *,
    spec: DatasetSpec,
    limit: int | None,
) -> Iterator[dict[str, Any]]:
    formatter = FORMATTERS[spec.formatter]
    schema_map = load_schema_map(spec.tables_path)
    semantic_model_map = load_semantic_model_map(spec.tables_path)
    for count, example in enumerate(dataset):
        if limit is not None and count >= limit:
            break
        row = dict(example)
        database_id = row.get("database_id") or row.get("db_id")
        schema = row.get("database_schema") or row.get("schema") or row.get("db_schema") or schema_map.get(database_id)
        if schema:
            row["schema_columns"] = schema_columns_from_context(str(schema))
            if spec.formatter == "cosql" and "database_schema" not in row:
                row["database_schema"] = schema
        if "semantic_model_context" not in row:
            semantic_model = semantic_model_map.get(database_id)
            if semantic_model:
                row["semantic_model_context"] = semantic_model
        row["include_sql_labels"] = spec.include_sql_labels
        row["prune_semantic_model"] = spec.prune_semantic_model
        messages = formatter(row)
        yield build_conversation(
            messages,
            source=spec.name,
            database_id=str(database_id) if database_id else None,
            schema_columns=row.get("schema_columns"),
            uses_oracle_planning_hints=spec.include_sql_labels,
            semantic_context_pruned_by_oracle_labels=spec.prune_semantic_model,
        )


def write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_dataset_manifest(
    *,
    records: list[dict[str, Any]],
    specs: list[DatasetSpec],
    source_counts: dict[str, int],
) -> dict[str, Any]:
    """Summarize prepared-data composition for reproducible training claims."""

    return {
        "schema_version": 1,
        "total_records": len(records),
        "source_counts": dict(source_counts),
        "evaluation_modes": dict(Counter(str(record.get("evaluation_mode", "unknown")) for record in records)),
        "turn_formats": dict(Counter(str(record.get("turn_format", "unknown")) for record in records)),
        "history_policies": dict(Counter(str(record.get("history_policy", "unknown")) for record in records)),
        "assistant_turns": {
            "total": sum(int(record.get("assistant_turn_count") or 0) for record in records),
            "max_per_record": max((int(record.get("assistant_turn_count") or 0) for record in records), default=0),
        },
        "dataset_specs": [
            {
                "name": spec.name,
                "split": spec.split,
                "formatter": spec.formatter,
                "configured_weight": spec.weight,
                "tables_path": spec.tables_path,
                "include_sql_labels": spec.include_sql_labels,
                "prune_semantic_model": spec.prune_semantic_model,
            }
            for spec in specs
        ],
    }


def write_dataset_manifest(manifest: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--section", choices=["train", "eval"], default="train")
    parser.add_argument("--output", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=None,
        help="Optional JSON summary of prepared-data composition and provenance.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit examples per dataset")
    parser.add_argument("--strict", action="store_true", help="Fail if a configured dataset is unavailable")
    parser.add_argument(
        "--include-sql-labels",
        action="store_true",
        help="Add gold SQL-derived schema-link and projection planning hints to prompts.",
    )
    parser.add_argument(
        "--prune-semantic-model",
        action="store_true",
        help="Prune semantic model context to gold relevant tables for each turn.",
    )
    args = parser.parse_args()

    specs = parse_dataset_specs(args.config, section=args.section)
    if args.include_sql_labels or args.prune_semantic_model:
        print(
            "WARNING: creating oracle planner diagnostic data from gold/reference SQL. "
            "Do not report these runs as production text-to-SQL accuracy.",
            file=sys.stderr,
        )
        specs = [
            replace(
                spec,
                include_sql_labels=spec.include_sql_labels or args.include_sql_labels,
                prune_semantic_model=spec.prune_semantic_model or args.prune_semantic_model,
            )
            for spec in specs
        ]
    all_records: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for spec in specs:
        try:
            dataset = load_source_dataset(spec)
            records = list(iter_formatted_records(dataset, spec=spec, limit=args.limit))
        except (HfHubHTTPError, FileNotFoundError, FormatterError, ValueError) as exc:
            if args.strict:
                raise
            print(f"Skipping {spec.name}: {exc}", file=sys.stderr)
            counts[spec.name] = 0
            continue

        all_records.extend(records)
        counts[spec.name] = len(records)

    total = write_jsonl(all_records, args.output)
    if args.manifest_output:
        write_dataset_manifest(
            build_dataset_manifest(records=all_records, specs=specs, source_counts=counts),
            args.manifest_output,
        )
    print(f"Wrote {args.output}")
    if args.manifest_output:
        print(f"Wrote dataset manifest: {args.manifest_output}")
    for name, count in counts.items():
        print(f"  {name}: {count} examples")
    print(f"  total: {total} examples")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
