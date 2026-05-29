"""Build semantic value-retrieval prepared inputs from a database value index."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from data.value_index import normalize_value_token
from eval.result_manifest import sha256_file

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "semantic_value_retrieval_prepared_inputs"
DEFAULT_INPUT = Path("data/processed/eval_cosql_dev_100.jsonl")
DEFAULT_VALUE_INDEX = Path("docs/data_artifacts/value_index_cosql_dev_100.jsonl")
DEFAULT_VALUE_INDEX_MANIFEST = Path("docs/data_artifacts/value_index_cosql_dev_100.manifest.json")
DEFAULT_OUTPUT = Path("data/processed/eval_cosql_dev_100_semantic_value_retrieval.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path(
    "docs/data_artifacts/semantic_value_retrieval_inputs_summary.json"
)
DEFAULT_MANIFEST_OUTPUT = Path(
    "docs/data_artifacts/semantic_value_retrieval_inputs.manifest.json"
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _validate_value_index_manifest(manifest_path: Path, value_index_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("artifact_type") != "non_oracle_value_index_v1":
        raise ValueError("value-index manifest must use artifact_type=non_oracle_value_index_v1")
    if manifest.get("index_source") != "database_contents":
        raise ValueError("value-index manifest must use index_source=database_contents")
    if manifest.get("output_sha256") != sha256_file(value_index_path):
        raise ValueError("value-index manifest output_sha256 does not match value index")
    return manifest


def _index_by_database(index_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_rows:
        if row.get("index_source") != "database_contents":
            raise ValueError("semantic value retrieval inputs require database-derived index rows")
        grouped[str(row.get("database_id"))].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                -int(row.get("source_frequency") or 0),
                str(row.get("table") or ""),
                str(row.get("column") or ""),
                str(row.get("raw_value") or ""),
            )
        )
    return dict(grouped)


def _assistant_turn_count(messages: list[dict[str, Any]]) -> int:
    return sum(1 for message in messages if message.get("role") == "assistant")


def _user_visible_text(messages: list[dict[str, Any]], *, through_index: int) -> str:
    return "\n".join(
        str(message.get("content") or "")
        for message in messages[: through_index + 1]
        if message.get("role") == "user"
    )


def _alias_matches(normalized_text: str, normalized_alias: str) -> bool:
    if not normalized_alias:
        return False
    return bool(re.search(rf"(?:^|\s){re.escape(normalized_alias)}(?:\s|$)", normalized_text))


def retrieve_value_matches(
    *,
    text: str,
    value_index_rows: list[dict[str, Any]],
    max_matches: int = 12,
) -> list[dict[str, Any]]:
    """Return database-derived value matches for user-authored text."""

    normalized_text = normalize_value_token(text)
    matches = []
    seen = set()
    for row in value_index_rows:
        matched_alias = None
        for alias in row.get("aliases") or []:
            normalized_alias = normalize_value_token(alias)
            if _alias_matches(normalized_text, normalized_alias):
                matched_alias = str(alias)
                break
        if matched_alias is None:
            continue
        key = (row.get("table"), row.get("column"), row.get("raw_value"))
        if key in seen:
            continue
        seen.add(key)
        matches.append(
            {
                "table": row.get("table"),
                "column": row.get("column"),
                "raw_value": row.get("raw_value"),
                "matched_alias": matched_alias,
                "source_frequency": row.get("source_frequency"),
            }
        )
        if len(matches) >= max_matches:
            break
    return matches


def _retrieval_block(matches: list[dict[str, Any]]) -> str:
    lines = [
        "Database value retrieval (non-oracle; matched only against user text so far):"
    ]
    for match in matches:
        lines.append(
            "- "
            f"{match['table']}.{match['column']} = {json.dumps(match['raw_value'])} "
            f"(matched: {json.dumps(match['matched_alias'])})"
        )
    return "\n".join(lines)


def add_semantic_value_retrieval_context(
    record: dict[str, Any],
    *,
    value_index_by_database: dict[str, list[dict[str, Any]]],
    max_matches_per_turn: int = 12,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one prepared record with retrieval context added to user turns."""

    if record.get("uses_oracle_planning_hints") or record.get(
        "semantic_context_pruned_by_oracle_labels"
    ):
        raise ValueError("semantic value-retrieval inputs must be non-oracle")
    messages = [dict(message) for message in record.get("messages") or []]
    database_id = str(record.get("database_id") or "")
    database_index = value_index_by_database.get(database_id) or []
    turn_match_counts = []
    matched_values = 0
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        visible_text = _user_visible_text(messages, through_index=index)
        matches = retrieve_value_matches(
            text=visible_text,
            value_index_rows=database_index,
            max_matches=max_matches_per_turn,
        )
        turn_match_counts.append(len(matches))
        if not matches:
            continue
        matched_values += len(matches)
        message["content"] = f"{message.get('content')}\n\n{_retrieval_block(matches)}"
    updated = {
        **record,
        "messages": messages,
        "evaluation_mode": "non_oracle_generation",
        "semantic_value_retrieval": {
            "artifact_type": "semantic_value_retrieval_context",
            "index_source": "database_contents",
            "matched_turn_count": sum(1 for count in turn_match_counts if count > 0),
            "matched_value_count": matched_values,
            "max_matches_per_turn": max_matches_per_turn,
            "leakage_boundary": "matches use user-authored text up to each turn only",
        },
    }
    summary = {
        "assistant_turn_count": _assistant_turn_count(messages),
        "user_turn_count": len(turn_match_counts),
        "matched_turn_count": sum(1 for count in turn_match_counts if count > 0),
        "matched_value_count": matched_values,
        "database_id": database_id,
    }
    return updated, summary


def build_semantic_value_retrieval_inputs(
    *,
    prepared_rows: list[dict[str, Any]],
    value_index_rows: list[dict[str, Any]],
    max_matches_per_turn: int = 12,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build semantic value-retrieval prepared rows and a summary."""

    value_index_by_database = _index_by_database(value_index_rows)
    output_rows = []
    row_summaries = []
    for record in prepared_rows:
        updated, summary = add_semantic_value_retrieval_context(
            record,
            value_index_by_database=value_index_by_database,
            max_matches_per_turn=max_matches_per_turn,
        )
        output_rows.append(updated)
        row_summaries.append(summary)

    matched_rows = [summary for summary in row_summaries if summary["matched_value_count"] > 0]
    database_counts = Counter(summary["database_id"] for summary in row_summaries)
    return output_rows, {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "row_count": len(output_rows),
        "assistant_turn_count": sum(summary["assistant_turn_count"] for summary in row_summaries),
        "matched_row_count": len(matched_rows),
        "matched_turn_count": sum(summary["matched_turn_count"] for summary in row_summaries),
        "matched_value_count": sum(summary["matched_value_count"] for summary in row_summaries),
        "database_count": len(database_counts),
        "database_row_counts": dict(sorted(database_counts.items())),
        "max_matches_per_turn": max_matches_per_turn,
        "oracle_policy": "non_oracle_database_value_index_matched_to_user_text_only",
        "leakage_boundary": (
            "no reference SQL, gold plans, expected rows, assistant SQL, or future user turns "
            "are used for retrieval matching"
        ),
        "evaluation_gate": "eval.run_semantic_value_retrieval_comparison",
    }


def write_semantic_value_retrieval_input_artifacts(
    *,
    input_path: Path = DEFAULT_INPUT,
    value_index_path: Path = DEFAULT_VALUE_INDEX,
    value_index_manifest_path: Path = DEFAULT_VALUE_INDEX_MANIFEST,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    max_matches_per_turn: int = 12,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write semantic prepared inputs, summary, and manifest."""

    value_index_manifest = _validate_value_index_manifest(
        value_index_manifest_path,
        value_index_path,
    )
    output_rows, summary = build_semantic_value_retrieval_inputs(
        prepared_rows=_load_jsonl(input_path),
        value_index_rows=_load_jsonl(value_index_path),
        max_matches_per_turn=max_matches_per_turn,
    )
    _write_jsonl(output_path, output_rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "value_index_path": str(value_index_path),
        "value_index_sha256": sha256_file(value_index_path),
        "value_index_manifest_path": str(value_index_manifest_path),
        "value_index_manifest_sha256": sha256_file(value_index_manifest_path),
        "value_index_index_source": value_index_manifest.get("index_source"),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "row_count": summary["row_count"],
        "assistant_turn_count": summary["assistant_turn_count"],
        "matched_row_count": summary["matched_row_count"],
        "matched_turn_count": summary["matched_turn_count"],
        "matched_value_count": summary["matched_value_count"],
        "oracle_policy": summary["oracle_policy"],
        "leakage_boundary": summary["leakage_boundary"],
        "evaluation_gate": summary["evaluation_gate"],
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--value-index", type=Path, default=DEFAULT_VALUE_INDEX)
    parser.add_argument(
        "--value-index-manifest",
        type=Path,
        default=DEFAULT_VALUE_INDEX_MANIFEST,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--max-matches-per-turn", type=int, default=12)
    args = parser.parse_args()

    command = [
        "python",
        "-m",
        "data.semantic_value_retrieval_inputs",
        "--input",
        str(args.input),
        "--value-index",
        str(args.value_index),
        "--value-index-manifest",
        str(args.value_index_manifest),
        "--output",
        str(args.output),
        "--summary-output",
        str(args.summary_output),
        "--manifest-output",
        str(args.manifest_output),
        "--max-matches-per-turn",
        str(args.max_matches_per_turn),
    ]
    manifest = write_semantic_value_retrieval_input_artifacts(
        input_path=args.input,
        value_index_path=args.value_index,
        value_index_manifest_path=args.value_index_manifest,
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        max_matches_per_turn=args.max_matches_per_turn,
        command=command,
    )
    print(
        "Wrote semantic value-retrieval prepared inputs: "
        f"{manifest['row_count']} rows, "
        f"{manifest['matched_turn_count']} matched turns, "
        f"{manifest['matched_value_count']} matched values"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
