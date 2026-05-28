"""Build tiny metric-DSL finetuning rows from synthetic method fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from data.synthetic_method_fixtures import build_synthetic_method_fixtures
from eval.result_manifest import sha256_file

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "metric_dsl_finetuning_rows"
DEFAULT_DSL_OUTPUT = Path("docs/data_artifacts/metric_dsl_training_rows.jsonl")
DEFAULT_DIRECT_SQL_OUTPUT = Path("docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/metric_dsl_training_rows_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/metric_dsl_training_rows.manifest.json")

METRIC_DSL_SYSTEM_PROMPT = (
    "You translate analytics questions into the project metric DSL, not SQL. "
    "Preserve governed metrics as MEASURE(name), choose business-level dimensions, "
    "and do not expand measures into raw table expressions."
)
DIRECT_SQL_SYSTEM_PROMPT = (
    "You are a SQL expert. Given database context and a user question, generate "
    "only the correct SQL query."
)
DSL_TARGET_OVERRIDES = {
    "grain_fanout_bridge": "MEASURE(revenue) BY campaign_name",
}


def _conversation_text(conversation: list[dict[str, Any]]) -> str:
    lines = []
    for turn in conversation:
        if "user" in turn:
            lines.append(f"User: {turn['user']}")
        if "assistant_sql" in turn:
            lines.append(f"Previous SQL: {turn['assistant_sql']}")
        if "observed_previous_rows" in turn:
            lines.append(f"Observed previous rows: {turn['observed_previous_rows']}")
    return "\n".join(lines)


def _metric_dsl_prompt(fixture: dict[str, Any]) -> str:
    visible = fixture["prompt_visible_input"]
    return (
        f"Fixture: {visible['fixture_id']}\n\n"
        f"Schema:\n{visible['schema_sql']}\n\n"
        f"Semantic model:\n{json.dumps(visible['semantic_model'], indent=2, sort_keys=True)}\n\n"
        f"Conversation:\n{_conversation_text(visible['conversation'])}\n\n"
        "Return only the metric DSL."
    )


def _direct_sql_prompt(fixture: dict[str, Any]) -> str:
    visible = fixture["prompt_visible_input"]
    return (
        f"Fixture: {visible['fixture_id']}\n\n"
        f"Schema:\n{visible['schema_sql']}\n\n"
        f"Semantic model:\n{json.dumps(visible['semantic_model'], indent=2, sort_keys=True)}\n\n"
        f"Conversation:\n{_conversation_text(visible['conversation'])}\n\n"
        "Return only SQL."
    )


def _dsl_target(fixture: dict[str, Any]) -> str | None:
    if fixture["fixture_id"] in DSL_TARGET_OVERRIDES:
        return DSL_TARGET_OVERRIDES[fixture["fixture_id"]]
    return fixture.get("gold_metric_dsl")


def _row(
    *,
    fixture: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    assistant_content: str,
    output_kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "fixture_id": fixture["fixture_id"],
        "schema_id": fixture["schema_id"],
        "training_target": output_kind,
        "failure_modes": fixture["failure_modes"],
        "required_artifacts": fixture["required_artifacts"],
        "oracle_policy": "synthetic_curated_label",
        "label_source": fixture["label_source"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def build_metric_dsl_finetuning_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return metric-DSL rows and same-fixture direct-SQL control rows."""

    dsl_rows = []
    direct_sql_rows = []
    for fixture in build_synthetic_method_fixtures():
        if "metric_dsl" not in fixture["training_targets"]:
            continue
        dsl_target = _dsl_target(fixture)
        if not dsl_target:
            continue
        dsl_rows.append(
            _row(
                fixture=fixture,
                system_prompt=METRIC_DSL_SYSTEM_PROMPT,
                user_prompt=_metric_dsl_prompt(fixture),
                assistant_content=dsl_target,
                output_kind="metric_dsl",
            )
        )
        direct_sql_rows.append(
            _row(
                fixture=fixture,
                system_prompt=DIRECT_SQL_SYSTEM_PROMPT,
                user_prompt=_direct_sql_prompt(fixture),
                assistant_content=fixture["reference_sql"],
                output_kind="direct_sql_control",
            )
        )
    return dsl_rows, direct_sql_rows


def summarize_metric_dsl_finetuning_rows(
    dsl_rows: list[dict[str, Any]],
    direct_sql_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_modes = Counter(mode for row in dsl_rows for mode in row["failure_modes"])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "metric_dsl_row_count": len(dsl_rows),
        "direct_sql_control_row_count": len(direct_sql_rows),
        "fixture_ids": [row["fixture_id"] for row in dsl_rows],
        "failure_mode_counts": dict(sorted(failure_modes.items())),
        "oracle_policy": "synthetic_curated_label",
        "comparison_contract": "same_fixture_metric_dsl_vs_direct_sql_control",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_metric_dsl_finetuning_artifacts(
    *,
    dsl_output_path: Path = DEFAULT_DSL_OUTPUT,
    direct_sql_output_path: Path = DEFAULT_DIRECT_SQL_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write metric-DSL rows, direct-SQL controls, summary, and manifest."""

    dsl_rows, direct_sql_rows = build_metric_dsl_finetuning_rows()
    summary = summarize_metric_dsl_finetuning_rows(dsl_rows, direct_sql_rows)
    _write_jsonl(dsl_output_path, dsl_rows)
    _write_jsonl(direct_sql_output_path, direct_sql_rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "metric_dsl_row_count": len(dsl_rows),
        "direct_sql_control_row_count": len(direct_sql_rows),
        "metric_dsl_output_path": str(dsl_output_path),
        "metric_dsl_output_sha256": sha256_file(dsl_output_path),
        "direct_sql_output_path": str(direct_sql_output_path),
        "direct_sql_output_sha256": sha256_file(direct_sql_output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "oracle_policy": "synthetic_curated_label",
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsl-output", type=Path, default=DEFAULT_DSL_OUTPUT)
    parser.add_argument("--direct-sql-output", type=Path, default=DEFAULT_DIRECT_SQL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    manifest = write_metric_dsl_finetuning_artifacts(
        dsl_output_path=args.dsl_output,
        direct_sql_output_path=args.direct_sql_output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(
        "Wrote "
        f"{manifest['metric_dsl_row_count']} metric DSL rows and "
        f"{manifest['direct_sql_control_row_count']} direct SQL control rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
