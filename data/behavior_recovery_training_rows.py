"""Build tiny behavior-recovery finetuning rows from synthetic fixtures."""

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
ARTIFACT_TYPE = "behavior_recovery_finetuning_rows"
DEFAULT_RECOVERY_OUTPUT = Path("docs/data_artifacts/behavior_recovery_training_rows.jsonl")
DEFAULT_CONTROL_OUTPUT = Path("docs/data_artifacts/behavior_recovery_direct_sql_training_rows.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/behavior_recovery_training_rows_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/behavior_recovery_training_rows.manifest.json")

RECOVERY_SYSTEM_PROMPT = (
    "You are a SQL repair assistant for multi-turn analytics. Use the conversation, "
    "previous generated SQL, observed rows, schema, and semantic model to repair the "
    "next SQL query. Do not repeat a value-grounding mistake after an empty result."
)
DIRECT_SQL_SYSTEM_PROMPT = (
    "You are a SQL expert. Given database context and a user question, generate "
    "only the correct SQL query."
)


def _conversation_text(conversation: list[dict[str, Any]]) -> str:
    lines = []
    for turn in conversation:
        if "user" in turn:
            lines.append(f"User: {turn['user']}")
        if "assistant_sql" in turn:
            lines.append(f"Previous generated SQL: {turn['assistant_sql']}")
        if "observed_previous_rows" in turn:
            lines.append(f"Observed previous rows: {turn['observed_previous_rows']}")
    return "\n".join(lines)


def _prompt(fixture: dict[str, Any], *, output_instruction: str) -> str:
    visible = fixture["prompt_visible_input"]
    return (
        f"Fixture: {visible['fixture_id']}\n\n"
        f"Schema:\n{visible['schema_sql']}\n\n"
        f"Semantic model:\n{json.dumps(visible['semantic_model'], indent=2, sort_keys=True)}\n\n"
        f"Conversation and generated-history trace:\n"
        f"{_conversation_text(visible['conversation'])}\n\n"
        f"{output_instruction}"
    )


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
        "leakage_policy": "synthetic_curated_label",
        "label_source": fixture["label_source"],
        "history_policy": "generated_history_trace",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
    }


def build_behavior_recovery_finetuning_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return recovery rows and same-fixture direct-SQL control rows."""

    recovery_rows = []
    control_rows = []
    for fixture in build_synthetic_method_fixtures():
        if "behavior_recovery" not in fixture["training_targets"]:
            continue
        recovery_rows.append(
            _row(
                fixture=fixture,
                system_prompt=RECOVERY_SYSTEM_PROMPT,
                user_prompt=_prompt(fixture, output_instruction="Return only the repaired SQL."),
                assistant_content=fixture["reference_sql"],
                output_kind="behavior_recovery",
            )
        )
        control_rows.append(
            _row(
                fixture=fixture,
                system_prompt=DIRECT_SQL_SYSTEM_PROMPT,
                user_prompt=_prompt(fixture, output_instruction="Return only SQL."),
                assistant_content=fixture["reference_sql"],
                output_kind="direct_sql_control",
            )
        )
    return recovery_rows, control_rows


def summarize_behavior_recovery_finetuning_rows(
    recovery_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_modes = Counter(mode for row in recovery_rows for mode in row["failure_modes"])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "behavior_recovery_row_count": len(recovery_rows),
        "direct_sql_control_row_count": len(control_rows),
        "fixture_ids": [row["fixture_id"] for row in recovery_rows],
        "failure_mode_counts": dict(sorted(failure_modes.items())),
        "leakage_policy": "synthetic_curated_label",
        "comparison_contract": "same_fixture_behavior_recovery_vs_direct_sql_control",
        "evaluation_command": "generated_history_rollout",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_behavior_recovery_finetuning_artifacts(
    *,
    recovery_output_path: Path = DEFAULT_RECOVERY_OUTPUT,
    control_output_path: Path = DEFAULT_CONTROL_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write recovery rows, direct-SQL controls, summary, and manifest."""

    recovery_rows, control_rows = build_behavior_recovery_finetuning_rows()
    summary = summarize_behavior_recovery_finetuning_rows(recovery_rows, control_rows)
    _write_jsonl(recovery_output_path, recovery_rows)
    _write_jsonl(control_output_path, control_rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "behavior_recovery_row_count": len(recovery_rows),
        "direct_sql_control_row_count": len(control_rows),
        "recovery_output_path": str(recovery_output_path),
        "recovery_output_sha256": sha256_file(recovery_output_path),
        "control_output_path": str(control_output_path),
        "control_output_sha256": sha256_file(control_output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "leakage_policy": "synthetic_curated_label",
        "evaluation_command": "generated_history_rollout",
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-output", type=Path, default=DEFAULT_RECOVERY_OUTPUT)
    parser.add_argument("--control-output", type=Path, default=DEFAULT_CONTROL_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    manifest = write_behavior_recovery_finetuning_artifacts(
        recovery_output_path=args.recovery_output,
        control_output_path=args.control_output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(
        "Wrote "
        f"{manifest['behavior_recovery_row_count']} behavior recovery rows and "
        f"{manifest['direct_sql_control_row_count']} direct SQL control rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
