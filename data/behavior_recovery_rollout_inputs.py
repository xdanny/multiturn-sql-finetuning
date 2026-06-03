"""Build tiny generated-history rollout inputs for behavior recovery."""

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
ARTIFACT_TYPE = "behavior_recovery_rollout_inputs"
DEFAULT_OUTPUT = Path("docs/data_artifacts/behavior_recovery_rollout_inputs.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/behavior_recovery_rollout_inputs_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/behavior_recovery_rollout_inputs.manifest.json")
HISTORY_POLICY = "seeded_generated_failure_then_rollout"

SYSTEM_PROMPT = (
    "You are a SQL repair assistant for multi-turn analytics. Use the schema, "
    "semantic model, visible conversation, previous generated SQL, and observed "
    "rows to write only the next SQL query."
)


def _recovery_fixture() -> dict[str, Any]:
    for fixture in build_synthetic_method_fixtures():
        if fixture["fixture_id"] == "recovery_empty_result":
            return fixture
    raise ValueError("recovery_empty_result fixture not found")


def _schema_context(fixture: dict[str, Any]) -> str:
    visible = fixture["prompt_visible_input"]
    return (
        f"Schema:\n{visible['schema_sql']}\n\n"
        f"Semantic model:\n{json.dumps(visible['semantic_model'], indent=2, sort_keys=True)}"
    )


def _first_user_message(fixture: dict[str, Any]) -> str:
    first_turn = fixture["conversation"][0]
    return f"{_schema_context(fixture)}\n\nQuestion:\n{first_turn['user']}"


def _repair_user_message(fixture: dict[str, Any]) -> str:
    repair_turn = fixture["conversation"][-1]
    return (
        f"Observed previous rows: {repair_turn['observed_previous_rows']}\n\n"
        f"Question:\n{repair_turn['user']}"
    )


def build_behavior_recovery_rollout_inputs() -> list[dict[str, Any]]:
    """Return dialog-level rollout rows for the generated-history recovery gate."""

    fixture = _recovery_fixture()
    seeded_failure_sql = fixture["conversation"][1]["assistant_sql"]
    repair_reference_sql = fixture["reference_sql"]
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "id": fixture["fixture_id"],
            "dialog_id": fixture["fixture_id"],
            "source": "synthetic_behavior_recovery",
            "database_id": fixture["schema_id"],
            "history_policy": HISTORY_POLICY,
            "evaluation_mode": "non_oracle_generation",
            "seeded_failure_turn_index": 0,
            "seed_failure_sql_visible_to_model": True,
            "repair_reference_sql": repair_reference_sql,
            "repair_reference_sql_visible_to_model": False,
            "failure_modes": fixture["failure_modes"],
            "required_artifacts": fixture["required_artifacts"],
            "leakage_policy": "seeded_generated_failure_with_repair_label_held_out",
            "comparison_contract": "same_dialog_rollout_vs_teacher_forced_history",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _first_user_message(fixture)},
                {"role": "assistant", "content": seeded_failure_sql},
                {"role": "user", "content": _repair_user_message(fixture)},
                {"role": "assistant", "content": repair_reference_sql},
            ],
        }
    ]


def summarize_behavior_recovery_rollout_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize behavior-recovery rollout input coverage."""

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "rollout_input_count": len(rows),
        "fixture_ids": [row["id"] for row in rows],
        "history_policy_counts": dict(
            sorted(Counter(row["history_policy"] for row in rows).items())
        ),
        "failure_mode_counts": dict(
            sorted(Counter(mode for row in rows for mode in row["failure_modes"]).items())
        ),
        "leakage_policy": "seeded_generated_failure_with_repair_label_held_out",
        "comparison_contract": "same_dialog_rollout_vs_teacher_forced_history",
        "evaluation_command": "eval.rollout_eval then eval.compare_rollout_history",
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_behavior_recovery_rollout_input_artifacts(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write behavior-recovery rollout inputs, summary, and manifest."""

    rows = build_behavior_recovery_rollout_inputs()
    summary = summarize_behavior_recovery_rollout_inputs(rows)
    _write_jsonl(output_path, rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "rollout_input_count": len(rows),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "leakage_policy": "seeded_generated_failure_with_repair_label_held_out",
        "evaluation_command": "eval.rollout_eval then eval.compare_rollout_history",
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    manifest = write_behavior_recovery_rollout_input_artifacts(
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(f"Wrote {manifest['rollout_input_count']} behavior recovery rollout inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
