"""Build value-choice consistency recovery rollout inputs."""

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
ARTIFACT_TYPE = "value_choice_consistency_rollout_inputs"
DEFAULT_OUTPUT = Path("docs/data_artifacts/value_choice_consistency_rollout_inputs.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path(
    "docs/data_artifacts/value_choice_consistency_rollout_inputs_summary.json"
)
DEFAULT_MANIFEST_OUTPUT = Path(
    "docs/data_artifacts/value_choice_consistency_rollout_inputs.manifest.json"
)
HISTORY_POLICY = "seeded_generated_failure_then_value_choice_repair"

SYSTEM_PROMPT = (
    "You are a SQL repair assistant for multi-turn analytics. Use the schema, "
    "visible generated-history trace, matched value choices, and allowed columns. "
    "Write only the next SQL query."
)

ALLOWED_COLUMNS = {
    "customers": ["id", "name", "country_code"],
    "orders": ["id", "customer_id", "amount", "order_date"],
}

MATCHED_VALUE_CHOICES = [
    {
        "source_mention": "France",
        "table": "customers",
        "column": "country_code",
        "candidate_storage_values": ["FR"],
        "distractor_storage_values": ["US"],
        "selection_rule": "choose the storage value whose display value matches the visible user mention",
        "source": "database_contents_plus_visible_user_text",
    }
]


def _recovery_fixture() -> dict[str, Any]:
    for fixture in build_synthetic_method_fixtures():
        if fixture["fixture_id"] == "recovery_empty_result":
            return fixture
    raise ValueError("recovery_empty_result fixture not found")


def _schema_context(fixture: dict[str, Any]) -> str:
    visible = fixture["prompt_visible_input"]
    return (
        f"Schema:\n{visible['schema_sql']}\n\n"
        f"Semantic model:\n{json.dumps(visible['semantic_model'], indent=2, sort_keys=True)}\n\n"
        f"Allowed columns:\n{json.dumps(ALLOWED_COLUMNS, indent=2, sort_keys=True)}"
    )


def _first_user_message(fixture: dict[str, Any]) -> str:
    first_turn = fixture["conversation"][0]
    return f"{_schema_context(fixture)}\n\nQuestion:\n{first_turn['user']}"


def _repair_user_message(fixture: dict[str, Any]) -> str:
    repair_turn = fixture["conversation"][-1]
    return (
        f"Observed previous rows: {repair_turn['observed_previous_rows']}\n\n"
        f"Matched value choices:\n{json.dumps(MATCHED_VALUE_CHOICES, indent=2, sort_keys=True)}\n\n"
        f"Question:\n{repair_turn['user']}\n\n"
        "Use the matched storage value for the visible country mention and only valid columns."
    )


def build_value_choice_consistency_rollout_inputs() -> list[dict[str, Any]]:
    """Return dialog-level rollout rows for value-choice consistency testing."""

    fixture = _recovery_fixture()
    seeded_failure_sql = fixture["conversation"][1]["assistant_sql"]
    repair_reference_sql = fixture["reference_sql"]
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "id": fixture["fixture_id"],
            "dialog_id": fixture["fixture_id"],
            "source": "synthetic_value_choice_consistency",
            "database_id": fixture["schema_id"],
            "history_policy": HISTORY_POLICY,
            "evaluation_mode": "non_oracle_generation",
            "uses_oracle_planning_hints": False,
            "semantic_context_pruned_by_oracle_labels": False,
            "seeded_failure_turn_index": 0,
            "seed_failure_sql_visible_to_model": True,
            "repair_reference_sql": repair_reference_sql,
            "repair_reference_sql_visible_to_model": False,
            "failure_modes": fixture["failure_modes"],
            "required_artifacts": [
                "generated_history_trace",
                "matched_value_choices",
                "allowed_columns",
            ],
            "oracle_policy": "seeded_failure_visible_non_oracle_value_choice_context",
            "comparison_contract": "same_dialog_value_choice_consistency_rollout",
            "matched_value_choice_source": "database_contents_plus_visible_user_text",
            "matched_value_choices_visible_to_model": True,
            "expected_value_choice": {
                "table": "customers",
                "column": "country_code",
                "source_mention": "France",
                "storage_value": "FR",
                "visible_to_model": False,
            },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _first_user_message(fixture)},
                {"role": "assistant", "content": seeded_failure_sql},
                {"role": "user", "content": _repair_user_message(fixture)},
                {"role": "assistant", "content": repair_reference_sql},
            ],
        }
    ]


def summarize_value_choice_consistency_rollout_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize value-choice consistency input coverage."""

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
        "oracle_policy": "seeded_failure_visible_non_oracle_value_choice_context",
        "comparison_contract": "same_dialog_value_choice_consistency_rollout",
        "evaluation_gate": "eval.run_behavior_recovery_comparison",
        "value_choice_scoring_gate": "eval.value_choice_consistency",
        "matched_value_choice_source": "database_contents_plus_visible_user_text",
        "matched_value_choices_visible_to_model": True,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_value_choice_consistency_rollout_input_artifacts(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write value-choice consistency inputs, summary, and manifest."""

    rows = build_value_choice_consistency_rollout_inputs()
    summary = summarize_value_choice_consistency_rollout_inputs(rows)
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
        "oracle_policy": "seeded_failure_visible_non_oracle_value_choice_context",
        "evaluation_gate": "eval.run_behavior_recovery_comparison",
        "value_choice_scoring_gate": "eval.value_choice_consistency",
        "matched_value_choice_source": "database_contents_plus_visible_user_text",
        "matched_value_choices_visible_to_model": True,
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

    manifest = write_value_choice_consistency_rollout_input_artifacts(
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(f"Wrote {manifest['rollout_input_count']} value-choice rollout inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
