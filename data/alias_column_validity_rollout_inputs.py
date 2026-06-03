"""Build alias/column-validity recovery rollout inputs."""

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
ARTIFACT_TYPE = "alias_column_validity_rollout_inputs"
DEFAULT_OUTPUT = Path("docs/data_artifacts/alias_column_validity_rollout_inputs.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/alias_column_validity_rollout_inputs_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/alias_column_validity_rollout_inputs.manifest.json")
HISTORY_POLICY = "seeded_generated_failure_then_alias_column_repair"

SYSTEM_PROMPT = (
    "You are a SQL repair assistant for multi-turn analytics. Use the schema, "
    "visible generated-history trace, matched value choices, and column-role "
    "constraints. Write only the next SQL query."
)

COLUMN_ROLE_CONSTRAINTS = {
    "artifact_type": "synthetic_column_role_constraints",
    "source": "schema_introspection_plus_visible_failure",
    "allowed_columns": {
        "customers": ["id", "name", "country_code"],
        "orders": ["id", "customer_id", "amount", "order_date"],
    },
    "join_keys": ["orders.customer_id = customers.id"],
    "projection_rules": [
        "To show the top customer, select customers.name.",
        "When selecting customers.name with SUM(orders.amount), group by customers.name.",
    ],
    "invalid_patterns_to_avoid": [
        "customers.customer_id",
    ],
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
        f"Semantic model:\n{json.dumps(visible['semantic_model'], indent=2, sort_keys=True)}"
    )


def _first_user_message(fixture: dict[str, Any]) -> str:
    first_turn = fixture["conversation"][0]
    return f"{_schema_context(fixture)}\n\nQuestion:\n{first_turn['user']}"


def _repair_user_message(fixture: dict[str, Any]) -> str:
    repair_turn = fixture["conversation"][-1]
    return (
        f"Observed previous rows: {repair_turn['observed_previous_rows']}\n\n"
        f"Matched value choices:\n{json.dumps(MATCHED_VALUE_CHOICES, indent=2, sort_keys=True)}\n\n"
        f"Column-role constraints:\n"
        f"{json.dumps(COLUMN_ROLE_CONSTRAINTS, indent=2, sort_keys=True)}\n\n"
        f"Question:\n{repair_turn['user']}\n\n"
        "Use the matched storage value and only table.column references allowed by the constraints."
    )


def build_alias_column_validity_rollout_inputs() -> list[dict[str, Any]]:
    """Return dialog-level rollout rows for alias/column-validity testing."""

    fixture = _recovery_fixture()
    seeded_failure_sql = fixture["conversation"][1]["assistant_sql"]
    repair_reference_sql = fixture["reference_sql"]
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "id": fixture["fixture_id"],
            "dialog_id": fixture["fixture_id"],
            "source": "synthetic_alias_column_validity",
            "database_id": fixture["schema_id"],
            "history_policy": HISTORY_POLICY,
            "evaluation_mode": "non_oracle_generation",
            "seeded_failure_turn_index": 0,
            "seed_failure_sql_visible_to_model": True,
            "repair_reference_sql": repair_reference_sql,
            "repair_reference_sql_visible_to_model": False,
            "failure_modes": fixture["failure_modes"],
            "required_artifacts": [
                "generated_history_trace",
                "matched_value_choices",
                "column_role_constraints",
            ],
            "leakage_policy": "seeded_failure_visible_alias_column_context",
            "comparison_contract": "same_dialog_alias_column_validity_rollout",
            "column_role_constraints_visible_to_model": True,
            "column_role_constraints_source": "schema_introspection_plus_visible_failure",
            "expected_column_validity": {
                "invalid_table_column_refs": [],
                "invalid_patterns_to_avoid": ["customers.customer_id"],
                "visible_to_model": False,
            },
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


def summarize_alias_column_validity_rollout_inputs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize alias/column-validity input coverage."""

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
        "leakage_policy": "seeded_failure_visible_alias_column_context",
        "comparison_contract": "same_dialog_alias_column_validity_rollout",
        "evaluation_command": "eval.run_behavior_recovery_comparison",
        "column_validity_scoring_command": "eval.alias_column_validity",
        "column_role_constraints_source": "schema_introspection_plus_visible_failure",
        "column_role_constraints_visible_to_model": True,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_alias_column_validity_rollout_input_artifacts(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write alias/column-validity inputs, summary, and manifest."""

    rows = build_alias_column_validity_rollout_inputs()
    summary = summarize_alias_column_validity_rollout_inputs(rows)
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
        "leakage_policy": "seeded_failure_visible_alias_column_context",
        "evaluation_command": "eval.run_behavior_recovery_comparison",
        "column_validity_scoring_command": "eval.alias_column_validity",
        "column_role_constraints_source": "schema_introspection_plus_visible_failure",
        "column_role_constraints_visible_to_model": True,
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

    manifest = write_alias_column_validity_rollout_input_artifacts(
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(f"Wrote {manifest['rollout_input_count']} alias/column-validity rollout inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
