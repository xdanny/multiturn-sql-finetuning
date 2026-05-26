"""Curated synthetic fixtures for multi-turn SQL method gates.

These rows are intentionally small and deterministic. They are not benchmark
results; they are data-engineering fixtures that isolate failure modes the real
CoSQL/BIRD-Interact evaluations need to score separately.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "synthetic_method_fixture_pack"
DEFAULT_OUTPUT = Path("docs/data_artifacts/synthetic_method_fixtures.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/synthetic_method_fixtures_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/synthetic_method_fixtures.manifest.json")
ORACLE_SCORING_FIELDS = (
    "reference_sql",
    "expected_rows",
    "gold_metric_dsl",
    "evaluation_checks",
    "direct_sql_trap",
)

SCHEMA_SQL = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country_code TEXT NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    amount REAL NOT NULL,
    order_date TEXT NOT NULL,
    FOREIGN KEY(customer_id) REFERENCES customers(id)
);

CREATE TABLE campaigns (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE order_promotions (
    order_id INTEGER NOT NULL,
    campaign_id INTEGER NOT NULL,
    source_event TEXT NOT NULL,
    FOREIGN KEY(order_id) REFERENCES orders(id),
    FOREIGN KEY(campaign_id) REFERENCES campaigns(id)
);
"""

SEED_DATA_SQL = """
INSERT INTO customers (id, name, country_code) VALUES
    (1, 'Alice', 'FR'),
    (2, 'Bruno', 'FR'),
    (3, 'Cara', 'US');

INSERT INTO orders (id, customer_id, amount, order_date) VALUES
    (1, 1, 100.0, '2026-01-01'),
    (2, 2, 25.0, '2026-01-02'),
    (3, 3, 200.0, '2026-01-03');

INSERT INTO campaigns (id, name) VALUES
    (1, 'Winter retention'),
    (2, 'VIP expansion');

INSERT INTO order_promotions (order_id, campaign_id, source_event) VALUES
    (1, 1, 'email'),
    (1, 1, 'coupon'),
    (2, 1, 'email'),
    (3, 2, 'email');
"""

SEMANTIC_MODEL = {
    "semantic_model_id": "synthetic_revenue_v1",
    "version": "1",
    "base_table": "orders",
    "entities": {
        "customer": {
            "table": "customers",
            "primary_key": "id",
            "display": "name",
        },
        "campaign": {
            "table": "campaigns",
            "primary_key": "id",
            "display": "name",
        },
    },
    "dimensions": {
        "customer_country": {
            "table": "customers",
            "column": "country_code",
            "sql": "customers.country_code",
            "aliases": {"France": "FR", "United States": "US"},
        },
        "customer_name": {
            "table": "customers",
            "column": "name",
            "sql": "customers.name",
        },
        "campaign_name": {
            "table": "campaigns",
            "column": "name",
            "sql": "campaigns.name",
        },
    },
    "measures": {
        "revenue": {
            "expression": "SUM(orders.amount)",
            "sql": "SUM(orders.amount)",
            "sql_by_policy": {
                "dedupe_bridge_rows": "SUM(DISTINCT orders.amount)",
            },
            "grain": "one row per order",
        }
    },
    "joins": [
        {
            "table": "customers",
            "sql_on": "orders.customer_id = customers.id",
            "required_by": ["customer_country", "customer_name"],
        },
        {
            "table": "order_promotions",
            "sql_on": "order_promotions.order_id = orders.id",
            "required_by": ["campaign_name"],
        },
        {
            "table": "campaigns",
            "sql_on": "order_promotions.campaign_id = campaigns.id",
            "required_by": ["campaign_name"],
        },
    ],
    "fanout_rules": {
        "order_promotions": "dedupe_bridge_rows before summing order revenue",
    },
}


def _execute_rows(sql: str) -> list[list[Any]]:
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executescript(SEED_DATA_SQL)
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [
        [round(value, 8) if isinstance(value, float) else value for value in row]
        for row in rows
    ]


def _base_fixture(
    *,
    fixture_id: str,
    description: str,
    conversation: list[dict[str, Any]],
    reference_sql: str,
    failure_modes: list[str],
    training_targets: list[str],
    required_artifacts: list[str],
    evaluation_checks: dict[str, Any],
    direct_sql_trap: dict[str, Any],
    gold_metric_dsl: str | None = None,
) -> dict[str, Any]:
    expected_rows = _execute_rows(reference_sql)
    wrong_sql = direct_sql_trap.get("wrong_sql")
    if wrong_sql:
        direct_sql_trap = {
            **direct_sql_trap,
            "wrong_rows": _execute_rows(str(wrong_sql)),
        }
    fixture = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "fixture_id": fixture_id,
        "schema_id": "synthetic_revenue_schema_v1",
        "description": description,
        "schema_sql": SCHEMA_SQL.strip(),
        "seed_data_sql": SEED_DATA_SQL.strip(),
        "conversation": conversation,
        "reference_sql": reference_sql,
        "reference_sql_visible_to_model": False,
        "expected_rows": expected_rows,
        "gold_metric_dsl": gold_metric_dsl,
        "semantic_model": SEMANTIC_MODEL,
        "failure_modes": failure_modes,
        "training_targets": training_targets,
        "required_artifacts": required_artifacts,
        "evaluation_checks": evaluation_checks,
        "direct_sql_trap": direct_sql_trap,
        "label_source": "synthetic_curated",
        "oracle_policy": "non_oracle_inputs_only",
        "claim_boundary": (
            "Synthetic fixture only: use it to test a failure mode or training "
            "target before making proxy, BIRD-Interact, or hosted-SOTA claims."
        ),
    }
    fixture["prompt_visible_input"] = prompt_visible_fixture_input(fixture)
    fixture["scoring_contract"] = {
        field: fixture[field]
        for field in ORACLE_SCORING_FIELDS
        if fixture.get(field) is not None
    }
    return fixture


def prompt_visible_fixture_input(fixture: dict[str, Any]) -> dict[str, Any]:
    """Return the model-facing fixture view with scoring/oracle fields removed."""

    return {
        "schema_version": fixture["schema_version"],
        "artifact_type": fixture["artifact_type"],
        "fixture_id": fixture["fixture_id"],
        "schema_id": fixture["schema_id"],
        "description": fixture["description"],
        "schema_sql": fixture["schema_sql"],
        "conversation": fixture["conversation"],
        "semantic_model": fixture["semantic_model"],
        "required_artifacts": fixture["required_artifacts"],
        "oracle_policy": fixture["oracle_policy"],
        "claim_boundary": fixture["claim_boundary"],
    }


def build_synthetic_method_fixtures() -> list[dict[str, Any]]:
    """Return curated fixtures for the method-comparison data contract."""

    value_reference = (
        "SELECT customers.country_code, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country_code = 'FR' GROUP BY customers.country_code"
    )
    entity_reference = (
        "SELECT customers.name, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "WHERE customers.country_code = 'FR' GROUP BY customers.name "
        "ORDER BY revenue DESC LIMIT 1"
    )
    fanout_reference = (
        "SELECT campaigns.name, SUM(deduped.amount) AS revenue "
        "FROM campaigns "
        "JOIN ("
        "  SELECT DISTINCT order_promotions.order_id, order_promotions.campaign_id, orders.amount "
        "  FROM order_promotions JOIN orders ON order_promotions.order_id = orders.id"
        ") AS deduped ON campaigns.id = deduped.campaign_id "
        "GROUP BY campaigns.name ORDER BY revenue DESC"
    )
    fanout_wrong = (
        "SELECT campaigns.name, SUM(orders.amount) AS revenue "
        "FROM campaigns "
        "JOIN order_promotions ON campaigns.id = order_promotions.campaign_id "
        "JOIN orders ON order_promotions.order_id = orders.id "
        "GROUP BY campaigns.name ORDER BY revenue DESC"
    )
    measure_reference = (
        "SELECT customers.country_code, SUM(orders.amount) AS revenue "
        "FROM orders JOIN customers ON orders.customer_id = customers.id "
        "GROUP BY customers.country_code ORDER BY revenue DESC"
    )

    return [
        _base_fixture(
            fixture_id="value_normalization_france",
            description="A follow-up says France while the database stores FR.",
            conversation=[
                {"turn": 1, "user": "Show revenue by country."},
                {"turn": 2, "user": "Only France."},
            ],
            reference_sql=value_reference,
            failure_modes=["value_normalization", "context_carryover"],
            training_targets=["planner_first_sql", "semantic_layer"],
            required_artifacts=["value_index", "entity_resolution_labels"],
            evaluation_checks={
                "requires_value_index": True,
                "requires_context_carryover": True,
                "display_value": "France",
                "storage_value": "FR",
            },
            direct_sql_trap={
                "wrong_sql": value_reference.replace("'FR'", "'France'"),
                "failure": "empty result from display value copied into SQL",
            },
        ),
        _base_fixture(
            fixture_id="entity_resolution_followup",
            description="The user says there, meaning the country established in the prior turn.",
            conversation=[
                {"turn": 1, "user": "Show revenue for France."},
                {"turn": 2, "user": "Which customer there spent the most?"},
            ],
            reference_sql=entity_reference,
            failure_modes=["entity_resolution", "context_carryover", "grain_change"],
            training_targets=["planner_first_sql", "semantic_layer"],
            required_artifacts=["entity_resolution_labels", "value_index"],
            evaluation_checks={
                "requires_prior_turn_reference": True,
                "mention_text": "there",
                "resolved_value": "FR",
                "resolved_grain": "customer_name",
            },
            direct_sql_trap={
                "wrong_sql": entity_reference.replace(
                    "WHERE customers.country_code = 'FR' ",
                    "",
                ),
                "failure": "grain changes but prior country filter is lost",
            },
        ),
        _base_fixture(
            fixture_id="grain_fanout_bridge",
            description="A bridge table duplicates an order under one campaign.",
            conversation=[
                {"turn": 1, "user": "Which campaigns generated the most revenue?"},
            ],
            reference_sql=fanout_reference,
            failure_modes=["grain_fanout", "duplicate_row_policy"],
            training_targets=["planner_first_sql", "metric_dsl", "semantic_layer"],
            required_artifacts=["grain_fanout_fixtures", "semantic_model_manifest"],
            evaluation_checks={
                "duplicate_row_policy": "dedupe_bridge_rows",
                "fanout_path": "campaigns -> order_promotions -> orders",
                "expected_metric_delta": 100.0,
            },
            direct_sql_trap={
                "wrong_sql": fanout_wrong,
                "failure": "bridge duplicates order 1 and inflates Winter retention revenue",
            },
            gold_metric_dsl=(
                "MEASURE(revenue) BY campaign_name "
                "USING duplicate_row_policy='dedupe_bridge_rows'"
            ),
        ),
        _base_fixture(
            fixture_id="measure_preservation_metric",
            description="A metric-heavy request should keep MEASURE(revenue) before SQL expansion.",
            conversation=[
                {"turn": 1, "user": "Show governed revenue by country."},
            ],
            reference_sql=measure_reference,
            failure_modes=["measure_preservation", "semantic_model"],
            training_targets=["metric_dsl", "semantic_layer"],
            required_artifacts=["semantic_model_manifest"],
            evaluation_checks={
                "requires_measure_preservation": True,
                "measure": "revenue",
                "semantic_model_id": SEMANTIC_MODEL["semantic_model_id"],
            },
            direct_sql_trap={
                "wrong_sql": (
                    "SELECT customers.country_code, orders.amount AS revenue "
                    "FROM orders JOIN customers ON orders.customer_id = customers.id"
                ),
                "failure": "expands raw order rows instead of governed revenue metric",
            },
            gold_metric_dsl="MEASURE(revenue) BY customer_country ORDER BY MEASURE(revenue) DESC",
        ),
        _base_fixture(
            fixture_id="recovery_empty_result",
            description="The previous generated SQL returned no rows and needs value repair.",
            conversation=[
                {"turn": 1, "user": "Show revenue for France."},
                {
                    "turn": 2,
                    "assistant_sql": value_reference.replace("'FR'", "'France'"),
                    "observed_previous_rows": [],
                },
                {
                    "turn": 3,
                    "user": "That returned no rows. Repair it and show the top customer there.",
                    "observed_previous_rows": [],
                },
            ],
            reference_sql=entity_reference,
            failure_modes=["recovery", "value_normalization", "entity_resolution"],
            training_targets=["behavior_recovery", "semantic_layer"],
            required_artifacts=["generated_history_trace", "value_index"],
            evaluation_checks={
                "requires_repair_action": "replace_display_value_with_storage_value",
                "requires_generated_history": True,
                "observed_failure": "empty_result",
            },
            direct_sql_trap={
                "wrong_sql": entity_reference.replace("'FR'", "'France'"),
                "failure": "blind retry preserves the value-grounding error",
            },
        ),
    ]


def summarize_synthetic_method_fixtures(fixtures: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize fixture coverage by failure mode, target, and artifact gate."""

    failure_modes: Counter[str] = Counter()
    training_targets: Counter[str] = Counter()
    required_artifacts: Counter[str] = Counter()
    schema_ids: set[str] = set()
    non_oracle_count = 0
    for fixture in fixtures:
        failure_modes.update(fixture["failure_modes"])
        training_targets.update(fixture["training_targets"])
        required_artifacts.update(fixture["required_artifacts"])
        schema_ids.add(str(fixture["schema_id"]))
        visible = prompt_visible_fixture_input(fixture)
        if fixture["oracle_policy"] == "non_oracle_inputs_only" and set(
            ORACLE_SCORING_FIELDS
        ).isdisjoint(visible):
            non_oracle_count += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "fixture_count": len(fixtures),
        "schema_count": len(schema_ids),
        "non_oracle_fixture_count": non_oracle_count,
        "failure_mode_counts": dict(sorted(failure_modes.items())),
        "training_target_counts": dict(sorted(training_targets.items())),
        "required_artifact_counts": dict(sorted(required_artifacts.items())),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_synthetic_method_fixture_artifacts(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write JSONL, summary, and manifest artifacts for the fixture pack."""

    fixtures = build_synthetic_method_fixtures()
    summary = summarize_synthetic_method_fixtures(fixtures)
    _write_jsonl(output_path, fixtures)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "fixture_count": len(fixtures),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "label_source": "synthetic_curated",
        "oracle_policy": "non_oracle_inputs_only",
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

    manifest = write_synthetic_method_fixture_artifacts(
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(f"Wrote {manifest['fixture_count']} synthetic method fixtures to {args.output}")
    print(f"Wrote summary to {args.summary_output}")
    print(f"Wrote manifest to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
