"""Shared data loaders for the blog companion marimo notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.metric_dsl import compile_metric_query, parse_metric_query, score_metric_query


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifact_path(relative_path: str) -> Path:
    return repo_root() / relative_path


def read_csv_artifact(relative_path: str) -> pd.DataFrame:
    path = artifact_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {relative_path}")
    return pd.read_csv(path)


def read_json_artifact(relative_path: str) -> dict[str, Any]:
    path = artifact_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {relative_path}")
    return json.loads(path.read_text())


def read_jsonl_artifact(relative_path: str, limit: int | None = None) -> list[dict[str, Any]]:
    path = artifact_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact: {relative_path}")
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def claim_ledger() -> pd.DataFrame:
    return pd.DataFrame(read_jsonl_artifact("docs/claim_ledgers/cosql_dev_100.jsonl"))


def accuracy_scorecard() -> pd.DataFrame:
    ledger = claim_ledger().set_index("claim_id")

    def score(run_id: str, metric: str) -> float:
        value = ledger.loc[run_id, metric]
        if pd.isna(value):
            raise ValueError(f"Missing {metric} for {run_id}")
        return float(value)

    rows = [
        {
            "run": "Base Qwen 3.5 9B",
            "mode": "non_oracle_generation",
            "metric": "strict_accuracy",
            "score": score("qwen35_9b_base_cosql_dev_100turns", "strict_execution_accuracy"),
        },
        {
            "run": "100-step LoRA",
            "mode": "non_oracle_generation",
            "metric": "strict_accuracy",
            "score": score("multiturn_sql_100_cosql_dev_100turns", "strict_execution_accuracy"),
        },
        {
            "run": "Best non-oracle prompt",
            "mode": "non_oracle_generation",
            "metric": "value_accuracy",
            "score": score(
                "semantic_prompt_minimal_executable_cosql_dev_100turns",
                "value_execution_accuracy",
            ),
        },
        {
            "run": "Oracle prompt ceiling",
            "mode": "oracle_planner_diagnostic",
            "metric": "value_accuracy",
            "score": score(
                "schema_pruned_minimal_schemafix_oracle_cosql_dev_100turns",
                "value_execution_accuracy",
            ),
        },
        {
            "run": "Oracle-trained ceiling",
            "mode": "oracle_planner_diagnostic",
            "metric": "value_accuracy",
            "score": score(
                "schema_pruned_trained100_oracle_cosql_dev_100turns",
                "value_execution_accuracy",
            ),
        },
    ]
    return pd.DataFrame(rows)


def planner_scorecard() -> pd.DataFrame:
    summary = read_json_artifact("docs/planner_baseline_cosql_dev_100_summary.json")
    fields = [
        "macro_planner_score",
        "table_f1",
        "column_f1",
        "join_f1",
        "skeleton_f1",
        "group_by_f1",
        "selected_count_match",
        "duplicate_policy_match",
    ]
    return pd.DataFrame(
        [{"metric": field, "score": float(summary[field])} for field in fields]
    )


def claim_table() -> pd.DataFrame:
    ledger = claim_ledger()
    selected = ledger[
        ledger["claim_id"].isin(
            [
                "qwen35_9b_base_cosql_dev_100turns",
                "multiturn_sql_100_cosql_dev_100turns",
                "semantic_prompt_minimal_executable_cosql_dev_100turns",
                "schema_pruned_trained100_oracle_cosql_dev_100turns",
                "model_generated_history_rollout",
                "rollout_beats_teacher_forced_history",
                "hosted_sota_same_protocol",
                "bird_interact_local_vs_hosted",
            ]
        )
    ].copy()
    selected["evidence"] = selected.apply(
        lambda row: row["blocking_reason"]
        if row["claim_status"] == "pending"
        else (
            f"value={row['value_execution_accuracy']}, "
            f"strict={row['strict_execution_accuracy']}, rows={row['row_count']}"
        ),
        axis=1,
    )
    return selected[
        [
            "claim_id",
            "claim_status",
            "evaluation_mode",
            "allowed_public_claim",
            "evidence",
        ]
    ].reset_index(drop=True)


def semantic_strategy_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "strategy": "Direct SQL SFT",
                "learns": "SQL syntax, schema names, conversation formatting",
                "main_risk": "Memorizes query surface without stable planning",
                "next_gate": "Non-oracle CoSQL and BIRD-Interact execution accuracy",
            },
            {
                "strategy": "Planner or DSL first, SQL second",
                "learns": "Tables, columns, joins, grain, filters, projections before SQL",
                "main_risk": "Planner can become another vague prompt unless scored directly",
                "next_gate": "Planner F1 plus predicted-plan SQL execution",
            },
            {
                "strategy": "Semantic layer / MEASURE() preservation",
                "learns": "Governed measures, dimensions, grain, and metric definitions",
                "main_risk": "Dumping all semantic context increases latency and noise",
                "next_gate": "Metric-DSL manifest with semantic intent and compiled-SQL scores",
            },
            {
                "strategy": "Behavior and recovery tuning",
                "learns": "Clarify, inspect, repair, and recover across turns",
                "main_risk": "Cannot be proven by teacher-forced history alone",
                "next_gate": "Interactive or rollout evaluation with model-generated history",
            },
        ]
    )


def metric_dsl_demo() -> dict[str, Any]:
    semantic_model = {
        "base_table": "orders",
        "measures": {
            "revenue": {"sql": "SUM(orders.amount)"},
            "orders_count": {"sql": "COUNT(DISTINCT orders.id)"},
        },
        "dimensions": {
            "customer_country": {"sql": "customers.country"},
            "order_month": {"sql": "strftime('%Y-%m', orders.created_at)"},
        },
        "joins": [
            {
                "table": "customers",
                "sql_on": "orders.customer_id = customers.id",
                "required_by": ["customer_country"],
            }
        ],
    }
    gold = parse_metric_query(
        "MEASURE(revenue) BY customer_country WHERE customer_country = 'FR' "
        "ORDER BY MEASURE(revenue) DESC LIMIT 5"
    )
    raw_sql_like = parse_metric_query("SUM(orders.amount) BY customer_country")
    return {
        "semantic_model": semantic_model,
        "gold_query": gold,
        "compiled_sql": compile_metric_query(gold, semantic_model),
        "raw_sql_like_score": score_metric_query(raw_sql_like, gold),
    }


def metric_dsl_eval_contract() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "metric": "metric_dsl_parse_rate",
                "why_it_matters": "The generated artifact can be parsed as the DSL target.",
                "status": "pending_manifest",
            },
            {
                "metric": "metric_dsl_compile_rate",
                "why_it_matters": "The parsed DSL can compile through a governed semantic model.",
                "status": "pending_manifest",
            },
            {
                "metric": "measure_preservation",
                "why_it_matters": "The model kept MEASURE(name) instead of expanding metric SQL early.",
                "status": "pending_manifest",
            },
            {
                "metric": "value_execution_accuracy",
                "why_it_matters": "The compiled SQL returns the right values when reference SQL and a database are available.",
                "status": "pending_manifest",
            },
            {
                "metric": "compiled_sql_execution_evaluated_rows",
                "why_it_matters": "Execution accuracy is meaningful only over database-backed attempts.",
                "status": "pending_manifest",
            },
            {
                "metric": "semantic_model_sha256s",
                "why_it_matters": "Metric results must identify the semantic model used for compilation.",
                "status": "pending_manifest",
            },
        ]
    )
