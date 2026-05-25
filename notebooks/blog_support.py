"""Shared data loaders for the blog companion marimo notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


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


def accuracy_scorecard() -> pd.DataFrame:
    strict = read_csv_artifact("plots/vllm_iterations_100turns/summary.csv")
    value = read_csv_artifact("plots/rescored_vllm_semantic_prompt_iteration_100turns/summary.csv")
    rows = [
        {
            "run": "Base Qwen 3.5 9B",
            "mode": "non_oracle_generation",
            "metric": "strict_accuracy",
            "score": float(strict.loc[strict["model_name"] == "unsloth/Qwen3.5-9B", "accuracy"].iloc[0]),
        },
        {
            "run": "100-step LoRA",
            "mode": "non_oracle_generation",
            "metric": "strict_accuracy",
            "score": float(strict.loc[strict["model_name"] == "multiturn-sql-100", "accuracy"].iloc[0]),
        },
        {
            "run": "Best non-oracle prompt",
            "mode": "non_oracle_generation",
            "metric": "value_accuracy",
            "score": float(
                value.loc[
                    (value["model_name"] == "multiturn-sql-semantic-50")
                    & (value["prompt_variant"] == "minimal_executable"),
                    "value_accuracy",
                ].iloc[0]
            ),
        },
        {
            "run": "Oracle prompt ceiling",
            "mode": "oracle_planner_diagnostic",
            "metric": "value_accuracy",
            "score": 0.850,
        },
        {
            "run": "Oracle-trained ceiling",
            "mode": "oracle_planner_diagnostic",
            "metric": "value_accuracy",
            "score": 0.890,
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
    return pd.DataFrame(
        [
            {
                "claim": "Small local model improves on the fixed CoSQL proxy",
                "status": "supported",
                "evidence": "Base strict 0.370 to 100-step LoRA strict 0.530",
            },
            {
                "claim": "Best current non-oracle value score is useful but not production-ready",
                "status": "supported",
                "evidence": "0.640 value accuracy on 100 teacher-forced CoSQL turns",
            },
            {
                "claim": "Schema linking and projection planning are high leverage",
                "status": "supported as diagnostic",
                "evidence": "Oracle-planner diagnostics reach 0.850 to 0.890 value accuracy",
            },
            {
                "claim": "Local 9B beats hosted SOTA on BIRD-Interact",
                "status": "not supported yet",
                "evidence": "No BIRD-Interact or hosted-model baseline in this repo yet",
            },
        ]
    )


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
                "next_gate": "Metric-preserving DSL accuracy before SQL compilation",
            },
            {
                "strategy": "Behavior and recovery tuning",
                "learns": "Clarify, inspect, repair, and recover across turns",
                "main_risk": "Cannot be proven by teacher-forced history alone",
                "next_gate": "Interactive or rollout evaluation with model-generated history",
            },
        ]
    )
