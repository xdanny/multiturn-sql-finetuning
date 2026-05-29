"""Shared data loaders for the blog companion lab and generated evidence."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from contextlib import suppress
from pathlib import Path
from typing import Any

import pandas as pd

from data.metric_dsl import compile_metric_query, parse_metric_query, score_metric_query
from eval.classify_errors import validate_sql_against_visible_schema
from eval.method_readiness import build_method_readiness
from notebooks.labs.local_multiturn_sql_lab_support import run_multiturn_lab
from train.finetuning_steps import finetuning_step_summary

BLOG_EVIDENCE_SOURCES = (
    "configs/benchmark_protocols.yaml",
    "configs/finetuning_methods.yaml",
    "configs/finetuning_steps.yaml",
    "docs/claim_ledgers/cosql_dev_100.jsonl",
    "docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl",
    "docs/data_artifacts/value_grounding_labels_cosql_dev_100.manifest.json",
    "docs/data_artifacts/value_grounding_labels_cosql_dev_100_summary.json",
    "docs/data_artifacts/value_index_cosql_dev_100.jsonl",
    "docs/data_artifacts/value_index_cosql_dev_100.manifest.json",
    "docs/data_artifacts/value_index_cosql_dev_100_summary.json",
    "docs/data_artifacts/synthetic_method_fixtures.jsonl",
    "docs/data_artifacts/synthetic_method_fixtures.manifest.json",
    "docs/data_artifacts/synthetic_method_fixtures_summary.json",
    "docs/planner_baseline_cosql_dev_100_summary.json",
    "docs/predicted_planner_comparison_preflight.json",
    "docs/planner_readiness_cosql_dev_100.json",
    "plots/rescored_vllm_semantic_prompt_iteration_100turns/summary.csv",
    "plots/failure_taxonomy/comparison/model_error_summary.csv",
    "plots/failure_taxonomy/comparison/pairwise_vs_baseline.csv",
    "results/prompt_search_semantic50_limit30/summary.csv",
    "results/prompt_search_schema_pruned_projection_schemafix_100/summary.csv",
)

BLOG_POST_SLUG = "local-multiturn-sql-finetuning"
FORBIDDEN_PUBLIC_SUBSTRINGS = (
    "notebooks/blog/",
    "notebook-contracts.md",
    "notebook-series.md",
    "notebook-walkthrough.md",
)
ASSET_CLAIM_IDS = {
    "accuracy_ladder_svg": (
        "qwen35_9b_base_cosql_dev_100turns",
        "multiturn_sql_100_cosql_dev_100turns",
        "semantic_prompt_minimal_executable_cosql_dev_100turns",
        "schema_pruned_minimal_schemafix_oracle_cosql_dev_100turns",
        "schema_pruned_trained100_oracle_cosql_dev_100turns",
    ),
    "planner_baseline_svg": ("planner_lexical_schema_baseline",),
    "claim_table_md": (
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
    ),
    "metric_dsl_contract_md": (
        "metric-dsl-bootstrap.metric_dsl",
        "metric_dsl_beats_direct_sql",
    ),
    "method_decision_rules_md": (
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "metric_dsl_beats_direct_sql",
        "rollout_beats_teacher_forced_history",
    ),
    "method_priority_backlog_md": (
        "predicted_planner_sql_execution",
        "metric_dsl_beats_direct_sql",
        "rollout_beats_teacher_forced_history",
    ),
    "data_engineering_gates_md": (
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
        "metric-dsl-bootstrap.metric_dsl",
        "rollout_beats_teacher_forced_history",
    ),
    "target_comparison_md": (
        "predicted_planner_sql_execution",
        "metric_dsl_beats_direct_sql",
        "rollout_beats_teacher_forced_history",
    ),
    "target_evidence_matrix_md": (
        "predicted_planner_sql_execution",
        "metric_dsl_beats_direct_sql",
        "hosted_sota_same_protocol",
    ),
    "experiment_ladder_md": (
        "predicted_planner_sql_execution",
        "metric-dsl-bootstrap.metric_dsl",
        "metric_dsl_beats_direct_sql",
        "rollout_beats_teacher_forced_history",
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
    ),
    "single_to_multiturn_gap_md": (
        "qwen35_9b_base_cosql_dev_100turns",
        "semantic_prompt_minimal_executable_cosql_dev_100turns",
        "rollout_beats_teacher_forced_history",
        "metric-dsl-bootstrap.metric_dsl",
        "bird_interact_local_vs_hosted",
    ),
    "hosted_comparison_protocol_md": (
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
        "rollout_beats_teacher_forced_history",
    ),
    "evaluation_harness_map_md": (
        "predicted_planner_sql_execution",
        "metric-dsl-bootstrap.metric_dsl",
        "metric_dsl_beats_direct_sql",
        "rollout_beats_teacher_forced_history",
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
    ),
    "method_readiness_report_md": (
        "predicted_planner_sql_execution",
        "metric-dsl-bootstrap.metric_dsl",
        "metric_dsl_beats_direct_sql",
        "rollout_beats_teacher_forced_history",
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
    ),
    "finetuning_step_plan_md": (
        "predicted_planner_sql_execution",
        "semantic_value_retrieval_improves_sql",
        "metric_dsl_beats_direct_sql",
        "behavior_recovery_beats_direct_sql",
        "hosted_sota_same_protocol",
        "local_beats_hosted_same_protocol",
        "bird_interact_local_vs_hosted",
    ),
}
ASSET_REQUIRED_IN_POST = {
    "accuracy_ladder_svg",
    "planner_baseline_svg",
}
ASSET_REQUIRED_REFERENCE = {
    "claim_table_md",
    "dataset_role_matrix_md",
    "metric_dsl_contract_md",
    "method_decision_rules_md",
    "method_priority_backlog_md",
    "data_engineering_gates_md",
    "experiment_ladder_md",
    "evaluation_harness_map_md",
    "method_readiness_report_md",
    "finetuning_step_plan_md",
    "hosted_comparison_protocol_md",
    "single_to_multiturn_gap_md",
    "synthetic_method_fixtures_md",
    "value_grounding_labels_md",
    "value_index_md",
    "data_artifact_contract_md",
    "prompt_optimization_findings_md",
    "target_comparison_md",
    "target_evidence_matrix_md",
    "endpoint_run_scorecard_md",
    "failure_taxonomy_delta_md",
    "schema_validation_findings_md",
    "planner_readiness_md",
}


LAB_NOTEBOOK = "notebooks/labs/local_multiturn_sql_lab.ipynb"
LAB_APP = "notebooks/labs/local_multiturn_sql_lab.py"
LAB_HTML_URL = "/labs/local-multiturn-sql-finetuning/"

LAB_READER_SECTIONS: tuple[dict[str, str], ...] = (
    {
        "lab_step": "Benchmark gap",
        "reader_action": (
            "Research question: start in the published HTML lab with the "
            "failure trace: single-turn SQL can look solved while a follow-up "
            "loses state."
        ),
        "evidence_to_inspect": "lab-failure-trace.md",
        "purpose": (
            "Turns the BIRD-style zero-shot premise into a concrete multi-turn "
            "failure before introducing any fine-tuning result."
        ),
        "claim_boundary": (
            "Published lab demonstration only; it is not a benchmark result and "
            "does not compare model leaderboard scores."
        ),
    },
    {
        "lab_step": "Evaluation protocol",
        "reader_action": (
            "Inspect the protocol before reading scores: CoSQL/SParC/synthetic/"
            "BIRD roles, teacher-forced history, oracle boundaries, and hosted "
            "baseline gates."
        ),
        "evidence_to_inspect": "claim-table.md and data-engineering-gates.md",
        "purpose": "Defines what a claim can mean before the post shows accuracy numbers.",
        "claim_boundary": (
            "CoSQL is the current proxy; hosted SOTA and BIRD-Interact claims stay "
            "pending until same-protocol manifests exist."
        ),
    },
    {
        "lab_step": "Fine-tuning targets",
        "reader_action": (
            "Compare direct SQL, planner-first SQL, semantic-layer grounding, "
            "MEASURE()-preserving DSL, and recovery as separate training hypotheses."
        ),
        "evidence_to_inspect": "lab-method-scores.md and target-evidence-matrix.md",
        "purpose": (
            "Keeps the post from treating LoRA movement as the whole result; each "
            "target has a falsifiable next gate."
        ),
        "claim_boundary": (
            "The lab isolates behaviors; the target matrix says which rows are "
            "manifest-backed, pending, or only diagnostic."
        ),
    },
    {
        "lab_step": "Results and diagnostics",
        "reader_action": (
            "Inspect strict vs value scoring, direct-SQL control results, oracle "
            "ceiling, and the non-oracle planner baseline."
        ),
        "evidence_to_inspect": "endpoint-run-scorecard.md, accuracy-ladder.svg, and planner-baseline.svg",
        "purpose": (
            "Shows why the direct-SQL control moved, why the scorer changed, and "
            "why oracle planning is a ceiling rather than a production result."
        ),
        "claim_boundary": (
            "The best current number is a non-oracle CoSQL proxy with "
            "teacher-forced history, not a hosted-SOTA result."
        ),
    },
    {
        "lab_step": "Next experiments",
        "reader_action": (
            "Turn the remaining misses into artifacts: predicted planner execution, "
            "metric-DSL comparison, rollout history, hosted baselines, and "
            "BIRD-Interact transfer."
        ),
        "evidence_to_inspect": "target-comparison.md, prompt-optimization-findings.md, and data-engineering-gates.md",
        "purpose": "Turns the post ending into a decision table instead of a generic roadmap.",
        "claim_boundary": (
            "A target becomes best only after it beats direct SQL on identical rows "
            "under the same scorer and protocol."
        ),
    },
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def artifact_path(relative_path: str) -> Path:
    return repo_root() / relative_path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def value_grounding_label_summary() -> pd.DataFrame:
    summary = read_json_artifact(
        "docs/data_artifacts/value_grounding_labels_cosql_dev_100_summary.json"
    )
    interpretations = {
        "value_reference_count": (
            "Gold SQL literal predicates turned into value-grounding labels on the "
            "fixed CoSQL proxy slice."
        ),
        "exact_in_current_turn_count": (
            "The user stated the stored value directly in the current turn."
        ),
        "exact_in_history_count": (
            "The current SQL depends on a value mentioned in an earlier user turn."
        ),
        "carried_from_prior_sql_count": (
            "The value is inherited from a prior assistant SQL turn rather than "
            "the current user text."
        ),
        "missing_from_user_text_count": (
            "The stored literal is not present in user text and needs a value index "
            "or display-to-storage normalization."
        ),
        "requires_context_carryover_count": (
            "Rows where the value-grounding target depends on previous turns."
        ),
        "requires_value_normalization_count": (
            "Rows where exact text matching is not enough to recover the stored value."
        ),
        "database_count": "Number of databases touched by the value-label artifact.",
        "dialog_turn_count": "Number of dialog turns with at least one value predicate.",
    }
    rows = [
        {
            "artifact": "value_grounding_labels_cosql_dev_100.jsonl",
            "metric": metric,
            "value": int(value),
            "interpretation": interpretations.get(metric, "artifact summary metric"),
        }
        for metric, value in summary.items()
    ]
    return pd.DataFrame(rows)


def value_index_summary() -> pd.DataFrame:
    summary = read_json_artifact("docs/data_artifacts/value_index_cosql_dev_100_summary.json")
    coverage = summary.get("coverage") or {}
    metrics = {
        "artifact_type": summary.get("artifact_type"),
        "index_source": summary.get("index_source"),
        "entry_count": summary.get("entry_count"),
        "database_count": summary.get("database_count"),
        "table_count": summary.get("table_count"),
        "column_count": summary.get("column_count"),
        "alias_count": summary.get("alias_count"),
        "resolved_value_indexed_rate": coverage.get("resolved_value_indexed_rate"),
        "mention_alias_indexed_rate": coverage.get("mention_alias_indexed_rate"),
        "resolved_value_indexed_count": coverage.get("resolved_value_indexed_count"),
        "mention_alias_indexed_count": coverage.get("mention_alias_indexed_count"),
        "label_count": coverage.get("label_count"),
    }
    interpretations = {
        "artifact_type": "Summary type for the generated value-index artifact.",
        "index_source": "Where index entries come from; database contents means no reference SQL.",
        "entry_count": "Number of distinct database values indexed under the per-column cap.",
        "database_count": "Number of CoSQL databases scanned from the fixed proxy input.",
        "table_count": "Number of tables with indexed values.",
        "column_count": "Number of columns with indexed values.",
        "alias_count": "Raw and normalized aliases available before entity-resolution enrichment.",
        "resolved_value_indexed_rate": (
            "Share of gold SQL literal values present in the database-derived index."
        ),
        "mention_alias_indexed_rate": (
            "Share of user-visible mentions already recoverable as value-index aliases."
        ),
        "resolved_value_indexed_count": "Count of gold resolved values found in the index.",
        "mention_alias_indexed_count": "Count of user mentions found as index aliases.",
        "label_count": "Gold value labels used only for coverage evaluation.",
    }
    return pd.DataFrame(
        [
            {
                "artifact": "value_index_cosql_dev_100_summary.json",
                "metric": metric,
                "value": value,
                "interpretation": interpretations[metric],
            }
            for metric, value in metrics.items()
        ]
    )


def synthetic_method_fixture_summary() -> pd.DataFrame:
    summary = read_json_artifact("docs/data_artifacts/synthetic_method_fixtures_summary.json")
    rows = [
        {
            "artifact": "synthetic_method_fixtures_summary.json",
            "metric": "fixture_count",
            "value": int(summary["fixture_count"]),
            "interpretation": (
                "Curated synthetic rows that isolate method-specific multi-turn "
                "failures before endpoint spend."
            ),
        },
        {
            "artifact": "synthetic_method_fixtures_summary.json",
            "metric": "schema_count",
            "value": int(summary["schema_count"]),
            "interpretation": "Number of deterministic SQLite schemas in the fixture pack.",
        },
        {
            "artifact": "synthetic_method_fixtures_summary.json",
            "metric": "non_oracle_fixture_count",
            "value": int(summary["non_oracle_fixture_count"]),
            "interpretation": "Rows designed so reference SQL is used for scoring, not prompt context.",
        },
        {
            "artifact": "synthetic_method_fixtures_summary.json",
            "metric": "failure_modes",
            "value": ", ".join(sorted((summary.get("failure_mode_counts") or {}).keys())),
            "interpretation": (
                "Failure classes covered by synthetic rows: values, entity "
                "resolution, grain/fanout, MEASURE() preservation, and recovery."
            ),
        },
        {
            "artifact": "synthetic_method_fixtures_summary.json",
            "metric": "training_targets",
            "value": ", ".join(sorted((summary.get("training_target_counts") or {}).keys())),
            "interpretation": (
                "Candidate tuning targets exercised before larger CoSQL or "
                "BIRD-Interact endpoint runs."
            ),
        },
        {
            "artifact": "synthetic_method_fixtures_summary.json",
            "metric": "required_artifacts",
            "value": ", ".join(sorted((summary.get("required_artifact_counts") or {}).keys())),
            "interpretation": "Data artifacts the fixtures are meant to validate or supervise.",
        },
    ]
    return pd.DataFrame(rows)


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


def planner_readiness_summary() -> pd.DataFrame:
    summary = read_json_artifact("docs/planner_readiness_cosql_dev_100.json")
    interpretations = {
        "endpoint_pair_ready": (
            "Whether direct and predicted prepared inputs have matching row identities "
            "before endpoint execution."
        ),
        "claim_boundary": "What this artifact is allowed to prove.",
        "row_count": "Number of planner-evaluated turns in the readiness report.",
        "dialog_count": "Number of dialogs covered by the planner readiness report.",
        "database_count": "Number of databases covered by the planner readiness report.",
        "column_zero_rate": "Share of turns where the predicted planner recovered no gold columns.",
        "selected_count_mismatch_rate": (
            "Share of turns where predicted projection width does not match the reference plan."
        ),
        "empty_projection_expression_rate": (
            "Share of turns where the planner produced no explicit projection expressions."
        ),
        "join_zero_when_gold_join_rate": (
            "Share of all turns where a gold join exists and the predicted join score is zero."
        ),
        "group_by_zero_when_gold_group_by_rate": (
            "Share of all turns where gold grouping exists and predicted group-by score is zero."
        ),
        "recommendation": "Decision for the next planner step before making SQL claims.",
        "top_risks": "Largest planner weaknesses to address before endpoint claims.",
    }
    metrics = [
        "endpoint_pair_ready",
        "claim_boundary",
        "row_count",
        "dialog_count",
        "database_count",
        "column_zero_rate",
        "selected_count_mismatch_rate",
        "empty_projection_expression_rate",
        "join_zero_when_gold_join_rate",
        "group_by_zero_when_gold_group_by_rate",
        "recommendation",
        "top_risks",
    ]
    rows = []
    for metric in metrics:
        value = summary.get(metric)
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        rows.append(
            {
                "artifact": "planner_readiness_cosql_dev_100.json",
                "metric": metric,
                "value": value,
                "interpretation": interpretations[metric],
            }
        )
    return pd.DataFrame(rows)


def claim_table() -> pd.DataFrame:
    ledger = claim_ledger()
    selected = ledger[
        ledger["claim_id"].isin(
            [
                "qwen35_9b_base_cosql_dev_100turns",
                "multiturn_sql_100_cosql_dev_100turns",
                "semantic_prompt_minimal_executable_cosql_dev_100turns",
                "value_index_coverage",
                "semantic_value_retrieval_improves_sql",
                "schema_pruned_trained100_oracle_cosql_dev_100turns",
                "metric-dsl-bootstrap.metric_dsl",
                "metric_dsl_beats_direct_sql",
                "model_generated_history_rollout",
                "rollout_beats_teacher_forced_history",
                "hosted_sota_same_protocol",
                "local_beats_hosted_same_protocol",
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
                "next_gate": "Metric-DSL manifest plus direct-SQL comparison on matching rows",
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
                "metric": "metric_dsl_value_delta_vs_direct_sql",
                "why_it_matters": "The DSL-first path must beat a direct-SQL baseline before it supports a superiority claim.",
                "status": "pending_comparison",
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


def dataset_role_matrix() -> pd.DataFrame:
    synthetic_summary = read_json_artifact(
        "docs/data_artifacts/synthetic_method_fixtures_summary.json"
    )
    return pd.DataFrame(
        [
            {
                "data_source": "BIRD-Interact",
                "project_role": "north-star interactive benchmark for the final hosted-SOTA comparison",
                "what_it_tests": (
                    "multi-turn data analysis over richer schemas, values, and "
                    "interaction state"
                ),
                "why_single_turn_is_not_enough": (
                    "The agent must carry intent, inspect results, and recover across "
                    "turns rather than solve one isolated SQL request."
                ),
                "current_status": "pending transfer target; no local-vs-hosted manifest yet",
                "next_artifact": (
                    "BIRD-Interact transfer manifest with local model, hosted baselines, "
                    "cost, latency, and identical scorer settings"
                ),
            },
            {
                "data_source": "BIRD mini-dev",
                "project_role": "single-turn execution-harness check, not a dialogue claim",
                "what_it_tests": "schema linking and SQL execution on BIRD-style databases",
                "why_single_turn_is_not_enough": (
                    "It can validate the SQL runner, but it cannot expose follow-up "
                    "state, generated-history drift, or recovery behavior."
                ),
                "current_status": "useful harness input; not promoted as multi-turn evidence",
                "next_artifact": "same-runner bridge into BIRD-Interact-style turns",
            },
            {
                "data_source": "CoSQL",
                "project_role": "current fixed proxy slice for fast multi-turn iteration",
                "what_it_tests": (
                    "dialogue-context SQL over Spider-style SQLite databases with "
                    "teacher-forced history"
                ),
                "why_single_turn_is_not_enough": (
                    "The same schema can require filter carryover, value grounding, "
                    "projection stability, and history alignment across turns."
                ),
                "current_status": "fixed 100-turn proxy used by the claim ledger",
                "next_artifact": (
                    "generated-history rollout plus predicted-planner SQL comparison "
                    "on the same CoSQL rows"
                ),
            },
            {
                "data_source": "SParC",
                "project_role": "context-dependent SQL training/evaluation support",
                "what_it_tests": (
                    "context-dependent follow-up SQL without full conversational "
                    "interaction pressure"
                ),
                "why_single_turn_is_not_enough": (
                    "It helps teach follow-up resolution, but still needs explicit "
                    "value grounding, semantic metric, and recovery labels."
                ),
                "current_status": "available as a related dataset role, not yet a public result",
                "next_artifact": "row-compatible SParC manifest with the same scorer and claim ledger",
            },
            {
                "data_source": "Synthetic schema-rich SQL",
                "project_role": "schema-rich stress data for joins, grain, fanout, and metrics",
                "what_it_tests": (
                    "long DDL, bridge tables, duplicated children, metric definitions, "
                    "and awkward value/entity mappings"
                ),
                "why_single_turn_is_not_enough": (
                    "Synthetic rows can isolate value grounding, join fanout, and "
                    "metric preservation failures before expensive endpoint runs."
                ),
                "current_status": (
                    "fixture pack available: "
                    f"{synthetic_summary['fixture_count']} curated rows across "
                    "value normalization, entity resolution, fanout, MEASURE(), and recovery"
                ),
                "next_artifact": (
                    "turn the synthetic fixtures into row-matched metric-DSL, "
                    "semantic retrieval, and recovery endpoint evals"
                ),
            },
            {
                "data_source": "Tiny SQLite lab",
                "project_role": "portable notebook demonstration attached to the post",
                "what_it_tests": (
                    "five candidate targets on one four-turn scenario: direct SQL, "
                    "planner-first, semantic state, metric DSL, and recovery"
                ),
                "why_single_turn_is_not_enough": (
                    "The tiny lab makes context carryover, value grounding, metric "
                    "intent, and recovery visible without model serving."
                ),
                "current_status": "runnable local artifact; not a benchmark result",
                "next_artifact": "promote each isolated behavior into a dataset-backed manifest",
            },
        ]
    )


def single_to_multiturn_gap() -> pd.DataFrame:
    """Explain why single-turn SQL competence does not transfer automatically."""

    return pd.DataFrame(
        [
            {
                "mechanism": "state carryover",
                "single_turn_assumption": (
                    "A BIRD-style one-shot request contains the whole task in one "
                    "question plus schema text."
                ),
                "multi_turn_breakage": (
                    "A follow-up such as 'now by month' depends on the prior metric "
                    "and filters, but the current utterance no longer states them."
                ),
                "dataset_surface": (
                    "CoSQL fixed proxy and SParC context-dependent turns expose "
                    "filter and projection carryover."
                ),
                "repo_artifact_needed": (
                    "resolved standalone question, carried filter labels, and "
                    "planner state deltas before SQL generation"
                ),
            },
            {
                "mechanism": "value grounding",
                "single_turn_assumption": (
                    "The question literal is usually usable directly or the model can "
                    "guess the stored value from schema names."
                ),
                "multi_turn_breakage": (
                    "The user says France while the database stores FR; the "
                    "France -> FR mapping is missing, so the SQL can execute and "
                    "still return empty rows."
                ),
                "dataset_surface": (
                    "CoSQL value labels, the database-derived value index, and "
                    "Synthetic schema-rich SQL alias fixtures."
                ),
                "repo_artifact_needed": (
                    "value index, entity-resolution labels, alias coverage, and "
                    "value-retrieval accuracy before final SQL"
                ),
            },
            {
                "mechanism": "grain shift",
                "single_turn_assumption": (
                    "The requested grouping and projection are stated once and can "
                    "be compiled directly."
                ),
                "multi_turn_breakage": (
                    "A follow-up changes country totals into monthly totals while "
                    "the model must preserve the measure and avoid duplicated rows."
                ),
                "dataset_surface": (
                    "CoSQL/SParC follow-ups plus Synthetic schema-rich SQL bridge "
                    "tables and fanout fixtures."
                ),
                "repo_artifact_needed": (
                    "grain labels, duplicate-row policy, join-path labels, and "
                    "fanout-safe value checks"
                ),
            },
            {
                "mechanism": "metric intent",
                "single_turn_assumption": (
                    "A raw SQL expression is enough because the benchmark asks for "
                    "one executable query."
                ),
                "multi_turn_breakage": (
                    "Once the user keeps editing the analysis, expanding the metric "
                    "too early hides whether revenue means MEASURE(revenue), a "
                    "governed expression, or an ad hoc SUM."
                ),
                "dataset_surface": (
                    "Synthetic schema-rich SQL metric rows now, then metric-heavy "
                    "CoSQL/SParC/BIRD-Interact transfers."
                ),
                "repo_artifact_needed": (
                    "semantic model manifest, MEASURE() preservation score, DSL "
                    "parse/compile rate, and compiled-SQL execution delta"
                ),
            },
            {
                "mechanism": "result-aware recovery",
                "single_turn_assumption": (
                    "The model is scored on the first answer, so it never has to "
                    "use an empty result or error as feedback."
                ),
                "multi_turn_breakage": (
                    "If a previous turn returned no rows, the next turn may require "
                    "repairing the value, inspecting the database, or asking a "
                    "clarifying question instead of retrying the same SQL."
                ),
                "dataset_surface": (
                    "Tiny SQLite lab for the visible example; Synthetic schema-rich "
                    "SQL and BIRD-Interact for real repair pressure."
                ),
                "repo_artifact_needed": (
                    "execution-result trace, repair action labels, clarification "
                    "policy, and recovery-success scoring"
                ),
            },
            {
                "mechanism": "generated-history drift",
                "single_turn_assumption": (
                    "There is no history, or evaluation gives the model clean "
                    "reference history for every turn."
                ),
                "multi_turn_breakage": (
                    "Teacher-forced history hides compounding failures; production "
                    "agents must continue after their own bad SQL and result state."
                ),
                "dataset_surface": (
                    "CoSQL generated-history rollout first, then BIRD-Interact "
                    "same-protocol local-vs-hosted runs."
                ),
                "repo_artifact_needed": (
                    "generated-history manifest, same-model teacher-forced control, "
                    "latency/cost trace, and local-vs-hosted comparison"
                ),
            },
        ]
    )


def hosted_comparison_protocol() -> pd.DataFrame:
    """Define the evidence required before making local-vs-hosted claims."""

    return pd.DataFrame(
        [
            {
                "protocol_gate": "same input rows",
                "required_evidence": (
                    "A frozen prepared-input manifest with row IDs, dialog IDs, "
                    "database hashes, schema text hashes, prompt template hashes, "
                    "and no model-specific filtering."
                ),
                "why_it_matters": (
                    "A hosted/local comparison is meaningless if the models answer "
                    "different questions or see different schema context."
                ),
                "minimum_acceptance": (
                    "The hosted model manifest and local model manifest must cover "
                    "the same input rows with matching hashes."
                ),
                "claim_ids": (
                    "hosted_sota_same_protocol, local_beats_hosted_same_protocol"
                ),
            },
            {
                "protocol_gate": "same scorer and output schema",
                "required_evidence": (
                    "One execution scorer version, one value-normalized result "
                    "schema, scorer hash, and per-row failure classification."
                ),
                "why_it_matters": (
                    "Strict SQL string differences, alias labels, and harmless "
                    "column names should not decide the hosted/SOTA claim."
                ),
                "minimum_acceptance": (
                    "Both manifests report strict accuracy, value accuracy, syntax "
                    "accuracy, and classified failures from the same scorer build."
                ),
                "claim_ids": (
                    "hosted_sota_same_protocol, local_beats_hosted_same_protocol"
                ),
            },
            {
                "protocol_gate": "same oracle boundary",
                "required_evidence": (
                    "Explicit flags for oracle planner hints, teacher-forced "
                    "history, schema pruning, semantic artifacts, and value indexes."
                ),
                "why_it_matters": (
                    "Oracle planner diagnostics are useful ceilings, but they cannot "
                    "be mixed into a production-style hosted comparison."
                ),
                "minimum_acceptance": (
                    "Any run using labels derived from reference SQL is excluded "
                    "from local-vs-hosted win claims."
                ),
                "claim_ids": (
                    "hosted_sota_same_protocol, local_beats_hosted_same_protocol"
                ),
            },
            {
                "protocol_gate": "hosted model manifest",
                "required_evidence": (
                    "Hosted model name, provider, model version/date, prompt "
                    "hashes, decoding settings, retry policy, raw outputs, parsed "
                    "SQL, execution results, and per-row latency."
                ),
                "why_it_matters": (
                    "The phrase hosted SOTA is too vague unless the exact models "
                    "and settings are reproducible."
                ),
                "minimum_acceptance": (
                    "At least one strong hosted baseline is run through the same "
                    "pipeline before the local model is compared."
                ),
                "claim_ids": "hosted_sota_same_protocol",
            },
            {
                "protocol_gate": "local model manifest",
                "required_evidence": (
                    "Local base model, adapter hash, training data manifest, "
                    "checkpoint hash, prompt hash, decoding settings, raw outputs, "
                    "parsed SQL, execution results, and per-row latency."
                ),
                "why_it_matters": (
                    "A local win should identify the exact specialized model, not "
                    "only the endpoint or repo branch used for the run."
                ),
                "minimum_acceptance": (
                    "The local model manifest must show a positive local-vs-hosted "
                    "delta on value accuracy before the post can claim competition."
                ),
                "claim_ids": "local_beats_hosted_same_protocol",
            },
            {
                "protocol_gate": "generated-history rollout",
                "required_evidence": (
                    "A rollout manifest where each turn consumes the model's own "
                    "previous SQL and result state, plus a same-model teacher-forced "
                    "control."
                ),
                "why_it_matters": (
                    "Multi-turn analysis breaks through compounding mistakes; clean "
                    "history can hide exactly the behavior being evaluated."
                ),
                "minimum_acceptance": (
                    "Report generated-history value accuracy and recovery delta "
                    "beside the teacher-forced run."
                ),
                "claim_ids": (
                    "rollout_beats_teacher_forced_history, "
                    "bird_interact_local_vs_hosted"
                ),
            },
            {
                "protocol_gate": "latency and cost",
                "required_evidence": (
                    "Per-row latency, total tokens or local throughput, hosted API "
                    "cost, local hardware, and batch/concurrency settings."
                ),
                "why_it_matters": (
                    "A small local model can be useful even when accuracy is close, "
                    "but only if latency and cost are measured on the same rows."
                ),
                "minimum_acceptance": (
                    "The comparison reports accuracy, latency and cost together; "
                    "none of them can be inferred from model names."
                ),
                "claim_ids": (
                    "hosted_sota_same_protocol, local_beats_hosted_same_protocol"
                ),
            },
            {
                "protocol_gate": "BIRD-Interact transfer",
                "required_evidence": (
                    "A BIRD-Interact transfer manifest with the same scorer, row "
                    "identity contract, non-oracle boundary, rollout policy, and "
                    "hosted/local model manifests."
                ),
                "why_it_matters": (
                    "CoSQL is a fast proxy; the final question is whether the method "
                    "survives a richer interactive analysis benchmark."
                ),
                "minimum_acceptance": (
                    "A BIRD-Interact transfer run exists before making broad "
                    "multi-turn data-analysis claims."
                ),
                "claim_ids": "bird_interact_local_vs_hosted",
            },
        ]
    )


def evaluation_harness_map() -> pd.DataFrame:
    """Map research targets to executable repo surfaces."""

    return pd.DataFrame(
        [
            {
                "research_target": "direct SQL control",
                "module_path": "eval/run_eval.py",
                "command_surface": (
                    "python -m eval.run_eval --input data/processed/eval_cosql_dev_100.jsonl "
                    "--output results/<run-id>.jsonl --manifest-output results/<run-id>.manifest.json"
                ),
                "implemented_gate": (
                    "same rows and same scorer through result manifests; direct SQL "
                    "is the control every structured target must beat"
                ),
                "current_status": "implemented: proxy manifests exist; hosted/BIRD transfer pending",
                "claim_ids": (
                    "qwen35_9b_base_cosql_dev_100turns, "
                    "multiturn_sql_100_cosql_dev_100turns"
                ),
            },
            {
                "research_target": "planner quality before SQL",
                "module_path": "eval/planner_eval.py",
                "command_surface": (
                    "python -m eval.planner_eval --input data/processed/eval_cosql_dev_planner_100.jsonl "
                    "--output results/planner_eval_cosql_dev_100.jsonl "
                    "--summary-output results/planner_eval_cosql_dev_100_summary.json"
                ),
                "implemented_gate": (
                    "planner labels are scored before SQL so planner movement is "
                    "not confused with execution movement"
                ),
                "current_status": "implemented: lexical planner summary and readiness report exist",
                "claim_ids": "planner_lexical_schema_baseline",
            },
            {
                "research_target": "predicted-planner SQL",
                "module_path": "eval/run_predicted_planner_comparison.py",
                "command_surface": (
                    "python -m eval.planner_optimize ...; python -m eval.planner_predict ...; "
                    "python -m eval.run_predicted_planner_comparison --direct-input ... "
                    "--predicted-input ..."
                ),
                "implemented_gate": (
                    "same rows, same scorer, same model, and same oracle policy for "
                    "direct SQL versus predicted-planner SQL"
                ),
                "current_status": (
                    "paired runner implemented; current planner readiness says "
                    "improve_planner_before_claim"
                ),
                "claim_ids": "predicted_planner_sql_execution",
            },
            {
                "research_target": "semantic/value artifacts",
                "module_path": "data/value_artifacts.py",
                "command_surface": (
                    "python -m data.value_artifacts ...; python -m data.value_index ...; "
                    "python -m eval.classify_errors ..."
                ),
                "implemented_gate": (
                    "value labels, database-derived value index, and schema diagnostics "
                    "make value/entity and alias failures visible before more tuning"
                ),
                "current_status": "implemented: value labels, value index, and schema findings are generated",
                "claim_ids": (
                    "semantic_prompt_minimal_executable_cosql_dev_100turns, "
                    "predicted_planner_sql_execution"
                ),
            },
            {
                "research_target": "MEASURE()-preserving DSL",
                "module_path": "eval/metric_dsl_eval.py",
                "command_surface": (
                    "python -m eval.metric_dsl_eval --input results/metric_dsl/<run-id>.predictions.jsonl "
                    "--manifest-output results/metric_dsl/<run-id>.manifest.json; "
                    "python -m eval.compare_metric_dsl_direct_sql ..."
                ),
                "implemented_gate": (
                    "DSL parse, compile, MEASURE() preservation, semantic model hashes, "
                    "and compiled SQL execution are compared with direct SQL"
                ),
                "current_status": "implemented evaluator; prediction/comparison manifest still pending",
                "claim_ids": "metric-dsl-bootstrap.metric_dsl, metric_dsl_beats_direct_sql",
            },
            {
                "research_target": "generated-history rollout",
                "module_path": "eval/rollout_eval.py",
                "command_surface": (
                    "python -m eval.rollout_eval --input data/processed/eval_cosql_dev_100.jsonl "
                    "--manifest-output results/rollout/<run-id>.manifest.json; "
                    "python -m eval.compare_rollout_history ..."
                ),
                "implemented_gate": (
                    "model-generated prior SQL and result state are compared with "
                    "same-model teacher-forced history"
                ),
                "current_status": "implemented evaluator; rollout comparison artifact still pending",
                "claim_ids": (
                    "model_generated_history_rollout, "
                    "rollout_beats_teacher_forced_history"
                ),
            },
            {
                "research_target": "hosted/local comparison",
                "module_path": "eval/compare_hosted_baseline.py",
                "command_surface": (
                    "python -m eval.compare_hosted_baseline --hosted-manifest ... "
                    "--local-manifest ... --output ..."
                ),
                "implemented_gate": (
                    "same rows, same scorer, latency/cost fields, and positive "
                    "local-vs-hosted delta before public comparison claims"
                ),
                "current_status": "implemented comparator; hosted baseline manifest still pending",
                "claim_ids": (
                    "hosted_sota_same_protocol, "
                    "local_beats_hosted_same_protocol"
                ),
            },
        ]
    )


def method_readiness_report() -> pd.DataFrame:
    """Summarize which method comparisons are supported, blocked, or rankable."""

    rows = build_method_readiness(
        ledger_path=artifact_path("docs/claim_ledgers/cosql_dev_100.jsonl"),
        repo_root=repo_root(),
    )
    frame = pd.DataFrame(rows)
    frame["control_ready_now"] = frame["control_ready_now"].astype(object)
    frame["rankable_now"] = frame["rankable_now"].astype(object)
    return frame[
        [
            "method",
            "readiness_level",
            "control_ready_now",
            "rankable_now",
            "supported_claim_ids",
            "blocking_claim_ids",
            "module_path",
            "next_command",
            "next_artifact",
            "claim_boundary",
        ]
    ]


def finetuning_step_plan() -> pd.DataFrame:
    """Summarize concrete train/evaluate/compare steps for each method arm."""

    summary = finetuning_step_summary(repo_root=repo_root())
    rows = []
    for step in summary["steps"]:
        rows.append(
            {
                "step_id": step["step_id"],
                "method": step["method"],
                "stage": step["stage"],
                "protocols": "; ".join(step["benchmark_protocol_ids"]),
                "rows_ready": (
                    f"train={step['train_rows_ready']}; "
                    f"eval={step['eval_rows_ready']}; "
                    f"control={step['control_rows_ready']}"
                ),
                "preflight_command_count": step["preflight_command_count"],
                "cheap_preflight_available": step["cheap_preflight_available"],
                "command_count": step["command_count"],
                "evidence_gate": step["evidence_gate"],
                "clears_claim_ids": "; ".join(step["clears_claim_ids"]),
                "blocks_claim_ids": "; ".join(step["blocks_claim_ids"]),
            }
        )
    return pd.DataFrame(rows)


def _manifest_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _manifest_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_manifest_json_value(item) for item in value]
    if hasattr(value, "item"):
        with suppress(AttributeError, TypeError, ValueError):
            value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def shareable_lab_attachment() -> pd.DataFrame:
    reader_flow = " -> ".join(
        [
            "Research question",
            "open the published HTML lab",
            "run the lab checkpoints",
            "inspect the failure trace",
            "compare fine-tuning targets",
            "read the evidence gates",
        ]
    )
    return pd.DataFrame(
        [
            {
                "artifact": "shareable lab notebook and attached codebase",
                "notebook": LAB_APP,
                "repo_url": "https://github.com/xdanny/multiturn-sql-finetuning",
                "published_html_url": LAB_HTML_URL,
                "run_command": f"marimo edit {LAB_APP}",
                "alternate_command": f"jupyter lab {LAB_NOTEBOOK}",
                "device_policy": (
                    "The lab auto-selects CUDA, MPS, or XPU when PyTorch detects "
                    "an available accelerator and falls back to CPU."
                ),
                "reader_flow": reader_flow,
                "what_runs": (
                    "One compact published lab runs the SQLite scenario, "
                    "compares direct SQL, planner-first, semantic-layer, "
                    "MEASURE()-preserving DSL, and behavior/recovery targets, "
                    "then connects those behaviors to generated repo evidence."
                ),
                "claim_boundary": (
                    "This is a notebook lab for reasoning about method targets, "
                    "not a benchmark result or hosted-SOTA comparison."
                ),
            }
        ]
    )


def lab_reader_flow() -> pd.DataFrame:
    rows = []
    for step, section in enumerate(LAB_READER_SECTIONS, start=1):
        rows.append(
            {
                "step": step,
                "lab_step": section["lab_step"],
                "notebook": LAB_APP,
                "run_command": f"marimo edit {LAB_APP}",
                "alternate_command": f"jupyter lab {LAB_NOTEBOOK}",
                "reader_action": section["reader_action"],
                "evidence_to_inspect": section["evidence_to_inspect"],
                "purpose": section["purpose"],
                "claim_boundary": section["claim_boundary"],
            }
        )
    return pd.DataFrame(rows)


def experiment_ladder() -> pd.DataFrame:
    """Define the ordered experiments required before a broad comparison claim."""

    return pd.DataFrame(
        [
            {
                "rung": 1,
                "experiment": "Direct SQL control",
                "question": (
                    "Does ordinary direct-SQL tuning help on the same rows, same "
                    "local endpoint path, same scorer, and no oracle hints?"
                ),
                "required_artifact": (
                    "direct-SQL endpoint manifest on the fixed CoSQL proxy and, "
                    "later, the BIRD-Interact transfer rows"
                ),
                "claim_gate": (
                    "This is the control arm. It can support a proxy movement "
                    "claim, but every structured target must beat it before being "
                    "called better."
                ),
            },
            {
                "rung": 2,
                "experiment": "Planner quality before SQL",
                "question": (
                    "Can a non-oracle planner recover tables, columns, joins, "
                    "projection shape, grouping, and duplicate policy before "
                    "the generator writes SQL?"
                ),
                "required_artifact": (
                    "planner scorecard with row identities, planner labels, and "
                    "planner-readiness diagnostics"
                ),
                "claim_gate": (
                    "First score the plan before SQL. Do not turn planner movement "
                    "into an execution claim until the plan is good enough to feed."
                ),
            },
            {
                "rung": 3,
                "experiment": "Predicted-planner SQL",
                "question": (
                    "Do non-oracle predicted plans improve generated SQL against "
                    "the direct-SQL control on identical rows?"
                ),
                "required_artifact": (
                    "paired direct-vs-predicted endpoint manifest from "
                    "eval.run_predicted_planner_comparison"
                ),
                "claim_gate": (
                    "The predicted-planner path must beat direct SQL with the same "
                    "model, rows, scorer, prompt boundary, and oracle policy."
                ),
            },
            {
                "rung": 4,
                "experiment": "Semantic-layer target",
                "question": (
                    "Can the model or retrieval layer recover governed entities, "
                    "dimensions, measures, grain, joins, and value aliases without "
                    "flooding the prompt?"
                ),
                "required_artifact": (
                    "versioned semantic artifacts, value/entity retrieval scores, "
                    "and row-matched SQL deltas"
                ),
                "claim_gate": (
                    "Semantic context must reduce value, entity, and grain errors "
                    "on the same rows without hiding cost or latency."
                ),
            },
            {
                "rung": 5,
                "experiment": "MEASURE()-preserving DSL",
                "question": (
                    "Can the model preserve MEASURE() intent and compile it through "
                    "a semantic model before comparing value execution?"
                ),
                "required_artifact": (
                    "metric-DSL prediction manifest with parse, compile, "
                    "MEASURE() preservation, semantic-model hashes, and compiled "
                    "SQL execution"
                ),
                "claim_gate": (
                    "Metric DSL must parse, compile, preserve governed measures, "
                    "and match or beat direct SQL on the same metric-heavy rows."
                ),
            },
            {
                "rung": 6,
                "experiment": "Generated-history rollout",
                "question": (
                    "Can the system continue after its own previous SQL, empty "
                    "results, and repair attempts instead of reading clean history?"
                ),
                "required_artifact": (
                    "model-generated history rollout manifest with execution "
                    "results, repair actions, stop reasons, and same-model controls"
                ),
                "claim_gate": (
                    "Generated-history rollout must beat or explain teacher-forced "
                    "history on the same model and rows before recovery claims count."
                ),
            },
            {
                "rung": 7,
                "experiment": "Hosted and BIRD-Interact comparison",
                "question": (
                    "Can the small specialized local model challenge hosted models "
                    "on the richer interactive analysis setting?"
                ),
                "required_artifact": (
                    "same-protocol hosted baselines, local-vs-hosted manifest, "
                    "latency/cost report, and BIRD-Interact transfer manifest"
                ),
                "claim_gate": (
                    "Only compare with hosted SOTA after the local protocol is "
                    "stable, non-oracle, row-matched, and transferred to "
                    "BIRD-Interact-style tasks."
                ),
            },
        ]
    )


def lab_method_scorecard() -> pd.DataFrame:
    report = run_multiturn_lab()
    target_by_system = {
        row["system"]: row["fine_tuning_target"] for row in report["method_matrix"]
    }
    rows = []
    for system, metrics in report["systems"].items():
        rows.append(
            {
                "system": system,
                "fine_tuning_target": target_by_system[system],
                "value_accuracy": float(metrics["value_accuracy"]),
                "context_carryover_accuracy": float(metrics["context_carryover_accuracy"]),
                "value_grounding_accuracy": float(metrics["value_grounding_accuracy"]),
                "measure_preservation_rate": float(metrics["measure_preservation_rate"]),
                "recovery_success_rate": float(metrics["recovery_success_rate"]),
                "lab_takeaway": (
                    "This is not a benchmark result; it shows which intermediate "
                    "behavior the target isolates in the shareable lab."
                ),
            }
        )
    return pd.DataFrame(rows)


def lab_failure_trace() -> pd.DataFrame:
    report = run_multiturn_lab()
    selected = {
        ("turn_2", "direct_sql_baseline"),
        ("turn_3", "direct_sql_baseline"),
        ("turn_4", "direct_sql_baseline"),
        ("turn_4", "planner_first_sql"),
        ("turn_4", "semantic_value_sql"),
        ("turn_4", "semantic_dsl_planner"),
        ("turn_4", "behavior_recovery_sql"),
    }
    why_by_key = {
        ("turn_2", "direct_sql_baseline"): (
            "The direct SQL target copies the display value instead of learning "
            "the value map France -> FR."
        ),
        ("turn_3", "direct_sql_baseline"): (
            "The direct SQL target changes grain but drops the carried country filter."
        ),
        ("turn_4", "direct_sql_baseline"): (
            "Retrying the same value-grounding mistake does not use execution feedback."
        ),
        ("turn_4", "planner_first_sql"): (
            "The planner carries state, but still repeats the display-value mistake; "
            "planning alone is not recovery."
        ),
        ("turn_4", "semantic_value_sql"): (
            "Semantic grounding returns the right rows, but this row does not prove "
            "the model inspected the failed previous result."
        ),
        ("turn_4", "semantic_dsl_planner"): (
            "The DSL keeps the metric and value semantics, but it still lacks an "
            "explicit empty-result repair action."
        ),
        ("turn_4", "behavior_recovery_sql"): (
            "The recovery target uses the empty-result signal to repair the previous turn."
        ),
    }
    rows = []
    for row in report["rows"]:
        key = (row["turn_id"], row["system"])
        if key not in selected:
            continue
        if row["value_match"] and row["requires_recovery"] and not row["recovery_success"]:
            failure_type = "correct_rows_no_repair"
        else:
            failure_type = row["failure_type"] or "recovery_success"
        rows.append(
            {
                "turn_id": row["turn_id"],
                "question": row["question"],
                "system": row["system"],
                "failure_type": failure_type,
                "requires_recovery": row["requires_recovery"],
                "value_match": row["value_match"],
                "recovery_success": row["recovery_success"],
                "actual_rows": row["actual_rows"],
                "expected_rows": row["expected_rows"],
                "intermediate_plan": row["intermediate_plan"],
                "why_it_matters": why_by_key[key],
            }
        )
    return pd.DataFrame(rows)


def data_engineering_gates() -> pd.DataFrame:
    value_summary = read_json_artifact(
        "docs/data_artifacts/value_grounding_labels_cosql_dev_100_summary.json"
    )
    value_index = read_json_artifact(
        "docs/data_artifacts/value_index_cosql_dev_100_summary.json"
    )
    synthetic_summary = read_json_artifact(
        "docs/data_artifacts/synthetic_method_fixtures_summary.json"
    )
    value_index_coverage = value_index.get("coverage") or {}
    return pd.DataFrame(
        [
            {
                "gate": "fixed_proxy_slice",
                "artifact": "CoSQL dev 100-turn manifest",
                "problem_exposed": (
                    "Small fixed proxy slice: 100 turns across 32 dialogs, held "
                    "constant while prompts, adapters, planners, and evaluators change."
                ),
                "why_it_matters": (
                    "Without a fixed slice, every model or prompt change can hide "
                    "whether the method improved or the task changed."
                ),
                "current_status": "in_use: fixed CoSQL proxy with teacher-forced history",
                "next_repo_action": (
                    "Keep the proxy manifest immutable and add matching generated-history runs."
                ),
                "blocks_claim": "Blocks broad benchmark claims beyond the proxy slice.",
                "source_artifacts": (
                    "data/processed/eval_cosql_dev_100.jsonl; "
                    "docs/claim_ledgers/cosql_dev_100.jsonl; "
                    "notebooks/labs/local_multiturn_sql_lab.ipynb"
                ),
                "claim_ids": (
                    "qwen35_9b_base_cosql_dev_100turns, "
                    "multiturn_sql_100_cosql_dev_100turns, "
                    "semantic_prompt_minimal_executable_cosql_dev_100turns"
                ),
            },
            {
                "gate": "generated_history_rollout",
                "artifact": "model-generated history manifest",
                "problem_exposed": (
                    "Teacher-forced evaluation gives the model clean prior turns; "
                    "real agents must live with their own earlier SQL and result state."
                ),
                "why_it_matters": (
                    "A multi-turn model can look good when prior context is clean "
                    "and still fail after its first wrong turn."
                ),
                "current_status": "pending: rollout claim is tracked but not yet supported",
                "next_repo_action": (
                    "Run the same model with generated prior turns and compare "
                    "against teacher-forced history."
                ),
                "blocks_claim": "Blocks recovery and interactive-agent claims.",
                "source_artifacts": "docs/claim_ledgers/cosql_dev_100.jsonl",
                "claim_ids": (
                    "model_generated_history_rollout, "
                    "rollout_beats_teacher_forced_history"
                ),
            },
            {
                "gate": "value_entity_normalization",
                "artifact": "value index and entity-resolution labels",
                "problem_exposed": (
                    "Display values and stored values diverge, e.g. France -> FR, "
                    "aliases, casing, abbreviations, dates, people, teams, and venues."
                ),
                "why_it_matters": (
                    "The SQL can be syntactically right and still return zero rows "
                    "because the value grounding is wrong."
                ),
                "current_status": (
                    "partial: value-grounding labels now cover "
                    f"{value_summary['value_reference_count']} SQL value references "
                    f"across {value_summary['database_count']} databases; the "
                    f"database-derived value index has {value_index['entry_count']} "
                    f"entries and covers "
                    f"{float(value_index_coverage['mention_alias_indexed_rate']):.3f} "
                    "of user mention aliases"
                ),
                "next_repo_action": (
                    "Add alias/entity expansion on top of the value index, then "
                    "score value-grounding retrieval before SQL generation."
                ),
                "blocks_claim": "Blocks semantic grounding and recovery claims.",
                "source_artifacts": (
                    "docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl; "
                    "docs/data_artifacts/value_grounding_labels_cosql_dev_100_summary.json; "
                    "docs/data_artifacts/value_index_cosql_dev_100.jsonl; "
                    "docs/data_artifacts/value_index_cosql_dev_100_summary.json; "
                    "docs/claim_ledgers/cosql_dev_100.jsonl; "
                    "results/classified/minimal_executable.jsonl"
                ),
                "claim_ids": (
                    "qwen35_9b_base_cosql_dev_100turns, "
                    "semantic_prompt_minimal_executable_cosql_dev_100turns"
                ),
            },
            {
                "gate": "join_fanout_fixtures",
                "artifact": "grain, bridge-table, and fanout fixtures",
                "problem_exposed": (
                    "bridge tables and duplicated child rows can multiply facts, "
                    "changing metric values while the query still executes."
                ),
                "why_it_matters": (
                    "Execution success is not enough if the join path changes the "
                    "grain or duplicates rows."
                ),
                "current_status": (
                    "partial: synthetic fixture pack now includes "
                    f"{synthetic_summary['failure_mode_counts']['grain_fanout']} "
                    "grain/fanout row with an explicit duplicate-row policy"
                ),
                "next_repo_action": (
                    "Run planner, semantic, and metric-DSL predictions on the "
                    "fixture row and require the duplicate-row policy before "
                    "promoting fanout-heavy endpoint claims."
                ),
                "blocks_claim": "Blocks trustworthy analytical metric claims.",
                "source_artifacts": (
                    "docs/data_artifacts/synthetic_method_fixtures.jsonl; "
                    "docs/data_artifacts/synthetic_method_fixtures_summary.json; "
                    "docs/claim_ledgers/cosql_dev_100.jsonl; "
                    "results/classified/vllm_qwen35_9b_base_cosql_dev_100turns.jsonl"
                ),
                "claim_ids": (
                    "qwen35_9b_base_cosql_dev_100turns, "
                    "multiturn_sql_100_cosql_dev_100turns"
                ),
            },
            {
                "gate": "alias_schema_validation",
                "artifact": "pre-execution schema and alias validator",
                "problem_exposed": (
                    "Wrong-table column references and alias-role mistakes collapse "
                    "into generic bad-SQL failures."
                ),
                "why_it_matters": (
                    "A repair loop needs to know whether the miss was planning, "
                    "aliasing, dialect, value grounding, or execution."
                ),
                "current_status": (
                    "partial: pre-execution schema validator now emits "
                    "wrong-table, unknown-column, and ambiguous-column diagnostics"
                ),
                "next_repo_action": (
                    "Promote schema diagnostics into repair prompts and retrain "
                    "planner/generator rows where the error is repairable."
                ),
                "blocks_claim": "Blocks precise planner-versus-generator attribution.",
                "source_artifacts": (
                    "docs/planner_baseline_cosql_dev_100_summary.json; "
                    "docs/claim_ledgers/cosql_dev_100.jsonl"
                ),
                "claim_ids": (
                    "planner_lexical_schema_baseline, "
                    "predicted_planner_sql_execution"
                ),
            },
            {
                "gate": "semantic_metric_manifest",
                "artifact": "versioned semantic model and MEASURE() prediction manifest",
                "problem_exposed": (
                    "A raw SQL target can expand governed metrics too early and "
                    "hide whether the model preserved metric intent."
                ),
                "why_it_matters": (
                    "Business analysis needs stable measures, dimensions, grain, "
                    "allowed joins, and semantic-model versions."
                ),
                "current_status": (
                    "partial: metric-DSL parser/evaluator exists and the synthetic "
                    "fixture pack now includes "
                    f"{synthetic_summary['training_target_counts']['metric_dsl']} "
                    "metric-DSL target rows; model predictions pending"
                ),
                "next_repo_action": (
                    "Generate metric-DSL predictions, compile them through the same "
                    "semantic model, and compare against direct SQL."
                ),
                "blocks_claim": "Blocks MEASURE()-first or semantic-layer superiority claims.",
                "source_artifacts": (
                    "docs/data_artifacts/synthetic_method_fixtures.jsonl; "
                    "docs/data_artifacts/synthetic_method_fixtures_summary.json; "
                    "docs/claim_ledgers/cosql_dev_100.jsonl"
                ),
                "claim_ids": (
                    "metric-dsl-bootstrap.metric_dsl, "
                    "metric_dsl_beats_direct_sql"
                ),
            },
            {
                "gate": "hosted_same_protocol_baseline",
                "artifact": "hosted-model baseline manifest",
                "problem_exposed": (
                    "Local-model gains are not meaningful unless hosted models run "
                    "the same inputs, scorer, prompt boundary, and latency/cost accounting."
                ),
                "why_it_matters": (
                    "A SOTA comparison is otherwise a story about different protocols."
                ),
                "current_status": (
                    "pending: claim ledger separately tracks the missing hosted "
                    "baseline and local-vs-hosted win"
                ),
                "next_repo_action": (
                    "Run hosted baselines through the fixed CoSQL proxy, then compare "
                    "the local manifest against the same rows with a positive value delta."
                ),
                "blocks_claim": "Blocks hosted-SOTA comparison claims.",
                "source_artifacts": "docs/claim_ledgers/cosql_dev_100.jsonl",
                "claim_ids": (
                    "hosted_sota_same_protocol, "
                    "local_beats_hosted_same_protocol"
                ),
            },
            {
                "gate": "bird_interact_transfer",
                "artifact": "BIRD-Interact transfer manifest",
                "problem_exposed": (
                    "CoSQL and SParC are useful proxy datasets, but BIRD-Interact "
                    "is closer to the final multi-turn data-analysis claim."
                ),
                "why_it_matters": (
                    "A method that only works on the proxy may not survive richer "
                    "schemas, values, and interaction patterns."
                ),
                "current_status": "pending: proxy loop only",
                "next_repo_action": (
                    "Port the planner, value, metric-DSL, and rollout contracts to "
                    "BIRD-Interact before claiming transfer."
                ),
                "blocks_claim": "Blocks final interactive benchmark claims.",
                "source_artifacts": "docs/claim_ledgers/cosql_dev_100.jsonl",
                "claim_ids": "bird_interact_local_vs_hosted",
            },
        ]
    )


def data_artifact_contract() -> pd.DataFrame:
    """Define the dataset artifacts required before method claims become credible."""

    return pd.DataFrame(
        [
            {
                "artifact": "value_index",
                "failure_isolated": (
                    "Display-to-storage mismatches such as France -> FR, casing, "
                    "abbreviations, Roman numerals, dates, teams, venues, and people."
                ),
                "labels_or_fields": (
                    "database_id, table, column, raw_value, normalized_value, "
                    "aliases, source_frequency, index_source, schema_version"
                ),
                "consumer": (
                    "semantic-layer prompts, planner value slots, recovery prompts, "
                    "and metric-DSL filters"
                ),
                "verification_gate": (
                    "docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl "
                    "for gold SQL-derived labels plus "
                    "docs/data_artifacts/value_index_cosql_dev_100.jsonl for the "
                    "database-derived non-oracle index; next gate is same rows "
                    "value-grounding retrieval accuracy before SQL generation"
                ),
                "claim_ids": (
                    "semantic_prompt_minimal_executable_cosql_dev_100turns, "
                    "predicted_planner_sql_execution"
                ),
            },
            {
                "artifact": "entity_resolution_labels",
                "failure_isolated": (
                    "Mentions in follow-up turns resolve to the wrong entity, wrong "
                    "table role, or stale conversation turn."
                ),
                "labels_or_fields": (
                    "turn_id, mention_text, resolved_table, resolved_column, "
                    "resolved_value, evidence_span, prior_turn_reference"
                ),
                "consumer": (
                    "planner supervision, semantic-state tuning, and generated-history "
                    "rollout diagnostics"
                ),
                "verification_gate": (
                    "same rows entity-resolution F1 before SQL generation and value "
                    "accuracy after SQL generation"
                ),
                "claim_ids": (
                    "semantic_prompt_minimal_executable_cosql_dev_100turns, "
                    "rollout_beats_teacher_forced_history"
                ),
            },
            {
                "artifact": "grain_fanout_fixtures",
                "failure_isolated": (
                    "Bridge tables, duplicated child rows, and many-to-many joins "
                    "change metric values while SQL still executes."
                ),
                "labels_or_fields": (
                    "schema_id, fact_table, bridge_table, fanout_path, grain, "
                    "duplicate_row_policy, expected_metric_delta"
                ),
                "consumer": (
                    "planner/DSL training, semantic-model manifests, and SQL "
                    "execution scorers"
                ),
                "verification_gate": (
                    "same rows fanout-safe value accuracy plus explicit duplicate "
                    "policy match"
                ),
                "claim_ids": (
                    "multiturn_sql_100_cosql_dev_100turns, "
                    "metric_dsl_beats_direct_sql"
                ),
            },
            {
                "artifact": "semantic_model_manifest",
                "failure_isolated": (
                    "Raw DDL does not define governed measures, dimensions, grain, "
                    "allowed joins, or when MEASURE() must be preserved."
                ),
                "labels_or_fields": (
                    "semantic_model_id, version, measures, dimensions, grains, "
                    "joins, MEASURE() definitions, model_sha256"
                ),
                "consumer": (
                    "semantic-layer tuning, MEASURE()-preserving metric DSL, "
                    "compiled SQL evaluation, and hosted baseline parity checks"
                ),
                "verification_gate": (
                    "metric-DSL parse/compile manifest and same rows direct-SQL "
                    "comparison with semantic model hashes"
                ),
                "claim_ids": (
                    "metric-dsl-bootstrap.metric_dsl, "
                    "metric_dsl_beats_direct_sql"
                ),
            },
            {
                "artifact": "schema_alias_validator",
                "failure_isolated": (
                    "Wrong-table columns, alias-role mistakes, and ambiguous "
                    "unqualified references look like generic execution failures."
                ),
                "labels_or_fields": (
                    "query_id, alias_map, column_bindings, unknown_columns, "
                    "wrong_table_columns, ambiguous_columns, repair_hint"
                ),
                "consumer": (
                    "planner/generator attribution, repair data, and pre-execution "
                    "SQL diagnostics"
                ),
                "verification_gate": (
                    "validator manifest with same rows repairable-error counts and "
                    "schema diagnostic examples"
                ),
                "claim_ids": (
                    "planner_lexical_schema_baseline, "
                    "predicted_planner_sql_execution"
                ),
            },
            {
                "artifact": "generated_history_trace",
                "failure_isolated": (
                    "teacher-forced history hides whether the model can recover from "
                    "its own previous SQL, empty results, or bad value grounding."
                ),
                "labels_or_fields": (
                    "dialog_id, turn_id, generated_sql, execution_result, feedback, "
                    "state_delta, repair_action, stop_reason"
                ),
                "consumer": (
                    "behavior/recovery tuning, rollout evaluation, and BIRD-Interact "
                    "transfer experiments"
                ),
                "verification_gate": (
                    "generated-history rollout manifest on same rows compared with "
                    f"the same model under teacher-forced history; lab demonstration "
                    f"is in {LAB_NOTEBOOK}"
                ),
                "claim_ids": (
                    "model_generated_history_rollout, "
                    "rollout_beats_teacher_forced_history, "
                    "bird_interact_local_vs_hosted"
                ),
            },
        ]
    )


def _prompt_summary_row(relative_path: str) -> dict[str, Any]:
    summary = read_csv_artifact(relative_path)
    if summary.empty:
        raise ValueError(f"Prompt summary is empty: {relative_path}")
    best = summary.sort_values(
        by=["accuracy", "syntax_accuracy"],
        ascending=[False, False],
    ).iloc[0]
    dspy_rows = summary[summary["prompt_variant_source"] == "dspy"]
    best_dspy_accuracy = None
    if not dspy_rows.empty:
        best_dspy_accuracy = float(dspy_rows["accuracy"].max())
    return {
        "best_variant": str(best["prompt_variant"]),
        "best_source": str(best["prompt_variant_source"]),
        "best_accuracy": float(best["accuracy"]),
        "best_dspy_accuracy": best_dspy_accuracy,
        "samples": int(best["samples"]),
    }


def prompt_optimization_findings() -> pd.DataFrame:
    smoke_path = "results/prompt_search_semantic50_limit30/summary.csv"
    oracle_path = "results/prompt_search_schema_pruned_projection_schemafix_100/summary.csv"
    smoke = _prompt_summary_row(smoke_path)
    oracle = _prompt_summary_row(oracle_path)
    rows = [
        {
            "optimization_scope": "non_oracle_sql_prompt_smoke",
            "source_artifact": smoke_path,
            **smoke,
            "promoted_decision": (
                "tie: DSPy matched the best static prompt on a 30-row smoke run, "
                "so it is useful as a harness but not a promotion by itself."
            ),
            "next_program_target": (
                "Run prompt search on a held-out proxy slice and score failure-taxonomy deltas."
            ),
            "claim_boundary": "Prompt search smoke evidence only; not a SOTA claim.",
        },
        {
            "optimization_scope": "oracle_schema_pruned_prompt_search",
            "source_artifact": oracle_path,
            **oracle,
            "promoted_decision": (
                "static: the concise schema_pruned_minimal contract beat the best "
                "DSPy wording, so longer final-SQL prompts are not the next bet."
            ),
            "next_program_target": (
                "Use DSPy to optimize the planner contract, not only final SQL wording."
            ),
            "claim_boundary": "Oracle diagnostic evidence only; not a SOTA claim.",
        },
        {
            "optimization_scope": "planner_program_optimization_gate",
            "source_artifact": "docs/planner_baseline_cosql_dev_100_summary.json",
            "best_variant": "not_run",
            "best_source": "pending",
            "best_accuracy": None,
            "best_dspy_accuracy": None,
            "samples": 0,
            "promoted_decision": (
                "harness_ready: eval.planner_optimize can score static and DSPy "
                "planner policies by planner F1 before SQL generation."
            ),
            "next_program_target": (
                "Run eval.planner_optimize with DSPy proposals, promote the best "
                "planner policy into eval.planner_predict, then use "
                "eval.run_predicted_planner_comparison to run direct SQL and "
                "predicted-planner SQL with the same model, rows, scorer, and "
                "oracle policy."
            ),
            "claim_boundary": (
                "Planner optimizer exists, but no planner-search summary is "
                "published yet; not a SOTA claim."
            ),
        },
    ]
    return pd.DataFrame(rows, dtype=object)


def target_comparison() -> pd.DataFrame:
    ledger = claim_ledger().set_index("claim_id")
    best_non_oracle = float(
        ledger.loc[
            "semantic_prompt_minimal_executable_cosql_dev_100turns",
            "value_execution_accuracy",
        ]
    )
    direct_strict = float(
        ledger.loc[
            "multiturn_sql_100_cosql_dev_100turns",
            "strict_execution_accuracy",
        ]
    )
    direct_value = float(
        ledger.loc[
            "multiturn_sql_100_cosql_dev_100turns",
            "value_execution_accuracy",
        ]
    )
    semantic_status = str(
        ledger.loc[
            "semantic_prompt_minimal_executable_cosql_dev_100turns",
            "claim_status",
        ]
    )
    planner_status = str(ledger.loc["predicted_planner_sql_execution", "claim_status"])
    metric_status = str(ledger.loc["metric_dsl_beats_direct_sql", "claim_status"])
    rollout_status = str(
        ledger.loc["rollout_beats_teacher_forced_history", "claim_status"]
    )

    return pd.DataFrame(
        [
            {
                "fine_tuning_target": "Direct SQL SFT",
                "hypothesis": "A small model can learn conversational SQL directly from chat-format SQL rows.",
                "current_evidence": (
                    f"Supported proxy: 100-step LoRA reached {direct_strict:.3f} strict "
                    f"accuracy and {direct_value:.3f} value accuracy on the fixed "
                    "CoSQL slice."
                ),
                "claim_status": "supported_proxy",
                "next_gate": "Run the same target on hosted baselines and BIRD-Interact-style tasks.",
            },
            {
                "fine_tuning_target": "Planner/DSL first, SQL second",
                "hypothesis": "Predicting tables, joins, grain, filters, and projection before SQL should reduce context failures.",
                "current_evidence": "Planner F1 is measurable, but predicted-plan SQL execution has not beaten direct SQL yet.",
                "claim_status": planner_status,
                "next_gate": "Run predicted-planner endpoint SQL and compare against matching direct SQL manifests.",
            },
            {
                "fine_tuning_target": "Semantic-layer tuning",
                "hypothesis": "Learning governed entities, dimensions, measures, grain, and joins should improve analytical correctness.",
                "current_evidence": (
                    f"Supported proxy: semantic prompt policy reaches {best_non_oracle:.3f} "
                    "value accuracy, but broad semantic context is not yet a proven method win."
                ),
                "claim_status": semantic_status,
                "next_gate": "Replace derived schema summaries with versioned semantic artifacts and measure deltas on matching rows.",
            },
            {
                "fine_tuning_target": "MEASURE()-preserving metric DSL",
                "hypothesis": "The model should preserve governed metric intent before SQL compilation.",
                "current_evidence": "Metric-DSL parse, compile, execution, and measure-preservation gates exist; comparison is still pending.",
                "claim_status": metric_status,
                "next_gate": "Evaluate metric-DSL predictions against a direct-SQL baseline on the same metric-heavy rows.",
            },
            {
                "fine_tuning_target": "Behavior/recovery tuning",
                "hypothesis": "A useful agent must clarify, inspect values, repair failures, and recover after its own earlier mistakes.",
                "current_evidence": (
                    "The shareable lab isolates empty-result repair as a separate "
                    "behavior; teacher-forced CoSQL history is supported, but "
                    "generated-history rollout improvement is still pending."
                ),
                "claim_status": rollout_status,
                "next_gate": "Run model-generated-history rollout and compare it with teacher-forced history for the same model.",
            },
        ]
    )


def target_evidence_matrix() -> pd.DataFrame:
    """Join the lab taxonomy to manifest-backed evidence and missing gates."""

    ledger = claim_ledger().set_index("claim_id")
    planner = read_json_artifact("docs/planner_baseline_cosql_dev_100_summary.json")
    preflight = read_json_artifact("docs/predicted_planner_comparison_preflight.json")

    def status(claim_id: str) -> str:
        return str(ledger.loc[claim_id, "claim_status"])

    def value(claim_id: str, metric: str) -> float:
        return float(ledger.loc[claim_id, metric])

    rows = [
        {
            "fine_tuning_target": "Direct SQL SFT",
            "lab_behavior": "Baseline failure mode: valid SQL loses follow-up state and value grounding.",
            "manifest_backed_evidence": (
                "100-step LoRA is manifest-backed on the fixed CoSQL proxy: "
                f"{value('multiturn_sql_100_cosql_dev_100turns', 'strict_execution_accuracy'):.3f} "
                "strict accuracy and "
                f"{value('multiturn_sql_100_cosql_dev_100turns', 'value_execution_accuracy'):.3f} "
                "value accuracy."
            ),
            "source_claim_ids": (
                "qwen35_9b_base_cosql_dev_100turns; "
                "multiturn_sql_100_cosql_dev_100turns"
            ),
            "evidence_level": status("multiturn_sql_100_cosql_dev_100turns"),
            "missing_gate": "Same-protocol hosted baseline and BIRD-Interact transfer.",
            "current_decision": "Use as the baseline every structured target must beat.",
        },
        {
            "fine_tuning_target": "Planner/DSL first, SQL second",
            "lab_behavior": "Carries metric, grain, and filters before SQL, but still needs value grounding.",
            "manifest_backed_evidence": (
                "Planner quality is manifest-backed, not SQL-improvement backed: "
                f"macro={float(planner['macro_planner_score']):.3f}, "
                f"table_f1={float(planner['table_f1']):.3f}, "
                f"column_f1={float(planner['column_f1']):.3f}. "
                "The direct-vs-predicted input preflight is "
                f"{preflight['status']} on {preflight['row_count']} turns "
                f"across {preflight['dialog_count']} dialogs."
            ),
            "source_claim_ids": "planner_lexical_schema_baseline; predicted_planner_sql_execution",
            "evidence_level": (
                f"{status('planner_lexical_schema_baseline')} + "
                f"{status('predicted_planner_sql_execution')}"
            ),
            "missing_gate": (
                "Predicted-planner endpoint SQL manifest compared with direct SQL "
                "through eval.run_predicted_planner_comparison."
            ),
            "current_decision": (
                "Highest-priority next build target because it attacks the oracle "
                "gap directly; the paired runner now prevents row/model/scorer drift."
            ),
        },
        {
            "fine_tuning_target": "Semantic-layer tuning",
            "lab_behavior": "Normalizes entities and values before SQL, fixing France -> FR.",
            "manifest_backed_evidence": (
                "Best non-oracle semantic prompt result is manifest-backed on the proxy: "
                f"{value('semantic_prompt_minimal_executable_cosql_dev_100turns', 'value_execution_accuracy'):.3f} "
                "value accuracy."
            ),
            "source_claim_ids": "semantic_prompt_minimal_executable_cosql_dev_100turns",
            "evidence_level": status("semantic_prompt_minimal_executable_cosql_dev_100turns"),
            "missing_gate": "Versioned semantic artifacts, retrieval/pruning, and row-matched method comparison.",
            "current_decision": "Keep, but do not treat broad semantic context as a proven method win yet.",
        },
        {
            "fine_tuning_target": "MEASURE()-preserving metric DSL",
            "lab_behavior": "Keeps governed metric intent as MEASURE(revenue) before SQL expansion.",
            "manifest_backed_evidence": (
                "The bootstrap metric-DSL manifest is ledger-backed for parse, "
                "compile, execution, and measure preservation; it does not beat "
                "the direct-SQL control yet."
            ),
            "source_claim_ids": "metric-dsl-bootstrap.metric_dsl; metric_dsl_beats_direct_sql",
            "evidence_level": (
                f"{status('metric-dsl-bootstrap.metric_dsl')} + "
                f"{status('metric_dsl_beats_direct_sql')}"
            ),
            "missing_gate": "Positive metric-DSL value delta against direct SQL on the same rows.",
            "current_decision": "Promising for metric-heavy tasks; not rankable against direct SQL yet.",
        },
        {
            "fine_tuning_target": "Behavior/recovery tuning",
            "lab_behavior": "Uses empty-result feedback to repair a previous value-grounding error.",
            "manifest_backed_evidence": (
                "The claim ledger tracks rollout and recovery claims, but both are still pending."
            ),
            "source_claim_ids": "model_generated_history_rollout; rollout_beats_teacher_forced_history",
            "evidence_level": (
                f"{status('model_generated_history_rollout')} + "
                f"{status('rollout_beats_teacher_forced_history')}"
            ),
            "missing_gate": "Model-generated-history rollout compared with same-model teacher-forced history.",
            "current_decision": "Cannot be judged from teacher-forced CoSQL; needs rollout evaluation.",
        },
    ]
    return pd.DataFrame(rows)


def method_decision_rules() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fine_tuning_target": "Direct SQL SFT",
                "control_arm": "Base Qwen and 100-step direct-SQL LoRA",
                "win_condition": (
                    "Defines the baseline every structured target must beat; "
                    "not enough for a hosted-SOTA claim by itself."
                ),
                "required_comparison": (
                    "Compare on same rows, same scorer, same prompt boundary, "
                    "and same oracle policy before reporting movement."
                ),
                "current_blocker": (
                    "Hosted same-protocol baseline, a positive local-vs-hosted "
                    "delta, and BIRD-Interact transfer are pending, so direct-SQL "
                    "movement is only proxy evidence."
                ),
                "claim_ids": (
                    "qwen35_9b_base_cosql_dev_100turns, "
                    "multiturn_sql_100_cosql_dev_100turns, "
                    "hosted_sota_same_protocol, "
                    "local_beats_hosted_same_protocol, "
                    "bird_interact_local_vs_hosted"
                ),
            },
            {
                "fine_tuning_target": "Planner/DSL first, SQL second",
                "control_arm": "Direct SQL SFT on the fixed proxy slice",
                "win_condition": (
                    "A non-oracle planner must recover tables, joins, projection "
                    "shape, grain, duplicate policy, and enough values for SQL "
                    "generation to beat direct SQL execution."
                ),
                "required_comparison": (
                    "Compare predicted plans and generated SQL on same rows, same "
                    "scorer, same schema text, and no labels extracted from gold SQL."
                ),
                "current_blocker": (
                    "non-oracle planner-to-SQL execution is pending; current oracle "
                    "planner rows are ceilings, not production evidence. "
                    "eval.run_predicted_planner_comparison is the required paired "
                    "endpoint runner for the next manifest."
                ),
                "claim_ids": (
                    "planner_lexical_schema_baseline, "
                    "predicted_planner_sql_execution"
                ),
            },
            {
                "fine_tuning_target": "Semantic-layer tuning",
                "control_arm": "Direct SQL prompt plus unpruned schema context",
                "win_condition": (
                    "A semantic-state target must improve value grounding, entity "
                    "resolution, and grain choices without flooding the prompt."
                ),
                "required_comparison": (
                    "Compare semantic artifacts and final SQL on same rows, same "
                    "scorer, same latency/cost accounting, and versioned semantic "
                    "model snapshots."
                ),
                "current_blocker": (
                    "Value index, entity-resolution labels, and versioned semantic "
                    "model manifests are not complete."
                ),
                "claim_ids": (
                    "semantic_prompt_minimal_executable_cosql_dev_100turns, "
                    "hosted_sota_same_protocol, "
                    "local_beats_hosted_same_protocol"
                ),
            },
            {
                "fine_tuning_target": "MEASURE()-preserving metric DSL",
                "control_arm": "Direct SQL SFT on metric-heavy rows",
                "win_condition": (
                    "The model must preserve governed metric intent as MEASURE() "
                    "until compilation, then match or beat direct SQL value accuracy."
                ),
                "required_comparison": (
                    "Compare DSL parse, compile, measure preservation, and execution "
                    "on same rows, same scorer, same semantic model version, and the "
                    "same direct-SQL baseline."
                ),
                "current_blocker": (
                    "metric-DSL prediction manifest and direct-SQL delta are pending."
                ),
                "claim_ids": (
                    "metric-dsl-bootstrap.metric_dsl, "
                    "metric_dsl_beats_direct_sql"
                ),
            },
            {
                "fine_tuning_target": "Behavior/recovery tuning",
                "control_arm": "Teacher-forced history and direct SQL retry behavior",
                "win_condition": (
                    "The model must recover from its own prior empty results or bad "
                    "SQL, not only answer when the history is clean."
                ),
                "required_comparison": (
                    "Compare teacher-forced and generated-history rollouts on same "
                    "rows, same scorer, same model, and same stopping rules."
                ),
                "current_blocker": "generated-history rollout and recovery delta are pending.",
                "claim_ids": (
                    "model_generated_history_rollout, "
                    "rollout_beats_teacher_forced_history"
                ),
            },
        ]
    )


def method_priority_backlog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "priority": 1,
                "fine_tuning_target": "Planner/DSL first, SQL second",
                "why_now": (
                    "It attacks the oracle gap directly: the best diagnostic result "
                    "gets much easier when schema linking, joins, projection, and "
                    "grain are already planned."
                ),
                "dataset_focus": "CoSQL fixed proxy first, then BIRD-Interact transfer",
                "success_metric": (
                    "predicted planner label F1 plus generated SQL value accuracy "
                    "beating direct SQL on the same rows"
                ),
                "build_next": (
                    "Run eval.planner_optimize with DSPy proposals, then use "
                    "eval.planner_predict as the endpoint harness for the promoted "
                    "planner program without gold SQL labels. Execute the final "
                    "direct-vs-predicted pair through "
                    "eval.run_predicted_planner_comparison."
                ),
                "falsifies_if": (
                    "planner F1 improves but final SQL does not beat direct SQL on "
                    "the same scorer and oracle policy"
                ),
                "claim_ids": "planner_lexical_schema_baseline, predicted_planner_sql_execution",
            },
            {
                "priority": 2,
                "fine_tuning_target": "Semantic-layer tuning",
                "why_now": (
                    "The current lab and proxy failures show value/entity grounding "
                    "and grain errors that raw DDL does not explain."
                ),
                "dataset_focus": "CoSQL plus synthetic value/entity fixtures",
                "success_metric": (
                    "value execution accuracy, entity-resolution accuracy, and "
                    "fanout-safe grain labels on the same rows"
                ),
                "build_next": (
                    "versioned semantic artifacts, value/entity indexes, and retrieval "
                    "or pruning manifests before another broad semantic prompt run"
                ),
                "falsifies_if": (
                    "semantic context increases prompt size or latency without reducing "
                    "value grounding and grain errors"
                ),
                "claim_ids": (
                    "semantic_prompt_minimal_executable_cosql_dev_100turns, "
                    "hosted_sota_same_protocol"
                ),
            },
            {
                "priority": 3,
                "fine_tuning_target": "MEASURE()-preserving metric DSL",
                "why_now": (
                    "It is the cleanest way to test whether governed metric intent "
                    "should be learned before SQL expansion."
                ),
                "dataset_focus": "synthetic schema-rich SQL and metric-heavy CoSQL/SParC rows",
                "success_metric": (
                    "metric_dsl_parse_rate, compile_rate, measure_preservation, and "
                    "compiled SQL value accuracy versus direct SQL"
                ),
                "build_next": (
                    "metric-DSL prediction manifest with the same semantic model "
                    "version and a row-matched direct-SQL baseline"
                ),
                "falsifies_if": (
                    "MEASURE() preservation is high but compiled execution does not "
                    "match or beat direct SQL"
                ),
                "claim_ids": "metric-dsl-bootstrap.metric_dsl, metric_dsl_beats_direct_sql",
            },
            {
                "priority": 4,
                "fine_tuning_target": "Behavior/recovery tuning",
                "why_now": (
                    "A real multi-turn analyst must live with its own earlier SQL and "
                    "result state, but the current proxy still uses teacher-forced history."
                ),
                "dataset_focus": "CoSQL generated-history rollout, then BIRD-Interact",
                "success_metric": (
                    "model-generated-history value accuracy and recovery-success delta "
                    "over teacher-forced evaluation"
                ),
                "build_next": (
                    "generated-history rollout manifest with empty-result repair, "
                    "clarification, and retry labels"
                ),
                "falsifies_if": (
                    "the model only works with clean teacher-forced history and collapses "
                    "after its own first wrong turn"
                ),
                "claim_ids": (
                    "model_generated_history_rollout, "
                    "rollout_beats_teacher_forced_history"
                ),
            },
            {
                "priority": 5,
                "fine_tuning_target": "Direct SQL SFT",
                "why_now": (
                    "It remains the control arm, not the most interesting next bet; "
                    "every structured method must beat it before claiming improvement."
                ),
                "dataset_focus": "fixed CoSQL proxy, hosted baselines, and BIRD-Interact",
                "success_metric": (
                    "strict and value execution accuracy under the same scorer, rows, "
                    "prompt boundary, and oracle policy"
                ),
                "build_next": (
                    "same-protocol hosted baseline, local-vs-hosted comparison, and "
                    "BIRD-Interact transfer so the direct-SQL control is not just a "
                    "local proxy number"
                ),
                "falsifies_if": (
                    "a more structured target cannot beat the direct-SQL control on "
                    "identical rows"
                ),
                "claim_ids": (
                    "qwen35_9b_base_cosql_dev_100turns, "
                    "multiturn_sql_100_cosql_dev_100turns, "
                    "hosted_sota_same_protocol, "
                    "local_beats_hosted_same_protocol, "
                    "bird_interact_local_vs_hosted"
                ),
            },
        ]
    )


def endpoint_run_scorecard() -> pd.DataFrame:
    summary = read_csv_artifact(
        "plots/rescored_vllm_semantic_prompt_iteration_100turns/summary.csv"
    )
    labels = {
        "unsloth/Qwen3.5-9B": "Base Qwen 3.5 9B",
        "multiturn-sql-50": "50-step LoRA",
        "multiturn-sql-100": "100-step LoRA",
        "multiturn-sql-semantic-50": "Semantic 50-step LoRA",
    }

    rows: list[dict[str, str]] = []
    for _, row in summary.iterrows():
        model_name = str(row["model_name"])
        prompt_variant = "" if pd.isna(row["prompt_variant"]) else str(row["prompt_variant"])
        label = labels.get(model_name, model_name)
        if prompt_variant:
            label = f"{label} + {prompt_variant}"
        rows.append(
            {
                "run": label,
                "prompt_variant": prompt_variant or "default",
                "strict_accuracy": f"{float(row['strict_accuracy']):.3f}",
                "value_accuracy": f"{float(row['value_accuracy']):.3f}",
                "syntax_accuracy": f"{float(row['syntax_accuracy']):.3f}",
                "mean_latency_ms": f"{float(row['mean_latency_ms']):.1f}",
            }
        )

    order = {
        "Base Qwen 3.5 9B": 0,
        "50-step LoRA": 1,
        "100-step LoRA": 2,
        "Semantic 50-step LoRA": 3,
        "Semantic 50-step LoRA + semantic_grounding": 4,
        "Semantic 50-step LoRA + minimal_executable": 5,
    }
    return pd.DataFrame(sorted(rows, key=lambda item: order.get(item["run"], 99)))


def _format_error_counts(value: Any) -> str:
    if isinstance(value, str):
        counts = json.loads(value)
    elif isinstance(value, dict):
        counts = value
    else:
        counts = {}
    if not counts:
        return "none"
    return ", ".join(
        f"{str(label).replace('_', ' ')} {int(count)}"
        for label, count in sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    )


def failure_taxonomy_delta() -> pd.DataFrame:
    """Summarize what non-oracle methods fixed and regressed on the proxy slice."""

    summary_path = "plots/failure_taxonomy/comparison/model_error_summary.csv"
    pairwise_path = "plots/failure_taxonomy/comparison/pairwise_vs_baseline.csv"
    summary = read_csv_artifact(summary_path).set_index("run")
    pairwise = read_csv_artifact(pairwise_path)
    rows: list[dict[str, Any]] = []
    for _, row in pairwise.iterrows():
        candidate = str(row["candidate"])
        fixed = _format_error_counts(row["fixed_error_primary"])
        regressed = _format_error_counts(row["regressed_error_primary"])
        net_fixed = int(row["net_fixed"])
        rows.append(
            {
                "candidate": candidate,
                "baseline": str(row["baseline"]),
                "value_accuracy": float(summary.loc[candidate, "value_accuracy"]),
                "fixed_turns": int(row["fixed_turns"]),
                "regressed_turns": int(row["regressed_turns"]),
                "net_fixed": net_fixed,
                "fixed_error_primary": fixed,
                "regressed_error_primary": regressed,
                "takeaway": (
                    f"Net {net_fixed:+d}: fixes concentrate in {fixed}; "
                    f"regressions concentrate in {regressed}. This turns the "
                    "score movement into a training-target diagnostic."
                ),
            }
        )
    return pd.DataFrame(
        sorted(rows, key=lambda item: (-item["net_fixed"], -item["value_accuracy"], item["candidate"]))
    )


def schema_validation_findings() -> pd.DataFrame:
    """Run pre-execution schema diagnostics over representative result files."""

    artifacts = [
        (
            "Base Qwen 3.5 9B",
            "results/rescored/vllm_qwen35_9b_base_cosql_dev_100turns.jsonl",
        ),
        (
            "100-step LoRA",
            "results/rescored/vllm_qwen35_9b_lora100_cosql_dev_100turns.jsonl",
        ),
        (
            "Semantic 50-step + minimal executable",
            "results/rescored/minimal_executable.jsonl",
        ),
    ]
    rows = []
    for label, relative_path in artifacts:
        counts: Counter[str] = Counter()
        mismatch_rows = 0
        first_example: tuple[str, dict[str, Any]] | None = None
        for row in read_jsonl_artifact(relative_path):
            diagnostics = validate_sql_against_visible_schema(
                str(row.get("generated_sql", "")),
                row.get("messages", []),
            )
            if not diagnostics["schema_validation_errors"]:
                continue
            mismatch_rows += 1
            counts.update(diagnostics["schema_validation_errors"])
            if first_example is None or (
                not first_example[1]["wrong_table_columns"] and diagnostics["wrong_table_columns"]
            ):
                first_example = (str(row.get("id", "")), diagnostics)

        example_turn = ""
        example_diagnostic = ""
        if first_example is not None:
            example_turn, diagnostics = first_example
            details = []
            for key in (
                "wrong_table_columns",
                "unknown_columns",
                "ambiguous_unqualified_columns",
                "unknown_tables",
            ):
                if diagnostics[key]:
                    details.append(f"{key}={', '.join(diagnostics[key])}")
            example_diagnostic = (
                "repairable: "
                + ", ".join(diagnostics["schema_validation_errors"])
                + ("; " + "; ".join(details) if details else "")
            )

        rows.append(
            {
                "run": label,
                "source_artifact": relative_path,
                "schema_mismatch_rows": mismatch_rows,
                "unknown_column": int(counts["unknown_column"]),
                "wrong_table_column": int(counts["wrong_table_column"]),
                "ambiguous_unqualified_column": int(counts["ambiguous_unqualified_column"]),
                "unknown_table": int(counts["unknown_table"]),
                "example_turn": example_turn,
                "example_diagnostic": example_diagnostic,
            }
        )
    return pd.DataFrame(rows)


def _write_text(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return str(path.name)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list | tuple | dict):
        return json.dumps(value, sort_keys=True)
    if pd.isna(value):
        return ""
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        cells = [_fmt(row[column]).replace("\n", " ") for column in frame.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _bar_svg(
    frame: pd.DataFrame,
    *,
    title: str,
    subtitle: str,
    label_column: str,
    value_column: str,
    width: int = 900,
    height: int = 420,
) -> str:
    chart_left = 78
    chart_top = 94
    chart_bottom = height - 88
    chart_width = width - chart_left - 52
    chart_height = chart_bottom - chart_top
    bar_gap = 26
    bar_width = max(34, int((chart_width - bar_gap * (len(frame) - 1)) / len(frame)))
    max_value = max(1.0, float(frame[value_column].max()))
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img">',
        f"<title>{html.escape(title)}</title>",
        '<rect width="100%" height="100%" fill="#fff"/>',
        (
            f'<text x="34" y="38" font-family="Arial, sans-serif" '
            f'font-size="24" font-weight="700" fill="#111">{html.escape(title)}</text>'
        ),
        (
            f'<text x="34" y="64" font-family="Arial, sans-serif" '
            f'font-size="14" fill="#555">{html.escape(subtitle)}</text>'
        ),
        (
            f'<line x1="{chart_left}" y1="{chart_bottom}" '
            f'x2="{width - 36}" y2="{chart_bottom}" stroke="#222"/>'
        ),
        (
            f'<line x1="{chart_left}" y1="{chart_top}" '
            f'x2="{chart_left}" y2="{chart_bottom}" stroke="#222"/>'
        ),
    ]
    for tick in (0.25, 0.5, 0.75, 1.0):
        y = chart_bottom - int(chart_height * tick / max_value)
        parts.append(
            f'<line x1="{chart_left}" y1="{y}" x2="{width - 36}" y2="{y}" '
            'stroke="#e8e8e8"/>'
        )
        parts.append(
            f'<text x="34" y="{y + 4}" font-family="Arial, sans-serif" '
            f'font-size="12" fill="#777">{tick:.2f}</text>'
        )

    for index, row in frame.reset_index(drop=True).iterrows():
        value = float(row[value_column])
        x = chart_left + 34 + index * (bar_width + bar_gap)
        bar_height = int(chart_height * value / max_value)
        y = chart_bottom - bar_height
        label = str(row[label_column])
        fill = "#222" if index < 3 else "#bdbdbd"
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{fill}"/>',
                (
                    f'<text x="{x}" y="{y - 8}" font-family="Arial, sans-serif" '
                    f'font-size="13" font-weight="700" fill="#111">{value:.3f}</text>'
                ),
                (
                    f'<text x="{x}" y="{chart_bottom + 24}" font-family="Arial, sans-serif" '
                    f'font-size="12" fill="#222">{html.escape(label[:24])}</text>'
                ),
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def export_blog_evidence(output_dir: Path | str = Path("docs/blog/generated")) -> dict[str, Any]:
    """Write publishable notebook-backed evidence assets for the blog post."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for retired_asset in (
        "notebook-contracts.md",
        "notebook-series.md",
        "notebook-walkthrough.md",
    ):
        retired_path = output / retired_asset
        if retired_path.exists():
            retired_path.unlink()

    scores = accuracy_scorecard()
    planner = planner_scorecard()
    planner_readiness = planner_readiness_summary()
    claims = claim_table()
    dataset_roles = dataset_role_matrix()
    gap = single_to_multiturn_gap()
    hosted_protocol = hosted_comparison_protocol()
    harness = evaluation_harness_map()
    method_readiness = method_readiness_report()
    finetuning_steps = finetuning_step_plan()
    metric_contract = metric_dsl_eval_contract()
    shareable_lab = shareable_lab_attachment()
    reader_flow = lab_reader_flow()
    ladder = experiment_ladder()
    decision_rules = method_decision_rules()
    method_priority = method_priority_backlog()
    lab_scores = lab_method_scorecard()
    lab_trace = lab_failure_trace()
    data_gates = data_engineering_gates()
    synthetic_fixtures = synthetic_method_fixture_summary()
    value_labels = value_grounding_label_summary()
    value_index = value_index_summary()
    artifact_contract = data_artifact_contract()
    prompt_findings = prompt_optimization_findings()
    targets = target_comparison()
    target_evidence = target_evidence_matrix()
    endpoint_runs = endpoint_run_scorecard()
    failure_delta = failure_taxonomy_delta()
    schema_findings = schema_validation_findings()

    assets = {
        "accuracy_ladder_svg": _write_text(
            output / "accuracy-ladder.svg",
            _bar_svg(
                scores,
                title="Proxy SQL accuracy ladder",
                subtitle="Fixed 100-turn CoSQL slice. Grey bars are oracle diagnostics.",
                label_column="run",
                value_column="score",
            ),
        ),
        "planner_baseline_svg": _write_text(
            output / "planner-baseline.svg",
            _bar_svg(
                planner,
                title="Planner baseline scores",
                subtitle="Lexical non-oracle planner on the same CoSQL proxy slice.",
                label_column="metric",
                value_column="score",
            ),
        ),
        "planner_readiness_md": _write_text(
            output / "planner-readiness.md",
            _markdown_table(planner_readiness),
        ),
        "claim_table_md": _write_text(output / "claim-table.md", _markdown_table(claims)),
        "dataset_role_matrix_md": _write_text(
            output / "dataset-role-matrix.md",
            _markdown_table(dataset_roles),
        ),
        "single_to_multiturn_gap_md": _write_text(
            output / "single-to-multiturn-gap.md",
            _markdown_table(gap),
        ),
        "hosted_comparison_protocol_md": _write_text(
            output / "hosted-comparison-protocol.md",
            _markdown_table(hosted_protocol),
        ),
        "evaluation_harness_map_md": _write_text(
            output / "evaluation-harness-map.md",
            _markdown_table(harness),
        ),
        "method_readiness_report_md": _write_text(
            output / "method-readiness-report.md",
            _markdown_table(method_readiness),
        ),
        "finetuning_step_plan_md": _write_text(
            output / "finetuning-step-plan.md",
            _markdown_table(finetuning_steps),
        ),
        "metric_dsl_contract_md": _write_text(
            output / "metric-dsl-contract.md",
            _markdown_table(metric_contract),
        ),
        "shareable_lab_md": _write_text(
            output / "shareable-lab.md",
            _markdown_table(shareable_lab),
        ),
        "lab_reader_flow_md": _write_text(
            output / "lab-reader-flow.md",
            _markdown_table(reader_flow),
        ),
        "experiment_ladder_md": _write_text(
            output / "experiment-ladder.md",
            _markdown_table(ladder),
        ),
        "method_decision_rules_md": _write_text(
            output / "method-decision-rules.md",
            _markdown_table(decision_rules),
        ),
        "method_priority_backlog_md": _write_text(
            output / "method-priority-backlog.md",
            _markdown_table(method_priority),
        ),
        "lab_method_scores_md": _write_text(
            output / "lab-method-scores.md",
            _markdown_table(lab_scores),
        ),
        "lab_failure_trace_md": _write_text(
            output / "lab-failure-trace.md",
            _markdown_table(lab_trace),
        ),
        "data_engineering_gates_md": _write_text(
            output / "data-engineering-gates.md",
            _markdown_table(data_gates),
        ),
        "synthetic_method_fixtures_md": _write_text(
            output / "synthetic-method-fixtures.md",
            _markdown_table(synthetic_fixtures),
        ),
        "value_grounding_labels_md": _write_text(
            output / "value-grounding-labels.md",
            _markdown_table(value_labels),
        ),
        "value_index_md": _write_text(
            output / "value-index.md",
            _markdown_table(value_index),
        ),
        "data_artifact_contract_md": _write_text(
            output / "data-artifact-contract.md",
            _markdown_table(artifact_contract),
        ),
        "prompt_optimization_findings_md": _write_text(
            output / "prompt-optimization-findings.md",
            _markdown_table(prompt_findings),
        ),
        "target_comparison_md": _write_text(
            output / "target-comparison.md",
            _markdown_table(targets),
        ),
        "target_evidence_matrix_md": _write_text(
            output / "target-evidence-matrix.md",
            _markdown_table(target_evidence),
        ),
        "endpoint_run_scorecard_md": _write_text(
            output / "endpoint-run-scorecard.md",
            _markdown_table(endpoint_runs),
        ),
        "failure_taxonomy_delta_md": _write_text(
            output / "failure-taxonomy-delta.md",
            _markdown_table(failure_delta),
        ),
        "schema_validation_findings_md": _write_text(
            output / "schema-validation-findings.md",
            _markdown_table(schema_findings),
        ),
    }
    asset_contracts = [
        {
            "id": asset_id,
            "path": asset_path,
            "kind": "svg" if asset_path.endswith(".svg") else "markdown",
            "required_in_post": asset_id in ASSET_REQUIRED_IN_POST,
            "required_reference": asset_id in ASSET_REQUIRED_REFERENCE,
            "sha256": sha256_file(output / asset_path),
            "claim_ids": list(ASSET_CLAIM_IDS.get(asset_id, ())),
        }
        for asset_id, asset_path in assets.items()
    ]
    source_contracts = [
        {
            "path": source_path,
            "sha256": sha256_file(artifact_path(source_path)),
        }
        for source_path in BLOG_EVIDENCE_SOURCES
    ]
    claim_snapshot = {
        str(row["claim_id"]): {
            str(field): _manifest_json_value(value)
            for field, value in row.items()
        }
        for row in read_jsonl_artifact("docs/claim_ledgers/cosql_dev_100.jsonl")
    }
    manifest = {
        "schema_version": 2,
        "source_repo": "multiturn-sql-finetuning",
        "post_slug": BLOG_POST_SLUG,
        "source_artifacts": source_contracts,
        "assets": asset_contracts,
        "claim_snapshot": claim_snapshot,
        "forbidden_public_substrings": list(FORBIDDEN_PUBLIC_SUBSTRINGS),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("docs/blog/generated"))
    args = parser.parse_args()
    manifest = export_blog_evidence(args.output_dir)
    print(f"Wrote {len(manifest['assets'])} blog evidence assets to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
