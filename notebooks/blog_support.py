"""Shared data loaders for the blog companion lab and generated evidence."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.metric_dsl import compile_metric_query, parse_metric_query, score_metric_query
from notebooks.labs.local_multiturn_sql_lab_support import run_multiturn_lab

BLOG_EVIDENCE_SOURCES = (
    "docs/claim_ledgers/cosql_dev_100.jsonl",
    "docs/planner_baseline_cosql_dev_100_summary.json",
    "plots/rescored_vllm_semantic_prompt_iteration_100turns/summary.csv",
    "results/prompt_search_semantic50_limit30/summary.csv",
    "results/prompt_search_schema_pruned_projection_schemafix_100/summary.csv",
)


LAB_NOTEBOOK = "notebooks/labs/local_multiturn_sql_lab.ipynb"
LAB_APP = "notebooks/labs/local_multiturn_sql_lab.py"

LAB_READER_SECTIONS: tuple[dict[str, str], ...] = (
    {
        "lab_step": "Benchmark gap",
        "reader_action": (
            "Research question: start in the lab notebook with the failure "
            "trace: single-turn SQL can look solved while a follow-up loses state."
        ),
        "evidence_to_inspect": "lab-failure-trace.md",
        "purpose": (
            "Turns the BIRD-style zero-shot premise into a concrete multi-turn "
            "failure before introducing any fine-tuning result."
        ),
        "claim_boundary": (
            "Notebook demonstration only; it is not a benchmark result and "
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
                "metric_dsl_evaluation_manifest",
                "metric_dsl_beats_direct_sql",
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
                "current_status": "design target for the next data-artifact buildout",
                "next_artifact": "fixture pack with value indexes, fanout cases, and MEASURE() labels",
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


def shareable_lab_attachment() -> pd.DataFrame:
    reader_flow = " -> ".join(
        [
            "Research question",
            "open the lab notebook",
            "run the lab sections",
            "inspect the failure trace",
            "compare fine-tuning targets",
            "read the evidence gates",
        ]
    )
    return pd.DataFrame(
        [
            {
                "artifact": "shareable lab notebook and attached codebase",
                "notebook": LAB_NOTEBOOK,
                "repo_url": "https://github.com/xdanny/multiturn-sql-finetuning",
                "run_command": f"jupyter lab {LAB_NOTEBOOK}",
                "alternate_command": f"marimo edit {LAB_APP}",
                "device_policy": (
                    "The lab auto-selects CUDA, MPS, or XPU when PyTorch detects "
                    "an available accelerator and falls back to CPU."
                ),
                "reader_flow": reader_flow,
                "what_runs": (
                    "One compact notebook runs the SQLite scenario, compares direct "
                    "SQL, planner-first, semantic-layer, MEASURE()-preserving DSL, "
                    "and behavior/recovery targets, then connects those behaviors "
                    "to generated repo evidence."
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
                "notebook": LAB_NOTEBOOK,
                "run_command": f"jupyter lab {LAB_NOTEBOOK}",
                "alternate_command": f"marimo edit {LAB_APP}",
                "reader_action": section["reader_action"],
                "evidence_to_inspect": section["evidence_to_inspect"],
                "purpose": section["purpose"],
                "claim_boundary": section["claim_boundary"],
            }
        )
    return pd.DataFrame(rows)


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
        ("turn_4", "behavior_recovery_sql"): (
            "The recovery target uses the empty-result signal to repair the previous turn."
        ),
    }
    rows = []
    for row in report["rows"]:
        key = (row["turn_id"], row["system"])
        if key not in selected:
            continue
        rows.append(
            {
                "turn_id": row["turn_id"],
                "question": row["question"],
                "system": row["system"],
                "failure_type": row["failure_type"] or "recovery_success",
                "value_match": row["value_match"],
                "actual_rows": row["actual_rows"],
                "expected_rows": row["expected_rows"],
                "intermediate_plan": row["intermediate_plan"],
                "why_it_matters": why_by_key[key],
            }
        )
    return pd.DataFrame(rows)


def data_engineering_gates() -> pd.DataFrame:
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
                    "data/processed/eval_100_each.jsonl; "
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
                "current_status": "pending: isolated in the lab, not yet a dataset artifact",
                "next_repo_action": (
                    "Build per-database value indexes and label entity resolutions "
                    "for CoSQL/SParC rows before training."
                ),
                "blocks_claim": "Blocks semantic grounding and recovery claims.",
                "source_artifacts": (
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
                "current_status": "pending: listed as a failure class, not yet fixture-backed",
                "next_repo_action": (
                    "Add duplicated-child and bridge-table cases with expected "
                    "duplicate-row policy labels."
                ),
                "blocks_claim": "Blocks trustworthy analytical metric claims.",
                "source_artifacts": (
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
                "current_status": "pending: planner metrics exist, validator is not complete",
                "next_repo_action": (
                    "Validate predicted columns against table roles and emit a "
                    "repairable failure label before execution scoring."
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
                "current_status": "partial: metric-DSL parser/evaluator exists; model predictions pending",
                "next_repo_action": (
                    "Generate metric-DSL predictions, compile them through the same "
                    "semantic model, and compare against direct SQL."
                ),
                "blocks_claim": "Blocks MEASURE()-first or semantic-layer superiority claims.",
                "source_artifacts": "docs/claim_ledgers/cosql_dev_100.jsonl",
                "claim_ids": (
                    "metric_dsl_evaluation_manifest, "
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
                "current_status": "pending: claim ledger tracks the missing hosted baseline",
                "next_repo_action": (
                    "Run hosted baselines through the fixed CoSQL proxy and publish "
                    "the same execution manifest fields."
                ),
                "blocks_claim": "Blocks hosted-SOTA comparison claims.",
                "source_artifacts": "docs/claim_ledgers/cosql_dev_100.jsonl",
                "claim_ids": "hosted_sota_same_protocol",
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
                "pending: optimize a two-stage planner-to-SQL program before "
                "using DSPy as evidence for a method claim."
            ),
            "next_program_target": (
                "Optimize planner label F1, value accuracy, and failure-taxonomy "
                "deltas on a development split before endpoint promotion; see "
                "the shareable lab notebook and planner evaluation docs."
            ),
            "claim_boundary": "Planner program gate is not run yet; not a SOTA claim.",
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
                    f"accuracy; best non-oracle prompt reached {best_non_oracle:.3f} "
                    "value accuracy on the fixed CoSQL slice."
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
                f"column_f1={float(planner['column_f1']):.3f}."
            ),
            "source_claim_ids": "planner_lexical_schema_baseline; predicted_planner_sql_execution",
            "evidence_level": (
                f"{status('planner_lexical_schema_baseline')} + "
                f"{status('predicted_planner_sql_execution')}"
            ),
            "missing_gate": "Predicted-planner endpoint SQL manifest compared with direct SQL.",
            "current_decision": "Highest-priority next build target because it attacks the oracle gap directly.",
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
                "Parser, compiler, and offline evaluator exist, but the claim ledger "
                "has no valid metric-DSL prediction/comparison manifest yet."
            ),
            "source_claim_ids": "metric_dsl_evaluation_manifest; metric_dsl_beats_direct_sql",
            "evidence_level": (
                f"{status('metric_dsl_evaluation_manifest')} + "
                f"{status('metric_dsl_beats_direct_sql')}"
            ),
            "missing_gate": "Metric-DSL prediction manifest plus direct-SQL comparison on the same rows.",
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
                    "Hosted same-protocol baseline and BIRD-Interact transfer are "
                    "pending, so direct-SQL movement is only proxy evidence."
                ),
                "claim_ids": (
                    "qwen35_9b_base_cosql_dev_100turns, "
                    "multiturn_sql_100_cosql_dev_100turns, "
                    "hosted_sota_same_protocol, "
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
                    "planner rows are ceilings, not production evidence."
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
                    "hosted_sota_same_protocol"
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
                    "metric_dsl_evaluation_manifest, "
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
                    "DSPy planner program that predicts tables, columns, joins, "
                    "projection shape, duplicate policy, and value candidates without "
                    "gold SQL labels."
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
                "claim_ids": "metric_dsl_evaluation_manifest, metric_dsl_beats_direct_sql",
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
                    "same-protocol hosted baseline and BIRD-Interact transfer so the "
                    "direct-SQL control is not just a local proxy number"
                ),
                "falsifies_if": (
                    "a more structured target cannot beat the direct-SQL control on "
                    "identical rows"
                ),
                "claim_ids": (
                    "qwen35_9b_base_cosql_dev_100turns, "
                    "multiturn_sql_100_cosql_dev_100turns, "
                    "hosted_sota_same_protocol, "
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
    claims = claim_table()
    dataset_roles = dataset_role_matrix()
    metric_contract = metric_dsl_eval_contract()
    shareable_lab = shareable_lab_attachment()
    reader_flow = lab_reader_flow()
    decision_rules = method_decision_rules()
    method_priority = method_priority_backlog()
    lab_scores = lab_method_scorecard()
    lab_trace = lab_failure_trace()
    data_gates = data_engineering_gates()
    prompt_findings = prompt_optimization_findings()
    targets = target_comparison()
    target_evidence = target_evidence_matrix()
    endpoint_runs = endpoint_run_scorecard()

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
        "claim_table_md": _write_text(output / "claim-table.md", _markdown_table(claims)),
        "dataset_role_matrix_md": _write_text(
            output / "dataset-role-matrix.md",
            _markdown_table(dataset_roles),
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
    }
    manifest = {
        "schema_version": 1,
        "source_repo": "multiturn-sql-finetuning",
        "source_artifacts": list(BLOG_EVIDENCE_SOURCES),
        "assets": assets,
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
