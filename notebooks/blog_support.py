"""Shared data loaders for the blog companion marimo notebooks."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from data.metric_dsl import compile_metric_query, parse_metric_query, score_metric_query

BLOG_EVIDENCE_SOURCES = (
    "docs/claim_ledgers/cosql_dev_100.jsonl",
    "docs/planner_baseline_cosql_dev_100_summary.json",
    "plots/rescored_vllm_semantic_prompt_iteration_100turns/summary.csv",
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


def notebook_walkthrough() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "checkpoint": "00 runnable lab",
                "reader_question": "Can a tiny multi-turn warehouse expose why single-turn SQL skill is not enough?",
                "notebook": "notebooks/labs/local_multiturn_sql_lab.py",
                "evidence_output": "Method matrix, subtask scores, behavior trace, generated plans and SQL",
                "claim_boundary": "Explains candidate fine-tuning targets; not a benchmark result.",
            },
            {
                "checkpoint": "01 problem and result",
                "reader_question": "Why does single-turn BIRD-style progress leave the multi-turn data-analysis claim unresolved?",
                "notebook": "notebooks/blog/01_problem_and_result.py",
                "evidence_output": "Accuracy ladder, claim ledger, planner baseline",
                "claim_boundary": "Separates non-oracle proxy results from oracle diagnostics and future hosted/BIRD-Interact claims.",
            },
            {
                "checkpoint": "02 local loop",
                "reader_question": "What local serving and evaluation contract makes adapter comparisons repeatable?",
                "notebook": "notebooks/blog/02_wsl_5090_setup.py",
                "evidence_output": "Training, serving, and evaluation environment contract",
                "claim_boundary": "Documents reproducibility constraints; does not make a model-quality claim.",
            },
            {
                "checkpoint": "03 data and eval",
                "reader_question": "What do CoSQL, SParC, BIRD-style rows, and synthetic schema-rich SQL contribute?",
                "notebook": "notebooks/blog/03_data_and_eval.py",
                "evidence_output": "Dataset role table and fixed CoSQL proxy manifest",
                "claim_boundary": "Treats CoSQL as a proxy slice, not as the final interactive benchmark.",
            },
            {
                "checkpoint": "04 fine-tuning targets",
                "reader_question": "Which target should the small model learn: direct SQL, planner-first SQL, semantic state, or DSL first?",
                "notebook": "notebooks/blog/04_training_iterations.py",
                "evidence_output": "Strategy table, metric DSL demo, strict runs, value-aware rescoring",
                "claim_boundary": "Shows current proxy movement and pending method comparisons; no DSL-first win is claimed yet.",
            },
            {
                "checkpoint": "05 serving tradeoffs",
                "reader_question": "How do endpoint latency, prompt length, and LoRA serving constraints affect the comparison?",
                "notebook": "notebooks/blog/05_vllm_blackwell_deep_dive.py",
                "evidence_output": "Latency and value-accuracy scatter from tracked summaries",
                "claim_boundary": "Keeps environment and latency effects visible before interpreting model quality.",
            },
            {
                "checkpoint": "06 data engineering agenda",
                "reader_question": "What data artifacts must exist before a multi-turn local model can credibly beat hosted SOTA?",
                "notebook": "notebooks/blog/06_data_engineering_for_multiturn_sql_eval.py",
                "evidence_output": "Planner metrics, semantic artifact backlog, MEASURE() contract, failure taxonomy",
                "claim_boundary": "Defines the next evidence gates: value index, entity resolution, fanout fixtures, MEASURE() preservation, rollout recovery.",
            },
        ]
    )


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
                "current_evidence": "Teacher-forced CoSQL history is supported; generated-history rollout improvement is still pending.",
                "claim_status": rollout_status,
                "next_gate": "Run model-generated-history rollout and compare it with teacher-forced history for the same model.",
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

    scores = accuracy_scorecard()
    planner = planner_scorecard()
    claims = claim_table()
    metric_contract = metric_dsl_eval_contract()
    walkthrough = notebook_walkthrough()
    targets = target_comparison()
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
        "metric_dsl_contract_md": _write_text(
            output / "metric-dsl-contract.md",
            _markdown_table(metric_contract),
        ),
        "notebook_walkthrough_md": _write_text(
            output / "notebook-walkthrough.md",
            _markdown_table(walkthrough),
        ),
        "target_comparison_md": _write_text(
            output / "target-comparison.md",
            _markdown_table(targets),
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
