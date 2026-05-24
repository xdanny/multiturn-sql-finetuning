"""
Search semantic SQL prompt variants against the existing execution evaluator.

The optimization target is intentionally the same metric used by endpoint eval:
execution accuracy on prepared SQL benchmark records. DSPy is optional and is
used to propose candidate prompt addenda; deterministic execution scoring still
decides which prompt wins.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from eval.ragas_metrics import extract_sql, score_single_turn
from eval.run_eval import (
    database_path_for_record,
    enforce_sql_only_instruction,
    generate_sql,
    load_benchmark_records,
    write_results,
)


@dataclass(frozen=True)
class PromptVariant:
    name: str
    instruction: str
    source: str = "static"
    requires_planning_hints: bool = False


DEFAULT_PROMPT_VARIANTS = [
    PromptVariant(
        name="baseline",
        instruction="Use the schema and conversation history to answer the current turn.",
    ),
    PromptVariant(
        name="semantic_grounding",
        instruction=(
            "Before choosing SQL objects, map the user request to the semantic model: identify the "
            "relevant cube, metric or count, dimensions, filters, and join hints. Then emit only the "
            "final SQL over physical table and column names."
        ),
    ),
    PromptVariant(
        name="history_delta",
        instruction=(
            "Treat each follow-up as a delta from the prior turns. Preserve earlier filters, entities, "
            "and grouping only when the current question refers to them; replace them when the user "
            "changes the target."
        ),
    ),
    PromptVariant(
        name="join_and_grain",
        instruction=(
            "Prefer join paths listed in the semantic model. Check the query grain before aggregating: "
            "count rows for entity counts, use SUM or AVG only for explicit numeric measures, and avoid "
            "joins that would duplicate the measured entity."
        ),
    ),
    PromptVariant(
        name="minimal_executable",
        instruction=(
            "Generate the simplest executable SQLite query that answers the current turn. Use only "
            "tables and columns present in the schema, and avoid extra joins or selected columns unless "
            "the question requires them."
        ),
    ),
]


def apply_prompt_variant(
    messages: list[dict[str, str]],
    variant: PromptVariant,
) -> list[dict[str, str]]:
    """Append a prompt variant to the system message without changing references."""

    updated = [dict(message) for message in messages]
    addendum = f"Additional prompt policy ({variant.name}): {variant.instruction}"
    for message in updated:
        if message.get("role") == "system":
            message["content"] = f"{message['content']}\n\n{addendum}"
            return updated
    return [{"role": "system", "content": addendum}, *updated]


def load_prompt_variants(path: Path | None) -> list[PromptVariant]:
    if path is None:
        return list(DEFAULT_PROMPT_VARIANTS)
    rows = json.loads(path.read_text())
    return [
        PromptVariant(
            name=str(row["name"]),
            instruction=str(row["instruction"]),
            source=str(row.get("source") or "file"),
            requires_planning_hints=bool(row.get("requires_planning_hints", False)),
        )
        for row in rows
    ]


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "accuracy": 0.0,
            "syntax_accuracy": 0.0,
            "mean_latency_ms": 0.0,
            "samples": 0,
        }
    return {
        "accuracy": sum(float(row["execution_score"]) for row in rows) / len(rows),
        "syntax_accuracy": sum(bool(row["syntax_valid"]) for row in rows) / len(rows),
        "mean_latency_ms": sum(float(row["generation_latency_ms"]) for row in rows) / len(rows),
        "samples": len(rows),
    }


def evaluate_variant(
    *,
    client: OpenAI,
    records: list[dict[str, Any]],
    variant: PromptVariant,
    model_name: str,
    database_root: Path | None,
    temperature: float,
    max_tokens: int,
) -> list[dict[str, Any]]:
    results = []
    for record in records:
        variant_messages = apply_prompt_variant(record["messages"], variant)
        raw_generation, generation_latency_ms = generate_sql(
            client,
            model_name=model_name,
            messages=enforce_sql_only_instruction(variant_messages),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        generated_sql = extract_sql(raw_generation)
        database_path = database_path_for_record(record, database_root)
        score = score_single_turn(record["reference_sql"], generated_sql, database_path=database_path)
        results.append(
            {
                **record,
                "model_name": model_name,
                "prompt_variant": variant.name,
                "prompt_variant_source": variant.source,
                "prompt_variant_requires_planning_hints": variant.requires_planning_hints,
                "prompt_instruction": variant.instruction,
                "raw_generation": raw_generation,
                "generated_sql": generated_sql,
                "generation_latency_ms": generation_latency_ms,
                "execution_score": score.execution_score,
                "strict_execution_score": score.strict_execution_score,
                "value_execution_score": score.value_execution_score,
                "order_sensitive": score.order_sensitive,
                "normalized_match": score.normalized_match,
                "syntax_valid": score.syntax_valid,
                "score_error": score.error,
                "database_path": str(database_path) if database_path else None,
            }
        )
    return results


def write_summary(rows: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_variant",
        "prompt_variant_source",
        "prompt_variant_requires_planning_hints",
        "accuracy",
        "syntax_accuracy",
        "mean_latency_ms",
        "samples",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def propose_dspy_variants(
    *,
    endpoint: str,
    proposer_model: str,
    current_prompt: str,
    count: int,
) -> list[PromptVariant]:
    """Ask DSPy to propose concise candidate prompt policies."""

    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - exercised when dspy is absent
        raise RuntimeError("DSPy is not installed. Install the `dspy` package first.") from exc

    class ProposeSqlPrompt(dspy.Signature):
        """Propose one concise system-prompt addendum for multi-turn text-to-SQL."""

        current_prompt: str = dspy.InputField()
        goal: str = dspy.InputField()
        variant_index: int = dspy.InputField()
        candidate_prompt: str = dspy.OutputField(
            desc=(
                "One concise prompt addendum. It must improve semantic model use, history "
                "resolution, join choice, grain handling, or executable SQL reliability."
            )
        )

    lm = dspy.LM(
        f"openai/{proposer_model}",
        api_base=endpoint,
        api_key="EMPTY",
        temperature=0.7,
        max_tokens=180,
        cache=False,
    )
    previous = dspy.settings.lm
    dspy.configure(lm=lm)
    predictor = dspy.Predict(ProposeSqlPrompt)
    variants = []
    try:
        for index in range(count):
            prediction = predictor(
                current_prompt=current_prompt,
                goal=(
                    "Improve execution accuracy for CoSQL-style multi-turn SQL. Use semantic "
                    "model context only when it helps choose physical tables, joins, measures, "
                    "dimensions, filters, and grouping."
                ),
                variant_index=index,
            )
            prompt = str(prediction.candidate_prompt).strip()
            if prompt:
                variants.append(
                    PromptVariant(
                        name=f"dspy_{index + 1}",
                        instruction=prompt,
                        source="dspy",
                    )
                )
    finally:
        dspy.configure(lm=previous)
    return variants


def run_prompt_search(
    *,
    benchmark: str,
    endpoint: str,
    model_name: str,
    output_dir: Path,
    input_path: Path | None,
    limit: int | None,
    database_root: Path | None,
    api_key: str,
    temperature: float,
    max_tokens: int,
    variants_path: Path | None,
    dspy_proposals: int,
    allow_oracle_plan: bool,
) -> int:
    records = load_benchmark_records(
        benchmark,
        input_path=input_path,
        limit=limit,
        allow_oracle_plan=allow_oracle_plan,
    )
    if not records:
        raise ValueError("benchmark produced no records")
    variants = load_prompt_variants(variants_path)
    if dspy_proposals:
        variants.extend(
            propose_dspy_variants(
                endpoint=endpoint,
                proposer_model=model_name,
                current_prompt=records[0]["messages"][0]["content"],
                count=dspy_proposals,
            )
        )

    client = OpenAI(base_url=endpoint, api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for variant in variants:
        rows = evaluate_variant(
            client=client,
            records=records,
            variant=variant,
            model_name=model_name,
            database_root=database_root,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result_path = output_dir / f"{variant.name}.jsonl"
        write_results(rows, result_path)
        summary = summarize_results(rows)
        summary_rows.append(
            {
                "prompt_variant": variant.name,
                "prompt_variant_source": variant.source,
                "prompt_variant_requires_planning_hints": variant.requires_planning_hints,
                **summary,
            }
        )
        print(
            f"{variant.name}: accuracy={summary['accuracy']:.3f} "
            f"syntax={summary['syntax_accuracy']:.3f} latency={summary['mean_latency_ms']:.1f}ms"
        )

    summary_rows.sort(key=lambda row: (row["accuracy"], row["syntax_accuracy"]), reverse=True)
    write_summary(summary_rows, output_dir / "summary.csv")
    best = summary_rows[0]
    print(
        f"Best prompt variant: {best['prompt_variant']} "
        f"(accuracy={best['accuracy']:.3f}, syntax={best['syntax_accuracy']:.3f})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["prepared", "sparc", "bird_mini_dev"], required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--variants", type=Path, default=None)
    parser.add_argument(
        "--dspy-proposals",
        type=int,
        default=0,
        help="Ask DSPy to propose this many additional prompt variants before scoring.",
    )
    parser.add_argument(
        "--allow-oracle-plan",
        action="store_true",
        help="Allow prepared inputs containing gold SQL-derived planning hints.",
    )
    args = parser.parse_args()
    return run_prompt_search(
        benchmark=args.benchmark,
        endpoint=args.endpoint,
        model_name=args.model_name,
        output_dir=args.output_dir,
        input_path=args.input,
        limit=args.limit,
        database_root=args.database_root,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        variants_path=args.variants,
        dspy_proposals=args.dspy_proposals,
        allow_oracle_plan=args.allow_oracle_plan,
    )


if __name__ == "__main__":
    raise SystemExit(main())
