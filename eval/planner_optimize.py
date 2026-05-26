"""
Search planner prompt/program variants before SQL generation.

This optimizer scores planner JSON output against gold SQL-derived planner
labels. It does not claim SQL improvement; it produces the planner-quality
evidence needed before running the more expensive planner-to-SQL path.
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

from eval.planner_eval import PLAN_FIELDS, parse_json_plan_prediction, score_plans
from eval.planner_predict import (
    GeneratePlannerFn,
    generate_planner_json,
    planner_messages_for_record,
)
from eval.run_eval import load_prepared_records, write_results


@dataclass(frozen=True)
class PlannerPromptVariant:
    name: str
    instruction: str
    source: str = "static"


DEFAULT_PLANNER_PROMPT_VARIANTS = [
    PlannerPromptVariant(
        "baseline",
        "Predict only the schema objects and query shape directly supported by the visible prompt.",
    ),
    PlannerPromptVariant(
        "schema_link_first",
        (
            "First choose the smallest sufficient table set, then select only columns "
            "needed for filters, joins, grouping, ordering, and projection."
        ),
    ),
    PlannerPromptVariant(
        "projection_shape",
        (
            "Be precise about selected_count, selected_expressions, aggregation outputs, "
            "group_by fields, distinct, limit, order_by, and duplicate policy."
        ),
    ),
    PlannerPromptVariant(
        "history_delta",
        (
            "For follow-up turns, preserve prior filters, entities, metrics, and grain "
            "only when the current question refers to them; replace them when the user "
            "changes the target."
        ),
    ),
]


def apply_planner_prompt_variant(
    messages: list[dict[str, str]],
    variant: PlannerPromptVariant,
) -> list[dict[str, str]]:
    """Append a planner policy to the system prompt without mutating input messages."""

    updated = [dict(message) for message in messages]
    addendum = f"Planner prompt policy ({variant.name}): {variant.instruction}"
    for message in updated:
        if message.get("role") == "system":
            message["content"] = f"{message['content']}\n\n{addendum}"
            return updated
    return [{"role": "system", "content": addendum}, *updated]


def load_planner_prompt_variants(path: Path | None) -> list[PlannerPromptVariant]:
    if path is None:
        return list(DEFAULT_PLANNER_PROMPT_VARIANTS)
    rows = json.loads(path.read_text())
    return [
        PlannerPromptVariant(
            name=str(row["name"]),
            instruction=str(row["instruction"]),
            source=str(row.get("source") or "file"),
        )
        for row in rows
    ]


def zero_planner_scores() -> dict[str, float]:
    return {field: 0.0 for field in PLAN_FIELDS}


def score_planner_prediction(
    gold_plan: dict[str, Any] | None,
    predicted_plan: dict[str, Any],
) -> dict[str, float]:
    """Score a planner prediction; malformed JSON cannot receive partial credit."""

    if predicted_plan.get("parseable") is False:
        return zero_planner_scores()
    return score_plans(gold_plan, predicted_plan)


def planner_summary_sort_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    """Rank parseable planner variants before field-level F1 ties."""

    return (
        float(row["parse_rate"]),
        float(row["macro_planner_score"]),
        float(row["column_f1"]),
        float(row["table_f1"]),
    )


def evaluate_planner_variant(
    *,
    records: list[dict[str, Any]],
    variant: PlannerPromptVariant,
    generate_fn: GeneratePlannerFn,
    model_name: str,
    prediction_source: str = "planner_prompt_search",
) -> list[dict[str, Any]]:
    """Generate and score planner predictions for one planner prompt variant."""

    rows = []
    for record in records:
        messages = apply_planner_prompt_variant(
            planner_messages_for_record(record),
            variant,
        )
        raw_output, latency_ms = generate_fn(messages)
        predicted_plan = parse_json_plan_prediction(
            raw_output,
            prediction_source=prediction_source,
        )
        planner_scores = score_planner_prediction(
            record.get("gold_plan") or record.get("schema_link_labels") or {},
            predicted_plan,
        )
        rows.append(
            {
                "id": str(record["id"]),
                "dialog_id": str(record.get("dialog_id") or ""),
                "turn_index": int(record.get("turn_index") or 0),
                "database_id": record.get("database_id"),
                "model_name": model_name,
                "prompt_variant": variant.name,
                "prompt_variant_source": variant.source,
                "prompt_instruction": variant.instruction,
                "raw_planner_output": raw_output,
                "predicted_plan": predicted_plan,
                "planner_scores": planner_scores,
                "planner_latency_ms": latency_ms,
            }
        )
    return rows


def summarize_planner_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        summary = {field: 0.0 for field in PLAN_FIELDS}
        return {
            "samples": 0,
            "parse_rate": 0.0,
            "mean_latency_ms": 0.0,
            **summary,
        }
    summary = {
        field: sum(float(row["planner_scores"][field]) for row in rows) / len(rows)
        for field in PLAN_FIELDS
    }
    return {
        "samples": len(rows),
        "parse_rate": sum(bool(row["predicted_plan"].get("parseable")) for row in rows)
        / len(rows),
        "mean_latency_ms": sum(float(row["planner_latency_ms"]) for row in rows)
        / len(rows),
        **summary,
    }


def write_planner_summary(rows: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "prompt_variant",
        "prompt_variant_source",
        "samples",
        "parse_rate",
        "mean_latency_ms",
        *PLAN_FIELDS,
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def propose_dspy_planner_variants(
    *,
    endpoint: str,
    proposer_model: str,
    current_prompt: str,
    count: int,
) -> list[PlannerPromptVariant]:
    """Ask DSPy to propose concise planner policies."""

    try:
        import dspy
    except ImportError as exc:  # pragma: no cover - exercised when dspy is absent
        raise RuntimeError("DSPy is not installed. Install the `dspy` package first.") from exc

    class ProposePlannerPrompt(dspy.Signature):
        """Propose one concise system-prompt addendum for SQL planning."""

        current_prompt: str = dspy.InputField()
        goal: str = dspy.InputField()
        variant_index: int = dspy.InputField()
        candidate_prompt: str = dspy.OutputField(
            desc=(
                "One concise planner-policy addendum. It must improve table, column, "
                "join, projection-shape, aggregation, grouping, value, or history planning."
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
    predictor = dspy.Predict(ProposePlannerPrompt)
    variants = []
    try:
        for index in range(count):
            prediction = predictor(
                current_prompt=current_prompt,
                goal=(
                    "Improve non-oracle planner label F1 for multi-turn text-to-SQL. "
                    "The planner must use only visible question, history, schema, and "
                    "semantic context. It must not rely on reference SQL."
                ),
                variant_index=index,
            )
            prompt = str(prediction.candidate_prompt).strip()
            if prompt:
                variants.append(
                    PlannerPromptVariant(
                        name=f"dspy_planner_{index + 1}",
                        instruction=prompt,
                        source="dspy",
                    )
                )
    finally:
        dspy.configure(lm=previous)
    return variants


def run_planner_prompt_search(
    *,
    input_path: Path,
    output_dir: Path,
    endpoint: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    limit: int | None,
    variants_path: Path | None,
    dspy_proposals: int,
) -> int:
    records = load_prepared_records(input_path, limit=limit, allow_oracle_plan=False)
    if not records:
        raise ValueError("prepared input produced no expanded records")

    variants = load_planner_prompt_variants(variants_path)
    if dspy_proposals:
        variants.extend(
            propose_dspy_planner_variants(
                endpoint=endpoint,
                proposer_model=model_name,
                current_prompt=planner_messages_for_record(records[0])[0]["content"],
                count=dspy_proposals,
            )
        )

    client = OpenAI(base_url=endpoint, api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    def endpoint_generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        return generate_planner_json(
            client,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    summary_rows = []
    for variant in variants:
        rows = evaluate_planner_variant(
            records=records,
            variant=variant,
            generate_fn=endpoint_generate,
            model_name=model_name,
        )
        result_path = output_dir / f"{variant.name}.jsonl"
        write_results(rows, result_path)
        summary = summarize_planner_variant(rows)
        summary_rows.append(
            {
                "prompt_variant": variant.name,
                "prompt_variant_source": variant.source,
                **summary,
            }
        )
        print(
            f"{variant.name}: macro={summary['macro_planner_score']:.3f} "
            f"table_f1={summary['table_f1']:.3f} "
            f"column_f1={summary['column_f1']:.3f} "
            f"parse={summary['parse_rate']:.3f}"
        )

    summary_rows.sort(key=planner_summary_sort_key, reverse=True)
    write_planner_summary(summary_rows, output_dir / "summary.csv")
    best = summary_rows[0]
    print(
        f"Best planner variant: {best['prompt_variant']} "
        f"(macro={best['macro_planner_score']:.3f}, "
        f"column_f1={best['column_f1']:.3f})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variants", type=Path, default=None)
    parser.add_argument(
        "--dspy-proposals",
        type=int,
        default=0,
        help="Ask DSPy to propose this many planner prompt variants before scoring.",
    )
    args = parser.parse_args()
    return run_planner_prompt_search(
        input_path=args.input,
        output_dir=args.output_dir,
        endpoint=args.endpoint,
        model_name=args.model_name,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
        variants_path=args.variants,
        dspy_proposals=args.dspy_proposals,
    )


if __name__ == "__main__":
    raise SystemExit(main())
