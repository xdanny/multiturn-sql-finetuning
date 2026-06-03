"""Evaluate MEASURE()-preserving metric DSL predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from data.metric_dsl import compile_metric_query, parse_metric_query, score_metric_query
from eval.ragas_metrics import score_single_turn
from eval.result_manifest import build_result_manifest, write_result_manifest

METRIC_DSL_MODE = "metric_dsl"
MEASURE_SEGMENT_RE = re.compile(
    r"MEASURE\([^)]+\)(\s*,\s*MEASURE\([^)]+\))*",
    re.IGNORECASE,
)
FORBIDDEN_SQL_KEYWORD_RE = re.compile(
    r"\b(SELECT|FROM|JOIN|GROUP\s+BY|HAVING|UNION|INTERSECT|EXCEPT)\b",
    re.IGNORECASE,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _zero_scores() -> dict[str, float]:
    return {
        "measure_f1": 0.0,
        "dimension_f1": 0.0,
        "filter_f1": 0.0,
        "measure_preservation": 0.0,
    }


def _metric_query_dict(query: Any) -> dict[str, Any]:
    return {
        "measures": list(query.measures),
        "dimensions": list(query.dimensions),
        "filters": list(query.filters),
        "order_by": query.order_by,
        "limit": query.limit,
    }


def _semantic_model_sha256(semantic_model: Any) -> str:
    encoded = json.dumps(semantic_model, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _field(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    raise KeyError(names[0])


def _optional_field(row: dict[str, Any], *names: str) -> str | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def _semantic_model_source(row: dict[str, Any]) -> str:
    return str(row.get("semantic_model_source") or "inline_unversioned")


def _semantic_model_is_oracle_derived(row: dict[str, Any]) -> bool:
    source = _semantic_model_source(row).lower()
    return "gold_reference" in source or source.startswith("oracle_")


def _values_for_database_backed(
    results: list[dict[str, Any]],
    field: str,
) -> list[float]:
    values = []
    for result in results:
        if not result.get("database_path"):
            continue
        if result.get("sql_execution_attempted") and result.get(field) is not None:
            values.append(float(result[field]))
        else:
            values.append(0.0)
    return values


def _measure_segment(text: str) -> str:
    remaining = " ".join(text.strip().split())
    limit_match = re.search(r"\s+LIMIT\s+\d+\s*$", remaining, re.IGNORECASE)
    if limit_match:
        remaining = remaining[: limit_match.start()].strip()
    order_match = re.search(r"\s+ORDER\s+BY\s+.+$", remaining, re.IGNORECASE)
    if order_match:
        remaining = remaining[: order_match.start()].strip()
    where_match = re.search(r"\s+WHERE\s+.+$", remaining, re.IGNORECASE)
    if where_match:
        remaining = remaining[: where_match.start()].strip()
    by_match = re.search(r"\s+BY\s+.+$", remaining, re.IGNORECASE)
    if by_match:
        remaining = remaining[: by_match.start()].strip()
    return remaining


def _validate_metric_dsl_text(text: str) -> None:
    if FORBIDDEN_SQL_KEYWORD_RE.search(text):
        raise ValueError("invalid metric DSL: SQL keyword found outside DSL contract")
    if not MEASURE_SEGMENT_RE.fullmatch(_measure_segment(text)):
        raise ValueError("invalid metric DSL measure segment; expected MEASURE(name)")


def evaluate_metric_dsl_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score metric intent, compile through the semantic model, and score SQL when possible."""

    results = []
    for row in rows:
        semantic_model = row.get("semantic_model")
        semantic_model_error = None
        semantic_model_sha256 = None
        semantic_model_source = _semantic_model_source(row)
        semantic_model_oracle_derived = _semantic_model_is_oracle_derived(row)
        if isinstance(semantic_model, dict):
            semantic_model_sha256 = _semantic_model_sha256(semantic_model)
        else:
            semantic_model_error = "missing semantic_model"
        predicted_dsl = _optional_field(row, "predicted_dsl", "generated_metric_dsl", "raw_generation")
        gold_dsl = _optional_field(row, "gold_dsl", "reference_metric_dsl")
        metric_scores = _zero_scores()
        parsed_predicted_metric_query = None
        parsed_gold_metric_query = None
        metric_dsl_parse_success = False
        parse_error = None
        compiled_sql = None
        compile_success = False
        compile_error = None
        sql_execution_attempted = False
        value_execution_score = None
        strict_execution_score = None
        syntax_valid = False
        score_error = None
        try:
            if predicted_dsl is None:
                raise ValueError("missing predicted_dsl")
            if gold_dsl is None:
                raise ValueError("missing gold_dsl")
            _validate_metric_dsl_text(predicted_dsl)
            _validate_metric_dsl_text(gold_dsl)
            predicted_query = parse_metric_query(predicted_dsl)
            gold_query = parse_metric_query(gold_dsl)
            parsed_predicted_metric_query = _metric_query_dict(predicted_query)
            parsed_gold_metric_query = _metric_query_dict(gold_query)
            metric_scores = score_metric_query(predicted_query, gold_query)
            if gold_query.measures and not predicted_query.measures:
                parse_error = "metric DSL prediction must preserve at least one MEASURE(...) token"
            else:
                metric_dsl_parse_success = True
        except Exception as exc:
            parse_error = str(exc)

        if metric_dsl_parse_success and semantic_model_error is None:
            try:
                compiled_sql = compile_metric_query(predicted_query, semantic_model)
                compile_success = True
                syntax_valid = True
            except Exception as exc:
                compile_error = str(exc)

        if compile_success and row.get("reference_sql"):
            if row.get("database_path"):
                sql_execution_attempted = True
                score = score_single_turn(
                    str(row["reference_sql"]),
                    compiled_sql,
                    database_path=row.get("database_path"),
                )
                value_execution_score = float(
                    score.value_execution_score
                    if score.value_execution_score is not None
                    else score.execution_score
                )
                strict_execution_score = float(
                    score.strict_execution_score
                    if score.strict_execution_score is not None
                    else score.execution_score
                )
                syntax_valid = bool(score.syntax_valid)
                score_error = score.error
            else:
                score_error = "database_path missing; skipped database-backed SQL execution"

        results.append(
            {
                **row,
                "predicted_dsl": predicted_dsl,
                "gold_dsl": gold_dsl,
                "evaluation_mode": METRIC_DSL_MODE,
                "semantic_model_sha256": semantic_model_sha256,
                "semantic_model_source": semantic_model_source,
                "semantic_model_oracle_derived": semantic_model_oracle_derived,
                "semantic_model_error": semantic_model_error,
                "parsed_predicted_metric_query": parsed_predicted_metric_query,
                "parsed_gold_metric_query": parsed_gold_metric_query,
                "metric_dsl_parse_success": metric_dsl_parse_success,
                "parse_error": parse_error,
                "metric_scores": metric_scores,
                "compiled_sql": compiled_sql,
                "compile_success": compile_success,
                "compile_error": compile_error,
                "sql_execution_attempted": sql_execution_attempted,
                "value_execution_score": value_execution_score,
                "strict_execution_score": strict_execution_score,
                "syntax_valid": syntax_valid,
                "score_error": score_error,
            }
        )
    return results


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_metric_dsl_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metric-DSL intent, compiler, and compiled-SQL execution scores."""

    if not results:
        return {}
    metric_scores = [result.get("metric_scores") or {} for result in results]
    semantic_model_hashes = sorted(
        {
            str(result["semantic_model_sha256"])
            for result in results
            if result.get("semantic_model_sha256")
        }
    )
    execution_values = _values_for_database_backed(results, "value_execution_score")
    strict_values = _values_for_database_backed(results, "strict_execution_score")
    semantic_model_sources = Counter(
        str(result.get("semantic_model_source") or "unknown") for result in results
    )
    return {
        "rows": len(results),
        "metric_dsl_parse_rate": _mean(
            [float(bool(result.get("metric_dsl_parse_success"))) for result in results]
        ),
        "metric_dsl_compile_rate": _mean(
            [float(bool(result.get("compile_success"))) for result in results]
        ),
        "compiled_sql_execution_attempt_rate": _mean(
            [float(bool(result.get("sql_execution_attempted"))) for result in results]
        ),
        "compiled_sql_execution_evaluated_rows": sum(
            1
            for result in results
            if result.get("sql_execution_attempted")
            and result.get("value_execution_score") is not None
        ),
        "measure_f1": _mean([float(score.get("measure_f1") or 0.0) for score in metric_scores]),
        "dimension_f1": _mean(
            [float(score.get("dimension_f1") or 0.0) for score in metric_scores]
        ),
        "filter_f1": _mean([float(score.get("filter_f1") or 0.0) for score in metric_scores]),
        "measure_preservation": _mean(
            [float(score.get("measure_preservation") or 0.0) for score in metric_scores]
        ),
        "value_execution_accuracy": _mean(execution_values) if execution_values else None,
        "strict_execution_accuracy": _mean(strict_values) if strict_values else None,
        "syntax_accuracy": _mean([float(bool(result.get("syntax_valid"))) for result in results]),
        "semantic_model_sha256s": semantic_model_hashes,
        "semantic_model_sources": dict(sorted(semantic_model_sources.items())),
        "semantic_model_oracle_derived_rows": sum(
            1 for result in results if result.get("semantic_model_oracle_derived")
        ),
    }


def run_metric_dsl_eval(
    *,
    input_path: Path,
    output_path: Path,
    manifest_output: Path | None,
    model_name: str,
    command: Sequence[str] | None = None,
) -> int:
    rows = _load_jsonl(input_path)
    results = evaluate_metric_dsl_rows(rows)
    written = _write_jsonl(results, output_path)
    metrics = summarize_metric_dsl_results(results)
    if manifest_output is None:
        manifest_output = output_path.with_suffix(".manifest.json")
    manifest = build_result_manifest(
        run_id=output_path.stem,
        benchmark=METRIC_DSL_MODE,
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        endpoint="offline",
        evaluation_mode=METRIC_DSL_MODE,
        oracle_allowed=bool(metrics.get("semantic_model_oracle_derived_rows")),
        prompt_variant=None,
        database_root=None,
        command=list(command or sys.argv),
        row_count=written,
        metrics=metrics,
    )
    write_result_manifest(manifest, manifest_output)
    print(f"Wrote {written} metric-DSL rows to {output_path}")
    print(f"Wrote metric-DSL manifest to {manifest_output}")
    print(f"Mean measure preservation: {metrics.get('measure_preservation', 0.0):.3f}")
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--model-name", required=True)
    args = parser.parse_args()

    return run_metric_dsl_eval(
        input_path=args.input,
        output_path=args.output,
        manifest_output=args.manifest_output,
        model_name=args.model_name,
        command=sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
