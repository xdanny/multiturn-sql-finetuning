"""Run the offline metric-DSL versus direct-SQL comparison gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from eval.compare_metric_dsl_direct_sql import (
    METRIC_DSL_DIRECT_SQL,
    NON_ORACLE_GENERATION,
    compare_metric_dsl_direct_sql_manifest_files,
)
from eval.metric_dsl_eval import run_metric_dsl_eval
from eval.ragas_metrics import extract_sql, score_single_turn
from eval.result_manifest import build_result_manifest, write_result_manifest
from eval.run_eval import summarize_eval_metrics


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _generated_sql(row: dict[str, Any]) -> str:
    value = row.get("generated_sql")
    if value not in (None, ""):
        return str(value)
    value = row.get("raw_generation")
    if value not in (None, ""):
        return extract_sql(str(value))
    raise ValueError(f"direct SQL prediction row {row.get('id')} is missing generated_sql")


def evaluate_direct_sql_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    """Score direct-SQL predictions on the metric-heavy comparison rows."""

    results = []
    for row in rows:
        generated_sql = _generated_sql(row)
        database_path = row.get("database_path")
        score = score_single_turn(
            str(row["reference_sql"]),
            generated_sql,
            database_path=database_path,
        )
        results.append(
            {
                **row,
                "model_name": model_name,
                "evaluation_mode": NON_ORACLE_GENERATION,
                "generated_sql": generated_sql,
                "execution_score": score.execution_score,
                "strict_execution_score": score.strict_execution_score,
                "value_execution_score": score.value_execution_score,
                "order_sensitive": score.order_sensitive,
                "normalized_match": score.normalized_match,
                "syntax_valid": score.syntax_valid,
                "score_error": score.error,
            }
        )
    return results


def write_direct_sql_prediction_results(
    *,
    input_path: Path,
    output_path: Path,
    manifest_output: Path,
    model_name: str,
    command: Sequence[str],
) -> dict[str, Any]:
    """Write direct-SQL scored rows and a metric-DSL-control manifest."""

    results = evaluate_direct_sql_prediction_rows(
        _load_jsonl(input_path),
        model_name=model_name,
    )
    row_count = _write_jsonl(output_path, results)
    metrics = summarize_eval_metrics(results)
    manifest = build_result_manifest(
        run_id=output_path.stem,
        benchmark=METRIC_DSL_DIRECT_SQL,
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        endpoint="offline",
        evaluation_mode=NON_ORACLE_GENERATION,
        oracle_allowed=False,
        prompt_variant=None,
        database_root=None,
        command=command,
        row_count=row_count,
        metrics=metrics,
    )
    write_result_manifest(manifest, manifest_output)
    return manifest


def run_metric_dsl_comparison(
    *,
    metric_dsl_predictions: Path,
    direct_sql_predictions: Path,
    output_dir: Path,
    run_id: str,
    metric_dsl_model_name: str,
    direct_sql_model_name: str,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score both arms and write the same-row comparison manifest."""

    command = list(command or sys.argv)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_output = output_dir / f"{run_id}.metric_dsl.jsonl"
    metric_manifest = output_dir / f"{run_id}.metric_dsl.manifest.json"
    direct_output = output_dir / f"{run_id}.direct_sql.jsonl"
    direct_manifest = output_dir / f"{run_id}.direct_sql.manifest.json"
    comparison_manifest = output_dir / f"{run_id}.comparison.manifest.json"

    metric_exit_code = run_metric_dsl_eval(
        input_path=metric_dsl_predictions,
        output_path=metric_output,
        manifest_output=metric_manifest,
        model_name=metric_dsl_model_name,
        command=command,
    )
    if metric_exit_code != 0:
        raise RuntimeError("metric-DSL evaluation wrote no rows")

    write_direct_sql_prediction_results(
        input_path=direct_sql_predictions,
        output_path=direct_output,
        manifest_output=direct_manifest,
        model_name=direct_sql_model_name,
        command=command,
    )

    return compare_metric_dsl_direct_sql_manifest_files(
        metric_dsl_manifest_path=metric_manifest,
        direct_sql_manifest_path=direct_manifest,
        output_path=comparison_manifest,
        repo_root=Path("."),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-dsl-predictions", type=Path, required=True)
    parser.add_argument("--direct-sql-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metric-dsl-model-name", required=True)
    parser.add_argument("--direct-sql-model-name", required=True)
    args = parser.parse_args()

    compared = run_metric_dsl_comparison(
        metric_dsl_predictions=args.metric_dsl_predictions,
        direct_sql_predictions=args.direct_sql_predictions,
        output_dir=args.output_dir,
        run_id=args.run_id,
        metric_dsl_model_name=args.metric_dsl_model_name,
        direct_sql_model_name=args.direct_sql_model_name,
        command=sys.argv,
    )
    metrics = compared["metrics"]
    print(f"Wrote metric-DSL comparison manifest under {args.output_dir}")
    print(
        "Value delta vs direct SQL: "
        f"{metrics['metric_dsl_value_delta_vs_direct_sql']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
