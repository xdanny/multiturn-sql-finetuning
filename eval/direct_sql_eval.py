"""Offline evaluation for direct-SQL outputs on synthetic metric-heavy rows."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from data.synthetic_method_fixtures import DEFAULT_OUTPUT as DEFAULT_FIXTURES_PATH
from data.synthetic_method_fixtures import build_synthetic_method_fixtures
from eval.ragas_metrics import extract_sql, score_single_turn
from eval.result_manifest import build_result_manifest, write_result_manifest

NON_ORACLE_GENERATION = "non_oracle_generation"
BENCHMARK = "metric_dsl_direct_sql"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def _fixture_map(fixtures_path: Path | None) -> dict[str, dict[str, Any]]:
    if fixtures_path is not None and fixtures_path.exists():
        fixtures = _load_jsonl(fixtures_path)
    else:
        fixtures = build_synthetic_method_fixtures()
    return {str(fixture["fixture_id"]): fixture for fixture in fixtures}


def _database_path_for_row(
    row: dict[str, Any],
    *,
    fixtures: dict[str, dict[str, Any]],
    working_dir: Path,
) -> Path | None:
    if row.get("database_path"):
        return Path(str(row["database_path"]))
    fixture_id = row.get("fixture_id")
    if not fixture_id:
        return None
    fixture = fixtures.get(str(fixture_id))
    if fixture is None:
        raise ValueError(f"unknown synthetic fixture: {fixture_id}")
    database_path = working_dir / f"{fixture_id}.sqlite"
    if not database_path.exists():
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database_path) as conn:
            conn.executescript(str(fixture["schema_sql"]))
            conn.executescript(str(fixture["seed_data_sql"]))
    return database_path


def _recovery_required(row: dict[str, Any], fixture: dict[str, Any] | None) -> bool:
    if row.get("recovery_required") is not None:
        return bool(row.get("recovery_required"))
    if fixture is None:
        return False
    checks = fixture.get("evaluation_checks") or {}
    return bool(checks.get("requires_generated_history")) or "recovery" in set(
        fixture.get("failure_modes") or []
    )


def evaluate_direct_sql_rows(
    rows: list[dict[str, Any]],
    *,
    fixtures_path: Path | None,
    working_dir: Path,
) -> list[dict[str, Any]]:
    fixtures = _fixture_map(fixtures_path)
    results: list[dict[str, Any]] = []
    for row in rows:
        generated_sql = str(
            row.get("generated_sql") or row.get("raw_generation") or row.get("predicted_sql") or ""
        )
        if not generated_sql:
            generated_sql = ""
        reference_sql = str(row["reference_sql"])
        fixture = fixtures.get(str(row.get("fixture_id"))) if row.get("fixture_id") else None
        database_path = _database_path_for_row(row, fixtures=fixtures, working_dir=working_dir)
        score = score_single_turn(reference_sql, extract_sql(generated_sql), database_path=database_path)
        recovery_required = _recovery_required(row, fixture)
        recovery_success = recovery_required and bool(score.strict_execution_score)
        results.append(
            {
                **row,
                "generated_sql": extract_sql(generated_sql),
                "evaluation_mode": row.get("evaluation_mode") or NON_ORACLE_GENERATION,
                "sql_execution_attempted": database_path is not None,
                "execution_score": score.execution_score,
                "strict_execution_score": score.strict_execution_score,
                "value_execution_score": score.value_execution_score,
                "normalized_match": score.normalized_match,
                "syntax_valid": score.syntax_valid,
                "score_error": score.error,
                "database_path": str(database_path) if database_path else None,
                "recovery_required": recovery_required,
                "recovery_success": recovery_success,
            }
        )
    return results


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    strict_scores = [
        float(result["strict_execution_score"])
        for result in results
        if result.get("strict_execution_score") is not None
    ]
    value_scores = [
        float(result["value_execution_score"])
        for result in results
        if result.get("value_execution_score") is not None
    ]
    recovery_rows = [result for result in results if result.get("recovery_required")]
    return {
        "rows": len(results),
        "execution_evaluated_rows": len(value_scores),
        "strict_execution_accuracy": _mean(strict_scores),
        "value_execution_accuracy": _mean(value_scores),
        "syntax_accuracy": _mean([float(bool(result.get("syntax_valid"))) for result in results]),
        "recovery_evaluated_rows": len(recovery_rows),
        "recovery_success_rate": _mean(
            [float(bool(result.get("recovery_success"))) for result in recovery_rows]
        ),
    }


def run_direct_sql_eval(
    *,
    input_path: Path,
    output_path: Path,
    manifest_output: Path | None,
    model_name: str,
    fixtures_path: Path | None,
    working_dir: Path,
    benchmark: str = BENCHMARK,
    prompt_variant: str = "direct_sql_control",
    command: list[str] | None = None,
) -> int:
    rows = _load_jsonl(input_path)
    results = evaluate_direct_sql_rows(rows, fixtures_path=fixtures_path, working_dir=working_dir)
    written = _write_jsonl(results, output_path)
    metrics = _summarize_results(results)
    if manifest_output is None:
        manifest_output = output_path.with_suffix(".manifest.json")
    manifest = build_result_manifest(
        run_id=output_path.stem,
        benchmark=benchmark,
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        endpoint="offline",
        evaluation_mode=NON_ORACLE_GENERATION,
        oracle_allowed=False,
        prompt_variant=prompt_variant,
        database_root=working_dir,
        command=list(command or sys.argv),
        row_count=written,
        metrics=metrics,
    )
    write_result_manifest(manifest, manifest_output)
    return 0 if written else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES_PATH)
    parser.add_argument("--working-dir", type=Path, default=Path(".tmp/direct_sql_eval"))
    args = parser.parse_args()
    return run_direct_sql_eval(
        input_path=args.input,
        output_path=args.output,
        manifest_output=args.manifest_output,
        model_name=args.model_name,
        fixtures_path=args.fixtures,
        working_dir=args.working_dir,
        command=sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
