"""Run direct SQL and predicted-planner SQL eval as one comparable pair."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

from data.plan_contract import validate_predicted_plan_for_prompt
from eval.compare_predicted_planner import compare_predicted_planner_manifest_files
from eval.local_generation import local_adapter_generate_fn
from eval.ragas_metrics import extract_sql, score_single_turn
from eval.result_manifest import build_result_manifest, sha256_file, write_result_manifest
from eval.run_eval import (
    database_path_for_record,
    generate_sql,
    load_prepared_records,
    messages_for_generation,
    summarize_eval_metrics,
    write_results,
)

NON_ORACLE_GENERATION = "non_oracle_generation"
PREDICTED_PLANNER = "predicted_planner"
GenerateFn = Callable[[list[dict[str, str]]], tuple[str, float]]


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dialog_id")),
        str(row.get("turn_index")),
        str(row.get("database_id")),
        str(row.get("reference_sql")),
    )


def _require_single_mode(rows: list[dict[str, Any]], *, mode: str, label: str) -> None:
    modes = {row.get("evaluation_mode") for row in rows}
    if modes != {mode}:
        raise ValueError(f"{label} input must expand only to {mode} rows; got {sorted(modes)}")


def _validate_predicted_prompt_plans(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        try:
            validate_predicted_plan_for_prompt(row.get("predicted_plan"))
        except ValueError as exc:
            row_id = row.get("id") or _row_identity(row)
            raise ValueError(f"predicted-planner row {row_id} has invalid prompt plan: {exc}") from exc


def validate_comparison_inputs(
    *,
    direct_input: Path,
    predicted_input: Path,
    limit: int | None,
) -> dict[str, Any]:
    """Validate direct and predicted prepared inputs before running endpoint eval."""

    direct_rows = load_prepared_records(direct_input, limit=limit, allow_oracle_plan=False)
    predicted_rows = load_prepared_records(predicted_input, limit=limit, allow_oracle_plan=False)
    if not direct_rows or not predicted_rows:
        raise ValueError("predicted-planner comparison requires non-empty prepared inputs")

    _require_single_mode(direct_rows, mode=NON_ORACLE_GENERATION, label="direct SQL")
    _require_single_mode(predicted_rows, mode=PREDICTED_PLANNER, label="predicted planner")
    _validate_predicted_prompt_plans(predicted_rows)

    direct_identities = [_row_identity(row) for row in direct_rows]
    predicted_identities = [_row_identity(row) for row in predicted_rows]
    if direct_identities != predicted_identities:
        raise ValueError("direct SQL and predicted-planner input row identity mismatch")

    return {
        "row_count": len(direct_rows),
        "dialog_count": len({row.get("dialog_id") for row in direct_rows}),
        "database_count": len({row.get("database_id") for row in direct_rows}),
    }


def write_comparison_preflight(
    *,
    direct_input: Path,
    predicted_input: Path,
    output_path: Path,
    limit: int | None,
) -> dict[str, Any]:
    """Write a preflight artifact proving the prepared inputs are comparable."""

    summary = validate_comparison_inputs(
        direct_input=direct_input,
        predicted_input=predicted_input,
        limit=limit,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "predicted_planner_comparison_preflight",
        "status": "ready_for_endpoint_pair",
        "claim_boundary": "preflight only; no SQL execution claim",
        "direct_input_path": str(direct_input),
        "direct_input_sha256": sha256_file(direct_input),
        "direct_evaluation_mode": NON_ORACLE_GENERATION,
        "predicted_input_path": str(predicted_input),
        "predicted_input_sha256": sha256_file(predicted_input),
        "predicted_evaluation_mode": PREDICTED_PLANNER,
        "limit": limit,
        **summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _endpoint_generate_fn(
    *,
    endpoint: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> GenerateFn:
    client = OpenAI(base_url=endpoint, api_key=api_key)

    def generate(messages: list[dict[str, str]]) -> tuple[str, float]:
        return generate_sql(
            client,
            model_name=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return generate


def _run_prepared_eval_with_generate_fn(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    model_name: str,
    endpoint: str,
    database_root: Path | None,
    limit: int | None,
    generate_fn: GenerateFn,
    prompt_variant: str,
    command: Sequence[str],
) -> None:
    records = load_prepared_records(input_path, limit=limit, allow_oracle_plan=False)
    results = []
    for record in records:
        raw_generation, generation_latency_ms = generate_fn(messages_for_generation(record))
        generated_sql = extract_sql(raw_generation)
        database_path = database_path_for_record(record, database_root)
        score = score_single_turn(record["reference_sql"], generated_sql, database_path=database_path)
        results.append(
            {
                **record,
                "model_name": model_name,
                "prompt_variant": prompt_variant,
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
    written = write_results(results, output_path)
    metrics = summarize_eval_metrics(results)
    modes = metrics.get("evaluation_modes", {})
    evaluation_mode = (
        next(iter(modes)) if len(modes) == 1 else ",".join(sorted(modes)) or "unknown"
    )
    manifest = build_result_manifest(
        run_id=output_path.stem,
        benchmark="prepared",
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        endpoint=endpoint,
        evaluation_mode=evaluation_mode,
        oracle_allowed=False,
        prompt_variant=prompt_variant,
        database_root=database_root,
        command=list(command),
        row_count=written,
        metrics=metrics,
    )
    write_result_manifest(manifest, manifest_path)


def run_predicted_planner_comparison(
    *,
    direct_input: Path,
    predicted_input: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    endpoint: str,
    database_root: Path | None,
    limit: int | None,
    generate_fn: GenerateFn,
    repo_root: Path = Path("."),
    preflight_output: Path | None = None,
) -> int:
    """Run both evals, then write the comparison manifest."""

    preflight = validate_comparison_inputs(
        direct_input=direct_input,
        predicted_input=predicted_input,
        limit=limit,
    )
    if preflight_output is not None:
        write_comparison_preflight(
            direct_input=direct_input,
            predicted_input=predicted_input,
            output_path=preflight_output,
            limit=limit,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_output = output_dir / f"{run_id}.direct.jsonl"
    direct_manifest = output_dir / f"{run_id}.direct.manifest.json"
    predicted_output = output_dir / f"{run_id}.predicted_planner.jsonl"
    predicted_manifest = output_dir / f"{run_id}.predicted_planner.manifest.json"
    compared_manifest = output_dir / f"{run_id}.compared.manifest.json"

    print(
        "Predicted-planner comparison preflight: "
        f"{preflight['row_count']} turns, "
        f"{preflight['dialog_count']} dialogs, "
        f"{preflight['database_count']} databases"
    )

    base_command = [
        "python",
        "-m",
        "eval.run_predicted_planner_comparison",
        "--run-id",
        run_id,
    ]
    _run_prepared_eval_with_generate_fn(
        input_path=direct_input,
        output_path=direct_output,
        manifest_path=direct_manifest,
        model_name=model_name,
        endpoint=endpoint,
        database_root=database_root,
        limit=limit,
        generate_fn=generate_fn,
        prompt_variant="direct_sql_control",
        command=[*base_command, "# direct_sql_control"],
    )
    _run_prepared_eval_with_generate_fn(
        input_path=predicted_input,
        output_path=predicted_output,
        manifest_path=predicted_manifest,
        model_name=model_name,
        endpoint=endpoint,
        database_root=database_root,
        limit=limit,
        generate_fn=generate_fn,
        prompt_variant="predicted_planner",
        command=[*base_command, "# predicted_planner"],
    )

    compare_predicted_planner_manifest_files(
        predicted_manifest_path=predicted_manifest,
        direct_manifest_path=direct_manifest,
        output_path=compared_manifest,
        repo_root=repo_root,
    )
    print(f"Wrote predicted-planner comparison manifest to {compared_manifest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-input", type=Path, required=True)
    parser.add_argument("--predicted-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--preflight-output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--backend", choices=["endpoint", "local"], default="endpoint")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--max-memory-gb", type=int, default=30)
    args = parser.parse_args()

    if args.preflight_only:
        if args.preflight_output is None:
            raise SystemExit("--preflight-output is required with --preflight-only")
        preflight = write_comparison_preflight(
            direct_input=args.direct_input,
            predicted_input=args.predicted_input,
            output_path=args.preflight_output,
            limit=args.limit,
        )
        print(
            "Predicted-planner comparison preflight: "
            f"{preflight['row_count']} turns, "
            f"{preflight['dialog_count']} dialogs, "
            f"{preflight['database_count']} databases"
        )
        print(f"Wrote preflight artifact to {args.preflight_output}")
        return 0

    if args.backend == "local":
        generate_fn = local_adapter_generate_fn(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            max_tokens=args.max_tokens,
            max_memory_gb=args.max_memory_gb,
        )
        endpoint = "local"
    else:
        generate_fn = _endpoint_generate_fn(
            endpoint=args.endpoint,
            api_key=args.api_key,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        endpoint = args.endpoint

    return run_predicted_planner_comparison(
        direct_input=args.direct_input,
        predicted_input=args.predicted_input,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        endpoint=endpoint,
        database_root=args.database_root,
        limit=args.limit,
        generate_fn=generate_fn,
        repo_root=args.repo_root,
        preflight_output=args.preflight_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
