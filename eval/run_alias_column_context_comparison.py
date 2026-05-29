"""Run direct SQL and alias/column-context eval as one comparable pair."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from openai import OpenAI

from eval.compare_alias_column_context import compare_alias_column_context_manifest_files
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
GenerateFn = Callable[[list[dict[str, str]]], tuple[str, float]]


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("dialog_id")),
        str(row.get("turn_index")),
        str(row.get("database_id")),
        str(row.get("reference_sql")),
    )


def _require_single_mode(rows: list[dict[str, Any]], *, label: str) -> None:
    modes = {row.get("evaluation_mode") for row in rows}
    if modes != {NON_ORACLE_GENERATION}:
        raise ValueError(
            f"{label} input must expand only to {NON_ORACLE_GENERATION} rows; got {sorted(modes)}"
        )


def validate_comparison_inputs(
    *,
    direct_input: Path,
    alias_input: Path,
    alias_context_manifest: Path,
    limit: int | None,
) -> dict[str, Any]:
    direct_rows = load_prepared_records(direct_input, limit=limit, allow_oracle_plan=False)
    alias_rows = load_prepared_records(alias_input, limit=limit, allow_oracle_plan=False)
    if not direct_rows or not alias_rows:
        raise ValueError("alias/column comparison requires non-empty prepared inputs")
    _require_single_mode(direct_rows, label="direct SQL")
    _require_single_mode(alias_rows, label="alias/column context")
    if [_row_identity(row) for row in direct_rows] != [_row_identity(row) for row in alias_rows]:
        raise ValueError("direct SQL and alias/column context input row identity mismatch")
    direct_hash = sha256_file(direct_input)
    alias_hash = sha256_file(alias_input)
    if direct_hash == alias_hash:
        raise ValueError("alias/column input must differ from direct SQL input")
    alias_manifest_hash = sha256_file(alias_context_manifest)
    if alias_manifest_hash is None:
        raise ValueError("alias/column context manifest does not exist")
    return {
        "row_count": len(direct_rows),
        "dialog_count": len({row.get("dialog_id") for row in direct_rows}),
        "database_count": len({row.get("database_id") for row in direct_rows}),
        "direct_input_sha256": direct_hash,
        "alias_input_sha256": alias_hash,
        "alias_context_manifest_path": str(alias_context_manifest),
        "alias_context_manifest_sha256": alias_manifest_hash,
    }


def write_comparison_preflight(
    *,
    direct_input: Path,
    alias_input: Path,
    alias_context_manifest: Path,
    output_path: Path,
    limit: int | None,
) -> dict[str, Any]:
    summary = validate_comparison_inputs(
        direct_input=direct_input,
        alias_input=alias_input,
        alias_context_manifest=alias_context_manifest,
        limit=limit,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "alias_column_context_comparison_preflight",
        "status": "ready_for_generation_pair",
        "claim_boundary": "preflight only; no SQL execution claim",
        "direct_input_path": str(direct_input),
        "direct_evaluation_mode": NON_ORACLE_GENERATION,
        "alias_input_path": str(alias_input),
        "alias_evaluation_mode": NON_ORACLE_GENERATION,
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


def run_alias_column_context_comparison(
    *,
    direct_input: Path,
    alias_input: Path,
    alias_context_manifest: Path,
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
    preflight = validate_comparison_inputs(
        direct_input=direct_input,
        alias_input=alias_input,
        alias_context_manifest=alias_context_manifest,
        limit=limit,
    )
    if preflight_output is not None:
        write_comparison_preflight(
            direct_input=direct_input,
            alias_input=alias_input,
            alias_context_manifest=alias_context_manifest,
            output_path=preflight_output,
            limit=limit,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    direct_output = output_dir / f"{run_id}.direct.jsonl"
    direct_manifest = output_dir / f"{run_id}.direct.manifest.json"
    alias_output = output_dir / f"{run_id}.alias_column_context.jsonl"
    alias_manifest = output_dir / f"{run_id}.alias_column_context.manifest.json"
    compared_manifest = output_dir / f"{run_id}.compared.manifest.json"

    print(
        "Alias/column context comparison preflight: "
        f"{preflight['row_count']} turns, "
        f"{preflight['dialog_count']} dialogs, "
        f"{preflight['database_count']} databases"
    )
    base_command = [
        "python",
        "-m",
        "eval.run_alias_column_context_comparison",
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
        input_path=alias_input,
        output_path=alias_output,
        manifest_path=alias_manifest,
        model_name=model_name,
        endpoint=endpoint,
        database_root=database_root,
        limit=limit,
        generate_fn=generate_fn,
        prompt_variant="alias_column_context",
        command=[*base_command, "# alias_column_context"],
    )
    compare_alias_column_context_manifest_files(
        alias_manifest_path=alias_manifest,
        direct_manifest_path=direct_manifest,
        alias_context_manifest_path=alias_context_manifest,
        output_path=compared_manifest,
        repo_root=repo_root,
    )
    print(f"Wrote alias/column context comparison manifest to {compared_manifest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-input", type=Path, required=True)
    parser.add_argument("--alias-input", type=Path, required=True)
    parser.add_argument("--alias-context-manifest", type=Path, required=True)
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
            alias_input=args.alias_input,
            alias_context_manifest=args.alias_context_manifest,
            output_path=args.preflight_output,
            limit=args.limit,
        )
        print(
            "Alias/column context comparison preflight: "
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

    return run_alias_column_context_comparison(
        direct_input=args.direct_input,
        alias_input=args.alias_input,
        alias_context_manifest=args.alias_context_manifest,
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
