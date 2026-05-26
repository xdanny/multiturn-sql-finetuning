"""Run direct SQL and predicted-planner SQL eval as one comparable pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_predicted_planner import compare_predicted_planner_manifest_files
from eval.result_manifest import sha256_file
from eval.run_eval import load_prepared_records, run_eval

NON_ORACLE_GENERATION = "non_oracle_generation"
PREDICTED_PLANNER = "predicted_planner"


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


def run_predicted_planner_comparison(
    *,
    direct_input: Path,
    predicted_input: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    endpoint: str,
    database_root: Path | None,
    api_key: str,
    temperature: float,
    max_tokens: int,
    limit: int | None,
    repo_root: Path = Path("."),
    preflight_output: Path | None = None,
) -> int:
    """Run both endpoint evals, then write the comparison manifest."""

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

    direct_code = run_eval(
        benchmark="prepared",
        endpoint=endpoint,
        model_name=model_name,
        output=direct_output,
        input_path=direct_input,
        limit=limit,
        database_root=database_root,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        allow_oracle_plan=False,
        manifest_output=direct_manifest,
        prompt_variant="direct_sql_control",
        command=[
            "python",
            "-m",
            "eval.run_predicted_planner_comparison",
            "--run-id",
            run_id,
            "# direct_sql_control",
        ],
    )
    if direct_code != 0:
        return direct_code

    predicted_code = run_eval(
        benchmark="prepared",
        endpoint=endpoint,
        model_name=model_name,
        output=predicted_output,
        input_path=predicted_input,
        limit=limit,
        database_root=database_root,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        allow_oracle_plan=False,
        manifest_output=predicted_manifest,
        prompt_variant="predicted_planner",
        command=[
            "python",
            "-m",
            "eval.run_predicted_planner_comparison",
            "--run-id",
            run_id,
            "# predicted_planner",
        ],
    )
    if predicted_code != 0:
        return predicted_code

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

    return run_predicted_planner_comparison(
        direct_input=args.direct_input,
        predicted_input=args.predicted_input,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        endpoint=args.endpoint,
        database_root=args.database_root,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        limit=args.limit,
        repo_root=args.repo_root,
        preflight_output=args.preflight_output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
