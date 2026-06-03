"""Run direct SQL and semantic value-retrieval SQL eval as one comparable pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.compare_semantic_value_retrieval import (
    compare_semantic_value_retrieval_manifest_files,
)
from eval.result_manifest import sha256_file, write_result_manifest
from eval.run_eval import load_prepared_records, run_eval

NON_ORACLE_GENERATION = "non_oracle_generation"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


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


def _value_index_manifest_contract(path: Path) -> dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("artifact_type") != "non_oracle_value_index_v1":
        raise ValueError("value-index manifest must use artifact_type=non_oracle_value_index_v1")
    if manifest.get("index_source") != "database_contents":
        raise ValueError("value-index manifest must use index_source=database_contents")
    digest = sha256_file(path)
    if digest is None:
        raise ValueError("value-index manifest does not exist")
    return {
        "value_index_manifest_path": str(path),
        "value_index_manifest_sha256": digest,
        "value_index_index_source": manifest.get("index_source"),
        "value_index_output_path": manifest.get("output_path"),
        "value_index_output_sha256": manifest.get("output_sha256"),
    }


def validate_comparison_inputs(
    *,
    direct_input: Path,
    semantic_input: Path,
    value_index_manifest: Path,
    limit: int | None,
) -> dict[str, Any]:
    """Validate direct and semantic prepared inputs before endpoint eval."""

    direct_rows = load_prepared_records(direct_input, limit=limit)
    semantic_rows = load_prepared_records(semantic_input, limit=limit)
    if not direct_rows or not semantic_rows:
        raise ValueError("semantic value-retrieval comparison requires non-empty prepared inputs")

    _require_single_mode(direct_rows, label="direct SQL")
    _require_single_mode(semantic_rows, label="semantic value-retrieval")

    direct_identities = [_row_identity(row) for row in direct_rows]
    semantic_identities = [_row_identity(row) for row in semantic_rows]
    if direct_identities != semantic_identities:
        raise ValueError("direct SQL and semantic value-retrieval input row identity mismatch")

    direct_hash = sha256_file(direct_input)
    semantic_hash = sha256_file(semantic_input)
    if direct_hash == semantic_hash:
        raise ValueError("semantic value-retrieval input must differ from direct SQL input")

    return {
        "row_count": len(direct_rows),
        "dialog_count": len({row.get("dialog_id") for row in direct_rows}),
        "database_count": len({row.get("database_id") for row in direct_rows}),
        "direct_input_sha256": direct_hash,
        "semantic_input_sha256": semantic_hash,
        **_value_index_manifest_contract(value_index_manifest),
    }


def write_comparison_preflight(
    *,
    direct_input: Path,
    semantic_input: Path,
    value_index_manifest: Path,
    output_path: Path,
    limit: int | None,
) -> dict[str, Any]:
    """Write a preflight artifact proving the prepared inputs are comparable."""

    summary = validate_comparison_inputs(
        direct_input=direct_input,
        semantic_input=semantic_input,
        value_index_manifest=value_index_manifest,
        limit=limit,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "semantic_value_retrieval_comparison_preflight",
        "status": "ready_for_endpoint_pair",
        "claim_boundary": "preflight only; no SQL execution claim",
        "direct_input_path": str(direct_input),
        "direct_evaluation_mode": NON_ORACLE_GENERATION,
        "semantic_input_path": str(semantic_input),
        "semantic_evaluation_mode": NON_ORACLE_GENERATION,
        "limit": limit,
        **summary,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def annotate_semantic_manifest_with_value_index(
    *,
    semantic_manifest_path: Path,
    value_index_manifest_path: Path,
) -> dict[str, Any]:
    """Add value-index provenance required by the semantic comparer."""

    manifest = _load_json(semantic_manifest_path)
    value_index = _value_index_manifest_contract(value_index_manifest_path)
    metrics = dict(manifest.get("metrics") or {})
    metrics.update(
        {
            "value_index_manifest_sha256": value_index["value_index_manifest_sha256"],
            "value_index_index_source": value_index["value_index_index_source"],
            "value_index_output_sha256": value_index["value_index_output_sha256"],
        }
    )
    manifest["metrics"] = metrics
    manifest["command"] = list(manifest.get("command") or []) + [
        "# value-index-manifest",
        str(value_index_manifest_path),
    ]
    write_result_manifest(manifest, semantic_manifest_path)
    return manifest


def run_semantic_value_retrieval_comparison(
    *,
    direct_input: Path,
    semantic_input: Path,
    value_index_manifest: Path,
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
    """Run both endpoint evals, then write the semantic comparison manifest."""

    preflight = validate_comparison_inputs(
        direct_input=direct_input,
        semantic_input=semantic_input,
        value_index_manifest=value_index_manifest,
        limit=limit,
    )
    if preflight_output is not None:
        write_comparison_preflight(
            direct_input=direct_input,
            semantic_input=semantic_input,
            value_index_manifest=value_index_manifest,
            output_path=preflight_output,
            limit=limit,
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    direct_output = output_dir / f"{run_id}.direct.jsonl"
    direct_manifest = output_dir / f"{run_id}.direct.manifest.json"
    semantic_output = output_dir / f"{run_id}.semantic_value_retrieval.jsonl"
    semantic_manifest = output_dir / f"{run_id}.semantic_value_retrieval.manifest.json"
    compared_manifest = output_dir / f"{run_id}.compared.manifest.json"

    print(
        "Semantic value-retrieval comparison preflight: "
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
        manifest_output=direct_manifest,
        prompt_variant="direct_sql_control",
        command=[
            "python",
            "-m",
            "eval.run_semantic_value_retrieval_comparison",
            "--run-id",
            run_id,
            "# direct_sql_control",
        ],
    )
    if direct_code != 0:
        return direct_code

    semantic_code = run_eval(
        benchmark="prepared",
        endpoint=endpoint,
        model_name=model_name,
        output=semantic_output,
        input_path=semantic_input,
        limit=limit,
        database_root=database_root,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        manifest_output=semantic_manifest,
        prompt_variant="semantic_value_retrieval",
        command=[
            "python",
            "-m",
            "eval.run_semantic_value_retrieval_comparison",
            "--run-id",
            run_id,
            "# semantic_value_retrieval",
        ],
    )
    if semantic_code != 0:
        return semantic_code

    annotate_semantic_manifest_with_value_index(
        semantic_manifest_path=semantic_manifest,
        value_index_manifest_path=value_index_manifest,
    )
    compare_semantic_value_retrieval_manifest_files(
        semantic_manifest_path=semantic_manifest,
        direct_manifest_path=direct_manifest,
        value_index_manifest_path=value_index_manifest,
        output_path=compared_manifest,
        repo_root=repo_root,
    )
    print(f"Wrote semantic value-retrieval comparison manifest to {compared_manifest}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-input", type=Path, required=True)
    parser.add_argument("--semantic-input", type=Path, required=True)
    parser.add_argument("--value-index-manifest", type=Path, required=True)
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
            semantic_input=args.semantic_input,
            value_index_manifest=args.value_index_manifest,
            output_path=args.preflight_output,
            limit=args.limit,
        )
        print(
            "Semantic value-retrieval comparison preflight: "
            f"{preflight['row_count']} turns, "
            f"{preflight['dialog_count']} dialogs, "
            f"{preflight['database_count']} databases"
        )
        print(f"Wrote preflight artifact to {args.preflight_output}")
        return 0

    return run_semantic_value_retrieval_comparison(
        direct_input=args.direct_input,
        semantic_input=args.semantic_input,
        value_index_manifest=args.value_index_manifest,
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
