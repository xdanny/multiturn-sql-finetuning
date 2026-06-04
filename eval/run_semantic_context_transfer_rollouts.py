"""Run Checkpoint 10 normal and semantic-context rollout arms."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from eval.rollout_eval import run_rollout_eval
from eval.semantic_context_transfer import compare_semantic_context_transfer_manifest_files

DEFAULT_NORMAL_INPUT = Path("data/processed/semantic_context_transfer/cp10_limit12_normal.jsonl")
DEFAULT_SEMANTIC_INPUT = Path(
    "data/processed/semantic_context_transfer/cp10_limit12_semantic_value.jsonl"
)
DEFAULT_PREFLIGHT = Path(
    "docs/training_runs/semantic_context_transfer_cp10_limit12_preflight_20260604.json"
)
DEFAULT_OUTPUT_DIR = Path("results/semantic_context_transfer_cp10")
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
DEFAULT_RUN_ID = "openrouter_claude_sonnet_4_6_cp10_limit12"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_PROMPT_TOKEN_COST_USD_PER_1K = 0.003
DEFAULT_COMPLETION_TOKEN_COST_USD_PER_1K = 0.015


def _api_key_from_env(api_key_env: str) -> tuple[str, str]:
    value = os.environ.get(api_key_env)
    if value:
        return value, api_key_env
    if api_key_env != OPENAI_API_KEY_ENV and os.environ.get(OPENAI_API_KEY_ENV):
        return os.environ[OPENAI_API_KEY_ENV], OPENAI_API_KEY_ENV
    raise ValueError(
        f"API key not found. Set {api_key_env} or {OPENAI_API_KEY_ENV}; "
        "do not pass secrets on the command line."
    )


def run_semantic_context_transfer_rollouts(
    *,
    normal_input: Path,
    semantic_input: Path,
    preflight_manifest: Path,
    output_dir: Path,
    run_id: str,
    model_name: str,
    endpoint: str,
    api_key: str,
    api_key_source: str,
    database_root: Path | None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    comparison_role: str = "hosted_sonnet",
    command: Sequence[str] | None = None,
    prompt_token_cost_usd_per_1k: float = DEFAULT_PROMPT_TOKEN_COST_USD_PER_1K,
    completion_token_cost_usd_per_1k: float = DEFAULT_COMPLETION_TOKEN_COST_USD_PER_1K,
) -> dict:
    """Run normal and semantic rollout arms, then compare semantic delta."""

    output_dir.mkdir(parents=True, exist_ok=True)
    command = list(command or sys.argv)
    normal_output = output_dir / f"{run_id}.normal.rollout.jsonl"
    normal_manifest = output_dir / f"{run_id}.normal.rollout.manifest.json"
    semantic_output = output_dir / f"{run_id}.semantic.rollout.jsonl"
    semantic_manifest = output_dir / f"{run_id}.semantic.rollout.manifest.json"
    comparison_manifest = output_dir / f"{run_id}.semantic_context_comparison.manifest.json"
    command_with_secret_policy = command + [
        "# api-key-source",
        api_key_source,
        "# secret-policy",
        "api key read from environment and not stored in argv",
    ]

    normal_exit = run_rollout_eval(
        endpoint=endpoint,
        model_name=model_name,
        input_path=normal_input,
        output=normal_output,
        limit_dialogs=None,
        database_root=database_root,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        allow_oracle_plan=False,
        manifest_output=normal_manifest,
        prompt_variant="normal_schema_context",
        command=command_with_secret_policy + ["# prompt-policy", "normal_schema_context"],
        prompt_token_cost_usd_per_1k=prompt_token_cost_usd_per_1k,
        completion_token_cost_usd_per_1k=completion_token_cost_usd_per_1k,
    )
    if normal_exit != 0:
        raise RuntimeError("normal-context rollout failed")

    semantic_exit = run_rollout_eval(
        endpoint=endpoint,
        model_name=model_name,
        input_path=semantic_input,
        output=semantic_output,
        limit_dialogs=None,
        database_root=database_root,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        allow_oracle_plan=False,
        manifest_output=semantic_manifest,
        prompt_variant="schema_context_plus_database_value_retrieval",
        command=command_with_secret_policy
        + ["# prompt-policy", "schema_context_plus_database_value_retrieval"],
        prompt_token_cost_usd_per_1k=prompt_token_cost_usd_per_1k,
        completion_token_cost_usd_per_1k=completion_token_cost_usd_per_1k,
    )
    if semantic_exit != 0:
        raise RuntimeError("semantic-context rollout failed")

    return compare_semantic_context_transfer_manifest_files(
        normal_manifest_path=normal_manifest,
        semantic_manifest_path=semantic_manifest,
        preflight_manifest_path=preflight_manifest,
        output_path=comparison_manifest,
        comparison_role=comparison_role,
        repo_root=Path("."),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-input", type=Path, default=DEFAULT_NORMAL_INPUT)
    parser.add_argument("--semantic-input", type=Path, default=DEFAULT_SEMANTIC_INPUT)
    parser.add_argument("--preflight-manifest", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--database-root", type=Path, default=Path("data/raw/cosql_dataset/database"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--comparison-role", default="hosted_sonnet")
    parser.add_argument(
        "--prompt-token-cost-usd-per-1k",
        type=float,
        default=DEFAULT_PROMPT_TOKEN_COST_USD_PER_1K,
    )
    parser.add_argument(
        "--completion-token-cost-usd-per-1k",
        type=float,
        default=DEFAULT_COMPLETION_TOKEN_COST_USD_PER_1K,
    )
    args = parser.parse_args()
    api_key, api_key_source = _api_key_from_env(args.api_key_env)
    compared = run_semantic_context_transfer_rollouts(
        normal_input=args.normal_input,
        semantic_input=args.semantic_input,
        preflight_manifest=args.preflight_manifest,
        output_dir=args.output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        endpoint=args.endpoint,
        api_key=api_key,
        api_key_source=api_key_source,
        database_root=args.database_root,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        comparison_role=args.comparison_role,
        command=sys.argv,
        prompt_token_cost_usd_per_1k=args.prompt_token_cost_usd_per_1k,
        completion_token_cost_usd_per_1k=args.completion_token_cost_usd_per_1k,
    )
    metrics = compared["metrics"]
    print(f"Wrote semantic context transfer outputs under {args.output_dir}")
    print(
        "Value delta vs normal context: "
        f"{metrics['semantic_context_value_delta_vs_normal']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
