"""
Run RAGAS evaluation pipeline against a served model (local vLLM or API).

Supports CoSQL, SParC, BIRD mini_dev benchmarks.

Usage:
    python -m eval.run_eval --benchmark cosql --endpoint http://localhost:8000/v1 --model-name multiturn-sql
    python -m eval.run_eval --benchmark bird_mini_dev --endpoint https://api.anthropic.com --model-name claude-haiku-4-5

TODO: implement.
"""

from __future__ import annotations

import argparse


BENCHMARKS = ["cosql", "sparc", "bird_mini_dev", "bird_interact_lite"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible API endpoint")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", default="results/eval.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (debug)")
    args = parser.parse_args()

    raise NotImplementedError(f"TODO: run {args.benchmark} eval against {args.model_name}")


if __name__ == "__main__":
    main()
