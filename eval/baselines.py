"""
Run API model baselines through the same RAGAS evaluation harness.

Models: Claude Haiku 4.5, GPT-4o-mini, Gemini Flash, Qwen 3.5 9B base (self-hosted).

Usage:
    python -m eval.baselines --model claude-haiku-4-5 --benchmark cosql
    python -m eval.baselines --model gpt-4o-mini --benchmark cosql
    python -m eval.baselines --model gemini-flash --benchmark cosql

TODO: implement.
"""

from __future__ import annotations

import argparse


API_MODELS = [
    "claude-haiku-4-5",
    "gpt-4o-mini",
    "gemini-flash",
    "qwen35-9b-base",  # self-hosted ablation
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=API_MODELS, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    raise NotImplementedError(f"TODO: baseline {args.model} on {args.benchmark}")


if __name__ == "__main__":
    main()
