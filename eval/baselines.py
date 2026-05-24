"""
Convenience wrappers for benchmark baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.run_eval import run_eval

BASELINES = {
    "qwen35-9b-base": {
        "model_name": "Qwen/Qwen3.5-9B",
        "endpoint": "http://localhost:8000/v1",
    },
    "qwen35-9b-finetuned": {
        "model_name": "multiturn-sql",
        "endpoint": "http://localhost:8000/v1",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=BASELINES, required=True)
    parser.add_argument("--benchmark", choices=["prepared", "sparc", "bird_mini_dev"], required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--database-root", type=Path, default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--allow-oracle-plan",
        action="store_true",
        help="Allow prepared inputs containing gold SQL-derived planning hints.",
    )
    args = parser.parse_args()

    baseline = BASELINES[args.model]
    output = args.output or Path(f"results/{args.model}_{args.benchmark}.jsonl")
    return run_eval(
        benchmark=args.benchmark,
        endpoint=args.endpoint or baseline["endpoint"],
        model_name=baseline["model_name"],
        output=output,
        input_path=args.input,
        limit=args.limit,
        database_root=args.database_root,
        api_key=args.api_key,
        temperature=0.0,
        max_tokens=512,
        allow_oracle_plan=args.allow_oracle_plan,
    )


if __name__ == "__main__":
    raise SystemExit(main())
