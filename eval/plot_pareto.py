"""
Generate the cost-vs-accuracy Pareto frontier chart for the blog post / LinkedIn.

Reads results/*.jsonl from all evaluations and produces:
  - plots/pareto_cost_vs_accuracy.png
  - plots/multi_turn_vs_single_turn.png
  - plots/per_benchmark_breakdown.png

TODO: implement.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="plots")
    args = parser.parse_args()

    raise NotImplementedError("TODO: aggregate results and plot Pareto frontier")


if __name__ == "__main__":
    main()
