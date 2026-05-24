"""
Generate benchmark summary plots from benchmark JSONL results.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def resolve_result_files(results_dir: Path, includes: Sequence[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for include in includes:
        for path in sorted(results_dir.glob(include)):
            if path.is_file() and path.suffix == ".jsonl" and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def load_results(results_dir: Path, includes: Sequence[str] = ("*.jsonl",)) -> pd.DataFrame:
    rows = []
    for path in resolve_result_files(results_dir, includes):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                row["result_file"] = path.name
                rows.append(row)
    if not rows:
        include_text = ", ".join(includes)
        raise FileNotFoundError(f"no JSONL results found in {results_dir} for {include_text}")
    return pd.DataFrame(rows)


def _with_prompt_variant(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    if "prompt_variant" not in results.columns:
        results["prompt_variant"] = ""
    results["prompt_variant"] = results["prompt_variant"].fillna("")
    return results


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    results = _with_prompt_variant(results)
    if "strict_execution_score" not in results.columns:
        results["strict_execution_score"] = results["execution_score"]
    if "value_execution_score" not in results.columns:
        results["value_execution_score"] = results["execution_score"]
    grouped = results.groupby(["model_name", "prompt_variant", "source"], dropna=False)
    return grouped.agg(
        accuracy=("execution_score", "mean"),
        strict_accuracy=("strict_execution_score", "mean"),
        value_accuracy=("value_execution_score", "mean"),
        syntax_accuracy=("syntax_valid", "mean"),
        mean_latency_ms=("generation_latency_ms", "mean"),
        samples=("execution_score", "count"),
        result_files=("result_file", lambda values: ",".join(sorted(set(values)))),
    ).reset_index()


def summarize_dialog_results(results: pd.DataFrame) -> pd.DataFrame:
    if "dialog_id" not in results.columns:
        return pd.DataFrame()

    results = _with_prompt_variant(results)
    if "strict_execution_score" not in results.columns:
        results["strict_execution_score"] = results["execution_score"]
    if "value_execution_score" not in results.columns:
        results["value_execution_score"] = results["execution_score"]
    per_dialog = (
        results.groupby(["model_name", "prompt_variant", "source", "dialog_id"], dropna=False)
        .agg(
            dialog_execution_accuracy=("execution_score", "mean"),
            dialog_strict_execution_accuracy=("strict_execution_score", "mean"),
            dialog_value_execution_accuracy=("value_execution_score", "mean"),
            dialog_syntax_accuracy=("syntax_valid", "mean"),
            interaction_match=("execution_score", lambda values: bool((values == 1.0).all())),
            turns=("execution_score", "count"),
            result_files=("result_file", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
    )
    return (
        per_dialog.groupby(["model_name", "prompt_variant", "source"], dropna=False)
        .agg(
            dialog_execution_accuracy=("dialog_execution_accuracy", "mean"),
            dialog_strict_execution_accuracy=("dialog_strict_execution_accuracy", "mean"),
            dialog_value_execution_accuracy=("dialog_value_execution_accuracy", "mean"),
            dialog_syntax_accuracy=("dialog_syntax_accuracy", "mean"),
            interaction_match_rate=("interaction_match", "mean"),
            dialogs=("dialog_id", "count"),
            turns=("turns", "sum"),
            result_files=("result_files", lambda values: ",".join(sorted(set(values)))),
        )
        .reset_index()
    )


def plot_accuracy_latency(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    summary = summary.copy()
    summary["plot_label"] = summary.apply(
        lambda row: (
            f"{row['model_name']} [{row['prompt_variant']}]"
            if row.get("prompt_variant")
            else row["model_name"]
        ),
        axis=1,
    )
    for plot_label, group in summary.groupby("plot_label"):
        ax.scatter(group["mean_latency_ms"], group["accuracy"], label=plot_label, s=80)
        for _, row in group.iterrows():
            ax.annotate(row["source"], (row["mean_latency_ms"], row["accuracy"]), fontsize=8)
    ax.set_xlabel("Mean generation latency (ms)")
    ax.set_ylabel("SQL accuracy")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Glob pattern relative to --results-dir. Repeat to include multiple result sets.",
    )
    args = parser.parse_args()

    includes = args.include or ["*.jsonl"]
    results = load_results(args.results_dir, includes)
    summary = summarize_results(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    dialog_summary = summarize_dialog_results(results)
    dialog_summary_path = args.output_dir / "dialog_summary.csv"
    if not dialog_summary.empty:
        dialog_summary.to_csv(dialog_summary_path, index=False)
    plot_accuracy_latency(summary, args.output_dir / "pareto_accuracy_latency.png")
    print(f"Wrote {summary_path}")
    if not dialog_summary.empty:
        print(f"Wrote {dialog_summary_path}")
    print(f"Wrote {args.output_dir / 'pareto_accuracy_latency.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
