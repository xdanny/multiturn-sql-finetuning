"""
Compare classified SQL benchmark results across models and prompt variants.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _run_label(row: dict[str, Any]) -> str:
    model = row.get("model_name", "")
    variant = row.get("prompt_variant") or ""
    return f"{model}[{variant}]" if variant else model


def load_classified_files(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    runs: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if not rows:
            continue
        label = _run_label(rows[0])
        runs[label] = rows
    if not runs:
        raise ValueError("no classified rows loaded")
    return runs


def write_model_summary(runs: dict[str, list[dict[str, Any]]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "samples",
        "value_accuracy",
        "strict_accuracy",
        "syntax_accuracy",
        "interaction_match_rate",
        "correct",
        "schema_link",
        "join_path",
        "aggregation",
        "grain_fanout",
        "history_resolution",
        "ordering_limit",
        "value_grounding",
        "projection",
        "invalid_sql",
        "execution_error",
        "other",
        "history_resolution_turns",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, rows in sorted(runs.items()):
            counts = Counter(row["error_primary"] for row in rows)
            history_resolution_turns = sum(
                row.get("error_primary") == "history_resolution"
                or "history_resolution" in (row.get("error_secondary") or [])
                for row in rows
            )
            dialog_scores: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                dialog_scores[str(row.get("dialog_id", row.get("id", "")))].append(
                    float(row.get("value_execution_score") or row.get("execution_score") or 0.0)
                )
            interaction_match_rate = (
                sum(all(score == 1.0 for score in scores) for scores in dialog_scores.values())
                / len(dialog_scores)
                if dialog_scores
                else 0.0
            )
            writer.writerow(
                {
                    "run": label,
                    "samples": len(rows),
                    "value_accuracy": sum(float(row.get("value_execution_score") or 0.0) for row in rows)
                    / len(rows),
                    "strict_accuracy": sum(float(row.get("strict_execution_score") or 0.0) for row in rows)
                    / len(rows),
                    "syntax_accuracy": sum(bool(row.get("syntax_valid")) for row in rows) / len(rows),
                    "interaction_match_rate": interaction_match_rate,
                    **{name: counts.get(name, 0) for name in fieldnames[6:-1]},
                    "history_resolution_turns": history_resolution_turns,
                }
            )


def write_pairwise_comparison(
    runs: dict[str, list[dict[str, Any]]],
    *,
    baseline_label: str,
    output: Path,
) -> None:
    if baseline_label not in runs:
        labels = ", ".join(sorted(runs))
        raise ValueError(f"baseline {baseline_label!r} not found. Available: {labels}")
    baseline_by_id = {str(row["id"]): row for row in runs[baseline_label]}
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate",
        "baseline",
        "samples_compared",
        "fixed_turns",
        "regressed_turns",
        "both_correct",
        "both_wrong",
        "net_fixed",
        "fixed_error_primary",
        "regressed_error_primary",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, rows in sorted(runs.items()):
            if label == baseline_label:
                continue
            fixed = []
            regressed = []
            both_correct = 0
            both_wrong = 0
            compared = 0
            for row in rows:
                row_id = str(row["id"])
                if row_id not in baseline_by_id:
                    continue
                compared += 1
                baseline = baseline_by_id[row_id]
                baseline_correct = float(baseline.get("value_execution_score") or 0.0) == 1.0
                candidate_correct = float(row.get("value_execution_score") or 0.0) == 1.0
                if candidate_correct and not baseline_correct:
                    fixed.append(baseline["error_primary"])
                elif baseline_correct and not candidate_correct:
                    regressed.append(row["error_primary"])
                elif candidate_correct and baseline_correct:
                    both_correct += 1
                else:
                    both_wrong += 1
            writer.writerow(
                {
                    "candidate": label,
                    "baseline": baseline_label,
                    "samples_compared": compared,
                    "fixed_turns": len(fixed),
                    "regressed_turns": len(regressed),
                    "both_correct": both_correct,
                    "both_wrong": both_wrong,
                    "net_fixed": len(fixed) - len(regressed),
                    "fixed_error_primary": json.dumps(Counter(fixed), sort_keys=True),
                    "regressed_error_primary": json.dumps(Counter(regressed), sort_keys=True),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", required=True)
    args = parser.parse_args()

    runs = load_classified_files(args.input)
    write_model_summary(runs, args.output_dir / "model_error_summary.csv")
    write_pairwise_comparison(
        runs,
        baseline_label=args.baseline,
        output=args.output_dir / "pairwise_vs_baseline.csv",
    )
    print(f"Wrote {args.output_dir / 'model_error_summary.csv'}")
    print(f"Wrote {args.output_dir / 'pairwise_vs_baseline.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
