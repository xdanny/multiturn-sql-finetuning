"""
Re-score existing benchmark JSONL files without regenerating model outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.ragas_metrics import score_single_turn


def rescore_row(row: dict[str, Any]) -> dict[str, Any]:
    database_path = row.get("database_path")
    score = score_single_turn(
        row["reference_sql"],
        row["generated_sql"],
        database_path=Path(database_path) if database_path else None,
    )
    rescored = dict(row)
    rescored["original_execution_score"] = row.get("execution_score")
    rescored["execution_score"] = score.execution_score
    rescored["strict_execution_score"] = score.strict_execution_score
    rescored["value_execution_score"] = score.value_execution_score
    rescored["order_sensitive"] = score.order_sensitive
    rescored["normalized_match"] = score.normalized_match
    rescored["syntax_valid"] = score.syntax_valid
    rescored["score_error"] = score.error
    return rescored


def rescore_file(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open() as source, output_path.open("w") as target:
        for line in source:
            if not line.strip():
                continue
            target.write(json.dumps(rescore_row(json.loads(line)), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    count = rescore_file(args.input, args.output)
    print(f"Wrote {count} rescored rows to {args.output}")
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
