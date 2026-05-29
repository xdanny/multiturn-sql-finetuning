"""Score value-choice consistency in rollout SQL generations."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

STRING_COMPARISON_RE = re.compile(
    r"(?P<table>[A-Za-z_][\w]*)\.(?P<column>[A-Za-z_][\w]*)\s*=\s*'(?P<value>[^']*)'",
    re.IGNORECASE,
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _extract_selected_value(sql: str, *, table: str, column: str) -> str | None:
    for match in STRING_COMPARISON_RE.finditer(sql):
        if (
            match.group("table").lower() == table.lower()
            and match.group("column").lower() == column.lower()
        ):
            return match.group("value")
    return None


def score_value_choice_consistency(
    *,
    input_path: Path,
    rollout_output_path: Path,
) -> dict[str, Any]:
    inputs_by_dialog = {
        str(row["dialog_id"]): row
        for row in _load_jsonl(input_path)
        if row.get("expected_value_choice")
    }
    output_rows = _load_jsonl(rollout_output_path)
    scored_rows = []
    for output in output_rows:
        dialog_id = str(output.get("dialog_id"))
        input_row = inputs_by_dialog.get(dialog_id)
        if not input_row:
            continue
        expected = input_row["expected_value_choice"]
        selected = _extract_selected_value(
            str(output.get("generated_sql") or ""),
            table=str(expected["table"]),
            column=str(expected["column"]),
        )
        scored_rows.append(
            {
                "id": output.get("id"),
                "dialog_id": dialog_id,
                "table": expected["table"],
                "column": expected["column"],
                "source_mention": expected["source_mention"],
                "expected_storage_value": expected["storage_value"],
                "selected_storage_value": selected,
                "value_choice_match": selected == expected["storage_value"],
            }
        )

    correct = sum(1 for row in scored_rows if row["value_choice_match"])
    total = len(scored_rows)
    return {
        "schema_version": 1,
        "artifact_type": "value_choice_consistency_score",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "rollout_output_path": str(rollout_output_path),
        "rollout_output_sha256": sha256_file(rollout_output_path),
        "scored_row_count": total,
        "value_choice_accuracy": correct / total if total else 0.0,
        "rows": scored_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rollout-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = score_value_choice_consistency(
        input_path=args.input,
        rollout_output_path=args.rollout_output,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        "Value-choice accuracy: "
        f"{payload['value_choice_accuracy']:.3f} over {payload['scored_row_count']} rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
