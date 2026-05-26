"""Freeze the non-oracle prepared slice for same-protocol hosted comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

ARTIFACT_TYPE = "hosted_baseline_dataset"
BENCHMARK = "prepared"
EVALUATION_MODE = "non_oracle_generation"
DEFAULT_INPUT = Path("data/processed/eval_cosql_dev_100.jsonl")
DEFAULT_OUTPUT = Path("docs/data_artifacts/hosted_baseline_rows.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/hosted_baseline_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/hosted_baseline.manifest.json")
FORBIDDEN_PROMPT_MARKERS = ("Oracle SQL planning hints", "SQL planning hints:")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _row_uses_oracle(row: dict[str, Any]) -> bool:
    if row.get("uses_oracle_planning_hints") or row.get("semantic_context_pruned_by_oracle_labels"):
        return True
    return any(
        marker in str(message.get("content", ""))
        for message in row.get("messages", [])
        for marker in FORBIDDEN_PROMPT_MARKERS
    )


def build_hosted_baseline_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packaged: list[dict[str, Any]] = []
    for row in rows:
        if _row_uses_oracle(row):
            raise ValueError("hosted baseline rows must be non-oracle")
        packaged.append(
            {
                **row,
                "artifact_type": ARTIFACT_TYPE,
                "benchmark": BENCHMARK,
                "training_target": "hosted_baseline_candidate",
                "evaluation_mode": row.get("evaluation_mode", EVALUATION_MODE),
                "oracle_policy": "non_oracle_inputs_only",
            }
        )
    return packaged


def summarize_hosted_baseline_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "artifact_type": ARTIFACT_TYPE,
        "benchmark": BENCHMARK,
        "evaluation_mode": EVALUATION_MODE,
        "row_count": len(rows),
        "dialog_count": len({str(row.get("dialog_id") or row.get("database_id")) for row in rows}),
        "database_count": len({str(row.get("database_id")) for row in rows}),
    }


def write_hosted_baseline_artifacts(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    rows = build_hosted_baseline_rows(_load_jsonl(input_path))
    summary = summarize_hosted_baseline_rows(rows)
    _write_jsonl(output_path, rows)
    _write_json(summary_path, summary)
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "benchmark": BENCHMARK,
        "evaluation_mode": EVALUATION_MODE,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    manifest = write_hosted_baseline_artifacts(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(f"Wrote hosted baseline rows to {args.output}")
    print(f"Wrote hosted baseline manifest to {args.manifest_output}")
    return 0 if manifest else 1


if __name__ == "__main__":
    raise SystemExit(main())
