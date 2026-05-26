"""Build metric-DSL finetuning rows from curated synthetic fixtures."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from data.synthetic_method_fixtures import (
    DEFAULT_OUTPUT as DEFAULT_FIXTURE_INPUT,
)
from data.synthetic_method_fixtures import (
    build_synthetic_method_fixtures,
)
from eval.result_manifest import sha256_file

ARTIFACT_TYPE = "metric_dsl_training_rows"
DEFAULT_OUTPUT = Path("docs/data_artifacts/metric_dsl_training_rows.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/metric_dsl_training_rows_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/metric_dsl_training_rows.manifest.json")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _conversation_text(conversation: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for turn in conversation:
        turn_id = turn.get("turn")
        if "user" in turn:
            lines.append(f"Turn {turn_id} user: {turn['user']}")
        if "assistant_sql" in turn:
            lines.append(f"Turn {turn_id} previous SQL: {turn['assistant_sql']}")
        if "observed_previous_rows" in turn:
            lines.append(
                f"Turn {turn_id} observed rows: {json.dumps(turn['observed_previous_rows'])}"
            )
    return "\n".join(lines)


def _metric_dsl_user_prompt(fixture: dict[str, Any]) -> str:
    visible = fixture["prompt_visible_input"]
    semantic_model = json.dumps(visible["semantic_model"], sort_keys=True)
    checks = json.dumps(fixture["evaluation_checks"], sort_keys=True)
    return (
        "You are learning a governed metric DSL for multi-turn data analysis.\n"
        "Respond with only the metric DSL query.\n\n"
        f"Schema SQL:\n{visible['schema_sql']}\n\n"
        f"Semantic model:\n{semantic_model}\n\n"
        f"Conversation:\n{_conversation_text(visible['conversation'])}\n\n"
        f"Behavior checks:\n{checks}\n"
    )


def _metric_dsl_system_prompt() -> str:
    return (
        "You write only governed metric DSL. Preserve MEASURE() tokens, keep business "
        "dimensions explicit, and do not emit SQL."
    )


def build_metric_dsl_training_rows(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        if "metric_dsl" not in fixture.get("training_targets", []):
            continue
        gold_dsl = fixture.get("gold_metric_dsl")
        if not gold_dsl:
            continue
        rows.append(
            {
                "artifact_type": ARTIFACT_TYPE,
                "benchmark": "synthetic_metric_dsl_bootstrap",
                "training_target": "metric_dsl",
                "evaluation_mode": "metric_dsl",
                "fixture_id": fixture["fixture_id"],
                "source_artifact_type": fixture["artifact_type"],
                "failure_modes": fixture["failure_modes"],
                "required_artifacts": fixture["required_artifacts"],
                "oracle_policy": fixture["oracle_policy"],
                "claim_boundary": fixture["claim_boundary"],
                "semantic_model": fixture["semantic_model"],
                "gold_dsl": gold_dsl,
                "reference_sql": fixture["reference_sql"],
                "messages": [
                    {"role": "system", "content": _metric_dsl_system_prompt()},
                    {"role": "user", "content": _metric_dsl_user_prompt(fixture)},
                    {"role": "assistant", "content": gold_dsl},
                ],
            }
        )
    return rows


def summarize_metric_dsl_training_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_mode_counts: Counter[str] = Counter()
    semantic_model_ids: set[str] = set()
    fixture_ids: list[str] = []
    for row in rows:
        failure_mode_counts.update(row["failure_modes"])
        fixture_ids.append(str(row["fixture_id"]))
        semantic_model_ids.add(str(row["semantic_model"]["semantic_model_id"]))
    return {
        "artifact_type": ARTIFACT_TYPE,
        "training_target": "metric_dsl",
        "row_count": len(rows),
        "fixture_ids": fixture_ids,
        "semantic_model_ids": sorted(semantic_model_ids),
        "failure_mode_counts": dict(sorted(failure_mode_counts.items())),
    }


def write_metric_dsl_training_artifacts(
    *,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    fixtures_path: Path | None = DEFAULT_FIXTURE_INPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    if fixtures_path is not None and fixtures_path.exists():
        fixtures = _load_jsonl(fixtures_path)
        fixture_sha256 = sha256_file(fixtures_path)
        fixture_path_value = str(fixtures_path)
    else:
        fixtures = build_synthetic_method_fixtures()
        fixture_sha256 = None
        fixture_path_value = None

    rows = build_metric_dsl_training_rows(fixtures)
    summary = summarize_metric_dsl_training_rows(rows)
    _write_jsonl(output_path, rows)
    _write_json(summary_path, summary)
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "training_target": "metric_dsl",
        "row_count": len(rows),
        "fixture_source_path": fixture_path_value,
        "fixture_source_sha256": fixture_sha256,
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURE_INPUT)
    args = parser.parse_args()

    manifest = write_metric_dsl_training_artifacts(
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        fixtures_path=args.fixtures,
        command=sys.argv,
    )
    print(f"Wrote {manifest['row_count']} metric DSL finetuning rows to {args.output}")
    print(f"Wrote summary to {args.summary_output}")
    print(f"Wrote manifest to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
