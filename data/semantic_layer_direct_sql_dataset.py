"""Build direct-SQL control rows for semantic-layer comparison."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from data.metric_dsl_dataset import _conversation_text, _load_jsonl, _write_json, _write_jsonl
from data.semantic_layer_dataset import FORBIDDEN_PROMPT_MARKERS
from data.synthetic_method_fixtures import DEFAULT_OUTPUT as DEFAULT_FIXTURE_INPUT
from data.synthetic_method_fixtures import build_synthetic_method_fixtures
from eval.result_manifest import sha256_file

ARTIFACT_TYPE = "semantic_layer_direct_sql_training_rows"
BENCHMARK = "synthetic_semantic_layer_direct_sql"
DEFAULT_OUTPUT = Path("docs/data_artifacts/semantic_layer_direct_sql_training_rows.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path(
    "docs/data_artifacts/semantic_layer_direct_sql_training_rows_summary.json"
)
DEFAULT_MANIFEST_OUTPUT = Path(
    "docs/data_artifacts/semantic_layer_direct_sql_training_rows.manifest.json"
)


def _direct_sql_system_prompt() -> str:
    return (
        "You write analytical SQL for multi-turn data analysis. Use the visible schema and "
        "conversation. Respond with only SQL."
    )


def _direct_sql_user_prompt(fixture: dict[str, Any]) -> str:
    visible = fixture["prompt_visible_input"]
    prompt = (
        "You are learning a direct-SQL control for multi-turn data analysis.\n"
        "Respond with only SQL.\n\n"
        f"Schema SQL:\n{visible['schema_sql']}\n\n"
        f"Conversation:\n{_conversation_text(visible['conversation'])}\n"
    )
    for marker in FORBIDDEN_PROMPT_MARKERS:
        if marker in prompt:
            raise ValueError(f"direct-SQL control prompt leaked forbidden marker: {marker}")
    return prompt


def build_semantic_layer_direct_sql_training_rows(fixtures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        if "semantic_layer" not in fixture.get("training_targets", []):
            continue
        rows.append(
            {
                "artifact_type": ARTIFACT_TYPE,
                "benchmark": BENCHMARK,
                "training_target": "direct_sql_control",
                "evaluation_mode": "non_oracle_generation",
                "fixture_id": fixture["fixture_id"],
                "source_artifact_type": fixture["artifact_type"],
                "failure_modes": fixture["failure_modes"],
                "required_artifacts": fixture["required_artifacts"],
                "oracle_policy": fixture["oracle_policy"],
                "claim_boundary": fixture["claim_boundary"],
                "reference_sql": fixture["reference_sql"],
                "messages": [
                    {"role": "system", "content": _direct_sql_system_prompt()},
                    {"role": "user", "content": _direct_sql_user_prompt(fixture)},
                    {"role": "assistant", "content": fixture["reference_sql"]},
                ],
            }
        )
    return rows


def summarize_semantic_layer_direct_sql_training_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    failure_mode_counts: Counter[str] = Counter()
    fixture_ids: list[str] = []
    for row in rows:
        failure_mode_counts.update(row["failure_modes"])
        fixture_ids.append(str(row["fixture_id"]))
    return {
        "artifact_type": ARTIFACT_TYPE,
        "training_target": "direct_sql_control",
        "benchmark": BENCHMARK,
        "row_count": len(rows),
        "fixture_ids": fixture_ids,
        "failure_mode_counts": dict(sorted(failure_mode_counts.items())),
    }


def write_semantic_layer_direct_sql_training_artifacts(
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

    rows = build_semantic_layer_direct_sql_training_rows(fixtures)
    summary = summarize_semantic_layer_direct_sql_training_rows(rows)
    _write_jsonl(output_path, rows)
    _write_json(summary_path, summary)
    manifest = {
        "artifact_type": ARTIFACT_TYPE,
        "training_target": "direct_sql_control",
        "benchmark": BENCHMARK,
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

    manifest = write_semantic_layer_direct_sql_training_artifacts(
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        fixtures_path=args.fixtures,
        command=sys.argv,
    )
    print(
        f"Wrote {manifest['row_count']} semantic-layer direct-SQL control rows to {args.output}"
    )
    print(f"Wrote summary to {args.summary_output}")
    print(f"Wrote manifest to {args.manifest_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
