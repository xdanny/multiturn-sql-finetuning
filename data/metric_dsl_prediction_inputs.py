"""Build paired metric-DSL and direct-SQL prediction input rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from data.metric_dsl_training_rows import (
    DIRECT_SQL_SYSTEM_PROMPT,
    DSL_TARGET_OVERRIDES,
    METRIC_DSL_SYSTEM_PROMPT,
    _conversation_text,
    _dsl_target,
)
from data.synthetic_method_fixtures import build_synthetic_method_fixtures
from eval.result_manifest import sha256_file

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "metric_dsl_prediction_inputs"
DEFAULT_METRIC_OUTPUT = Path("docs/data_artifacts/metric_dsl_prediction_inputs.jsonl")
DEFAULT_DIRECT_OUTPUT = Path("docs/data_artifacts/metric_dsl_direct_sql_prediction_inputs.jsonl")
DEFAULT_SUMMARY_OUTPUT = Path("docs/data_artifacts/metric_dsl_prediction_inputs_summary.json")
DEFAULT_MANIFEST_OUTPUT = Path("docs/data_artifacts/metric_dsl_prediction_inputs.manifest.json")


def _metric_user_prompt(fixture: dict[str, Any]) -> str:
    visible = fixture["prompt_visible_input"]
    return (
        f"Fixture: {visible['fixture_id']}\n\n"
        f"Schema:\n{visible['schema_sql']}\n\n"
        f"Semantic model:\n{json.dumps(visible['semantic_model'], indent=2, sort_keys=True)}\n\n"
        f"Conversation:\n{_conversation_text(visible['conversation'])}\n\n"
        "Return only the metric DSL."
    )


def _direct_user_prompt(fixture: dict[str, Any]) -> str:
    visible = fixture["prompt_visible_input"]
    return (
        f"Fixture: {visible['fixture_id']}\n\n"
        f"Schema:\n{visible['schema_sql']}\n\n"
        f"Semantic model:\n{json.dumps(visible['semantic_model'], indent=2, sort_keys=True)}\n\n"
        f"Conversation:\n{_conversation_text(visible['conversation'])}\n\n"
        "Return only SQL."
    )


def _row(
    *,
    fixture: dict[str, Any],
    generation_target: str,
    system_prompt: str,
    user_prompt: str,
    gold_dsl: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "id": fixture["fixture_id"],
        "fixture_id": fixture["fixture_id"],
        "schema_id": fixture["schema_id"],
        "generation_target": generation_target,
        "failure_modes": fixture["failure_modes"],
        "required_artifacts": fixture["required_artifacts"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "semantic_model": fixture["semantic_model"],
        "semantic_model_source": "synthetic_prompt_visible_fixture",
        "reference_sql": fixture["reference_sql"],
        "gold_dsl": gold_dsl,
        "database_id": fixture["schema_id"],
        "reference_sql_visible_to_model": False,
        "scoring_fields_visible_to_model": False,
        "leakage_policy": "prompt_with_held_out_scorer_fields",
        "label_source": fixture["label_source"],
        "comparison_contract": "same_fixture_metric_dsl_vs_direct_sql_predictions",
    }


def build_metric_dsl_prediction_inputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return paired generation inputs for the metric-DSL comparison gate."""

    metric_rows = []
    direct_rows = []
    for fixture in build_synthetic_method_fixtures():
        if "metric_dsl" not in fixture["training_targets"]:
            continue
        gold_dsl = _dsl_target(fixture)
        if not gold_dsl:
            continue
        metric_rows.append(
            _row(
                fixture=fixture,
                generation_target="metric_dsl",
                system_prompt=METRIC_DSL_SYSTEM_PROMPT,
                user_prompt=_metric_user_prompt(fixture),
                gold_dsl=gold_dsl,
            )
        )
        direct_rows.append(
            _row(
                fixture=fixture,
                generation_target="direct_sql",
                system_prompt=DIRECT_SQL_SYSTEM_PROMPT,
                user_prompt=_direct_user_prompt(fixture),
                gold_dsl=gold_dsl,
            )
        )
    return metric_rows, direct_rows


def summarize_metric_dsl_prediction_inputs(
    metric_rows: list[dict[str, Any]],
    direct_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_modes = Counter(mode for row in metric_rows for mode in row["failure_modes"])
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "metric_dsl_prediction_input_count": len(metric_rows),
        "direct_sql_prediction_input_count": len(direct_rows),
        "fixture_ids": [row["fixture_id"] for row in metric_rows],
        "failure_mode_counts": dict(sorted(failure_modes.items())),
        "leakage_policy": "prompt_with_held_out_scorer_fields",
        "comparison_contract": "same_fixture_metric_dsl_vs_direct_sql_predictions",
        "evaluation_command": (
            "eval.run_metric_dsl_comparison after generated predictions are added "
            "to these same row identities"
        ),
        "dsl_target_overrides": sorted(DSL_TARGET_OVERRIDES),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_metric_dsl_prediction_input_artifacts(
    *,
    metric_output_path: Path = DEFAULT_METRIC_OUTPUT,
    direct_output_path: Path = DEFAULT_DIRECT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY_OUTPUT,
    manifest_path: Path = DEFAULT_MANIFEST_OUTPUT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Write paired metric-DSL/direct-SQL prediction input contracts."""

    metric_rows, direct_rows = build_metric_dsl_prediction_inputs()
    summary = summarize_metric_dsl_prediction_inputs(metric_rows, direct_rows)
    _write_jsonl(metric_output_path, metric_rows)
    _write_jsonl(direct_output_path, direct_rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "metric_dsl_prediction_input_count": len(metric_rows),
        "direct_sql_prediction_input_count": len(direct_rows),
        "metric_output_path": str(metric_output_path),
        "metric_output_sha256": sha256_file(metric_output_path),
        "direct_output_path": str(direct_output_path),
        "direct_output_sha256": sha256_file(direct_output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "leakage_policy": "prompt_with_held_out_scorer_fields",
        "evaluation_command": "eval.run_metric_dsl_comparison",
        "command": command or sys.argv,
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric-output", type=Path, default=DEFAULT_METRIC_OUTPUT)
    parser.add_argument("--direct-output", type=Path, default=DEFAULT_DIRECT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()

    manifest = write_metric_dsl_prediction_input_artifacts(
        metric_output_path=args.metric_output,
        direct_output_path=args.direct_output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(
        "Wrote "
        f"{manifest['metric_dsl_prediction_input_count']} metric DSL prediction inputs and "
        f"{manifest['direct_sql_prediction_input_count']} direct SQL prediction inputs"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
