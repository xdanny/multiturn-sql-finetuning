"""Build generated-history recovery clean-holdout rollout candidate rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file
from eval.rollout_eval import MODEL_GENERATED_SQL_ROLLOUT, load_rollout_prepared_records

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "generated_history_recovery_rollout_candidate"
SUMMARY_ARTIFACT_TYPE = "generated_history_recovery_readiness_summary"
MANIFEST_ARTIFACT_TYPE = "generated_history_recovery_candidate_manifest"
REQUIRED_SPLIT_ROLE = "clean_local_holdout"
DEFAULT_INPUT = Path("data/processed/direct_sql_full/cosql_dev_clean_holdout_v1.jsonl")
DEFAULT_OUTPUT = Path("data/processed/generated_history_recovery/clean_holdout_candidates.jsonl")
DEFAULT_SUMMARY = Path(
    "docs/training_runs/generated_history_recovery_readiness_20260602.json"
)
DEFAULT_MANIFEST = Path(
    "docs/data_artifacts/generated_history_recovery_candidates.manifest.json"
)
RECOVERY_ROLLOUT_BLOCKER = "multi-dialog recovery adapter rollout missing"
NEXT_STEP = (
    "run eval.run_behavior_recovery_comparison on this candidate slice for the "
    "recovery adapter and the direct-SQL control, then compare identical "
    "model-generated-history rollout rows"
)


def _assistant_turn_count(record: dict[str, Any]) -> int:
    return sum(1 for message in record.get("messages", []) if message.get("role") == "assistant")


def _candidate_from_record(record: dict[str, Any]) -> dict[str, Any] | None:
    if record.get("split_role") != REQUIRED_SPLIT_ROLE:
        raise ValueError(
            f"{record.get('dialog_id') or record.get('id')}: generated-history "
            f"recovery candidates require split_role={REQUIRED_SPLIT_ROLE}"
        )
    assistant_turn_count = _assistant_turn_count(record)
    if assistant_turn_count < 2:
        return None
    return {
        **record,
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "assistant_turn_count": assistant_turn_count,
        "recoverable_later_turn_count": assistant_turn_count - 1,
        "rollout_target_history_policy": MODEL_GENERATED_SQL_ROLLOUT,
        "oracle_policy": "non_oracle_generation",
        "comparison_contract": "generated_history_recovery_clean_holdout_rollout",
        "reference_sql_visible_to_model_prompt": False,
        "future_turns_visible_to_model_prompt": False,
        "scorer_labels_visible_to_model_prompt": False,
        "readiness_blockers": [RECOVERY_ROLLOUT_BLOCKER],
    }


def build_generated_history_recovery_candidates(input_path: Path) -> list[dict[str, Any]]:
    """Return multi-turn clean-holdout dialogs usable for recovery rollout."""

    rows = []
    for record in load_rollout_prepared_records(input_path, allow_oracle_plan=False):
        candidate = _candidate_from_record(record)
        if candidate is not None:
            rows.append(candidate)
    return rows


def summarize_generated_history_recovery_candidates(
    rows: list[dict[str, Any]], *, input_path: Path
) -> dict[str, Any]:
    split_ids = Counter(str(row.get("split_id") or "unknown") for row in rows)
    split_roles = Counter(str(row.get("split_role") or "unknown") for row in rows)
    blockers = Counter(blocker for row in rows for blocker in row["readiness_blockers"])
    turn_count = sum(int(row.get("assistant_turn_count") or 0) for row in rows)
    later_turn_count = sum(int(row.get("recoverable_later_turn_count") or 0) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": SUMMARY_ARTIFACT_TYPE,
        "checkpoint": 8,
        "run_id": "generated_history_recovery_readiness_20260602",
        "claim_boundary": (
            "Clean-holdout generated-history recovery rollout readiness only; no "
            "recovery adapter rollout, direct-SQL comparison, value delta, or method "
            "win is claimed."
        ),
        "candidate_dialog_count": len(rows),
        "candidate_turn_count": turn_count,
        "recoverable_later_turn_count": later_turn_count,
        "database_count": len({row.get("database_id") for row in rows if row.get("database_id")}),
        "split_ids": dict(split_ids),
        "split_roles": dict(split_roles),
        "rollout_target_history_policy": MODEL_GENERATED_SQL_ROLLOUT,
        "oracle_policy": "non_oracle_generation",
        "comparison_contract": "generated_history_recovery_clean_holdout_rollout",
        "reference_sql_visible_to_model_prompt": False,
        "future_turns_visible_to_model_prompt": False,
        "scorer_labels_visible_to_model_prompt": False,
        "readiness_blockers": dict(sorted(blockers.items())),
        "promotion_status": "not_ready",
        "next_step": NEXT_STEP,
        "input_path": str(input_path),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return len(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_generated_history_recovery_candidate_artifacts(
    *,
    input_path: Path,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
    manifest_path: Path = DEFAULT_MANIFEST,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write generated-history recovery candidates, summary, and manifest."""

    rows = build_generated_history_recovery_candidates(input_path)
    summary = summarize_generated_history_recovery_candidates(rows, input_path=input_path)
    _write_jsonl(output_path, rows)
    _write_json(summary_path, summary)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": MANIFEST_ARTIFACT_TYPE,
        "candidate_dialog_count": len(rows),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path) if input_path.exists() else None,
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "oracle_policy": "non_oracle_generation",
        "promotion_status": "not_ready",
        "command": list(command or []),
    }
    _write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    manifest = write_generated_history_recovery_candidate_artifacts(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_output,
        manifest_path=args.manifest_output,
        command=sys.argv,
    )
    print(
        f"Wrote {manifest['candidate_dialog_count']} generated-history "
        f"recovery candidate dialogs to {args.output}"
    )
    return 0 if manifest["candidate_dialog_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
