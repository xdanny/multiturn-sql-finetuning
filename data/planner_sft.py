"""Build train-split planner SFT rows from prepared SQL conversations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from data.plan_contract import normalize_plan
from eval.planner_predict import planner_messages_for_record
from eval.run_eval import load_prepared_records

PLANNER_SFT_LABEL_SOURCE = "train_split_schema_link_labels_preferred"
PLANNER_SFT_ORACLE_POLICY = "train_split_supervision_current_answer_key_not_in_prompt"
PLANNER_SFT_PROMPT_POLICY = "projection_sequence_instruction_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_plan_json(plan: dict[str, Any]) -> str:
    return json.dumps(plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _prompt_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages)


def _planner_prompt_record(turn: dict[str, Any]) -> dict[str, Any]:
    """Drop teacher-forced SQL history before building the planner prompt."""

    return {
        **turn,
        "messages": [
            dict(message)
            for message in turn["messages"]
            if message.get("role") in {"system", "user"}
        ],
    }


def planner_sft_record_from_turn(turn: dict[str, Any]) -> dict[str, Any]:
    """Return one planner-SFT row for an expanded train-split turn."""

    if turn.get("split_role") != "train":
        raise ValueError(f"{turn.get('id')}: planner SFT rows must come from split_role=train")
    gold_plan = normalize_plan(turn.get("gold_plan"))
    prompt_messages = planner_messages_for_record(_planner_prompt_record(turn))
    current_reference_sql = str(turn.get("reference_sql") or "")
    prompt_text = _prompt_text(prompt_messages)
    if current_reference_sql and current_reference_sql in prompt_text:
        raise ValueError(f"{turn.get('id')}: current reference SQL leaked into planner prompt")
    if any(message.get("role") == "assistant" for message in prompt_messages):
        raise ValueError(f"{turn.get('id')}: assistant SQL history leaked into planner prompt")

    target = _target_plan_json(gold_plan)
    return {
        "id": str(turn["id"]),
        "dialog_id": str(turn.get("dialog_id") or ""),
        "turn_index": int(turn.get("turn_index") or 0),
        "turn_count": int(turn.get("turn_count") or 0),
        "database_id": turn.get("database_id"),
        "source": turn.get("source"),
        "split_id": turn.get("split_id"),
        "split_role": turn.get("split_role"),
        "split_row_id": turn.get("split_row_id"),
        "history_policy": turn.get("history_policy"),
        "messages": [*prompt_messages, {"role": "assistant", "content": target}],
        "target_plan": gold_plan,
        "planner_label_source": PLANNER_SFT_LABEL_SOURCE,
        "oracle_policy": PLANNER_SFT_ORACLE_POLICY,
        "planner_prompt_policy": PLANNER_SFT_PROMPT_POLICY,
        "reference_sql_visible_to_model": False,
        "gold_plan_visible_to_model_prompt": False,
        "target_plan_visible_as_assistant_label": True,
    }


def build_planner_sft_records(input_path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    turns = load_prepared_records(input_path, limit=limit, allow_oracle_plan=False)
    return [planner_sft_record_from_turn(turn) for turn in turns]


def write_jsonl(rows: list[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def build_planner_sft_manifest(
    *,
    rows: list[dict[str, Any]],
    input_path: Path,
    output_path: Path,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    split_ids = Counter(str(row.get("split_id") or "unknown") for row in rows)
    split_roles = Counter(str(row.get("split_role") or "unknown") for row in rows)
    database_ids = {str(row.get("database_id")) for row in rows if row.get("database_id")}
    return {
        "schema_version": 1,
        "artifact_type": "planner_sft_dataset",
        "row_count": len(rows),
        "dialog_count": len({row.get("dialog_id") for row in rows}),
        "database_count": len(database_ids),
        "split_ids": dict(split_ids),
        "split_roles": dict(split_roles),
        "planner_label_source": PLANNER_SFT_LABEL_SOURCE,
        "oracle_policy": PLANNER_SFT_ORACLE_POLICY,
        "planner_prompt_policy": PLANNER_SFT_PROMPT_POLICY,
        "reference_sql_visible_to_model": False,
        "gold_plan_visible_to_model_prompt": False,
        "target_plan_visible_as_assistant_label": True,
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path) if input_path.exists() else None,
        "output_path": str(output_path),
        "output_sha256": _sha256_file(output_path) if output_path.exists() else None,
        "command": list(command or []),
    }


def write_planner_sft_dataset(
    *,
    input_path: Path,
    output_path: Path,
    manifest_output_path: Path,
    limit: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    rows = build_planner_sft_records(input_path, limit=limit)
    write_jsonl(rows, output_path)
    manifest = build_planner_sft_manifest(
        rows=rows,
        input_path=input_path,
        output_path=output_path,
        command=command,
    )
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = write_planner_sft_dataset(
        input_path=args.input,
        output_path=args.output,
        manifest_output_path=args.manifest_output,
        limit=args.limit,
        command=sys.argv,
    )
    print(
        f"Wrote {args.output} and {args.manifest_output} "
        f"({manifest['row_count']} planner SFT rows)"
    )
    return 0 if manifest["row_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
