"""Validate and summarize the ordered finetuning steps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from eval.benchmark_protocols import DEFAULT_PROTOCOL_CONFIG, benchmark_protocol_map
from eval.method_readiness import DEFAULT_METHOD_CONFIG, load_method_configs

DEFAULT_STEP_CONFIG = Path("configs/finetuning_steps.yaml")

LIST_FIELDS = {
    "benchmark_protocol_ids",
    "train_rows",
    "eval_rows",
    "control_rows",
    "commands",
    "clears_claim_ids",
    "blocks_claim_ids",
}
REQUIRED_TEXT_FIELDS = {
    "step_id",
    "method",
    "stage",
    "purpose",
    "evidence_gate",
    "leakage_boundary",
}


def _path_status(repo_root: Path, paths: tuple[str, ...]) -> dict[str, bool]:
    return {
        path: "<" in path or ">" in path or (repo_root / path).exists()
        for path in paths
    }


def _method_map(path: Path) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in load_method_configs(path)}


def load_finetuning_steps(
    path: Path = DEFAULT_STEP_CONFIG,
    *,
    method_config_path: Path = DEFAULT_METHOD_CONFIG,
    protocol_config_path: Path = DEFAULT_PROTOCOL_CONFIG,
    repo_root: Path = Path("."),
) -> tuple[dict[str, Any], ...]:
    """Return validated finetuning steps with path readiness metadata."""

    payload = yaml.safe_load(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("finetuning step config must use schema_version=1")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("finetuning step config must include non-empty steps")

    methods = _method_map(method_config_path)
    protocols = benchmark_protocol_map(protocol_config_path)
    normalized = []
    seen_ids = set()
    for step in steps:
        row = dict(step)
        step_id = row.get("step_id") or "<unknown step>"
        missing_text_fields = [
            field
            for field in sorted(REQUIRED_TEXT_FIELDS)
            if not str(row.get(field) or "").strip()
        ]
        if missing_text_fields:
            raise ValueError(f"{step_id}: missing {', '.join(missing_text_fields)}")
        if step_id in seen_ids:
            raise ValueError(f"{step_id}: duplicate finetuning step id")
        seen_ids.add(step_id)
        if row["method"] not in methods:
            raise ValueError(f"{step_id}: unknown method {row['method']}")
        for field in LIST_FIELDS:
            row[field] = tuple(row.get(field) or ())
        if not row["benchmark_protocol_ids"]:
            raise ValueError(f"{step_id}: missing benchmark_protocol_ids")
        unknown_protocol_ids = [
            protocol_id
            for protocol_id in row["benchmark_protocol_ids"]
            if protocol_id not in protocols
        ]
        if unknown_protocol_ids:
            raise ValueError(
                f"{step_id}: unknown benchmark protocols: {', '.join(unknown_protocol_ids)}"
            )
        if not row["commands"]:
            raise ValueError(f"{step_id}: missing commands")
        non_uv_commands = [
            command
            for command in row["commands"]
            if not command.startswith("uv run --active --no-sync")
        ]
        if non_uv_commands:
            raise ValueError(f"{step_id}: commands must use uv run --active --no-sync")
        if any("reference SQL" in command for command in row["commands"]):
            raise ValueError(f"{step_id}: command text mentions reference SQL")
        row["train_rows_status"] = _path_status(repo_root, row["train_rows"])
        row["eval_rows_status"] = _path_status(repo_root, row["eval_rows"])
        row["control_rows_status"] = _path_status(repo_root, row["control_rows"])
        normalized.append(row)
    return tuple(normalized)


def finetuning_step_summary(
    *,
    path: Path = DEFAULT_STEP_CONFIG,
    method_config_path: Path = DEFAULT_METHOD_CONFIG,
    protocol_config_path: Path = DEFAULT_PROTOCOL_CONFIG,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Return a JSON-serializable summary for docs and review."""

    steps = load_finetuning_steps(
        path,
        method_config_path=method_config_path,
        protocol_config_path=protocol_config_path,
        repo_root=repo_root,
    )
    return {
        "schema_version": 1,
        "step_count": len(steps),
        "steps": [
            {
                "step_id": step["step_id"],
                "method": step["method"],
                "stage": step["stage"],
                "benchmark_protocol_ids": list(step["benchmark_protocol_ids"]),
                "train_rows_ready": all(step["train_rows_status"].values()),
                "eval_rows_ready": all(step["eval_rows_status"].values()),
                "control_rows_ready": all(step["control_rows_status"].values()),
                "evidence_gate": step["evidence_gate"],
                "clears_claim_ids": list(step["clears_claim_ids"]),
                "blocks_claim_ids": list(step["blocks_claim_ids"]),
                "command_count": len(step["commands"]),
            }
            for step in steps
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-config", type=Path, default=DEFAULT_STEP_CONFIG)
    parser.add_argument("--method-config", type=Path, default=DEFAULT_METHOD_CONFIG)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL_CONFIG)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = finetuning_step_summary(
        path=args.steps_config,
        method_config_path=args.method_config,
        protocol_config_path=args.protocol_config,
        repo_root=Path.cwd(),
    )
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {summary['step_count']} finetuning steps to {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
