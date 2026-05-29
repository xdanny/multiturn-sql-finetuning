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
DEFAULT_CLAIM_LEDGER = Path("docs/claim_ledgers/cosql_dev_100.jsonl")

LIST_FIELDS = {
    "benchmark_protocol_ids",
    "train_rows",
    "eval_rows",
    "control_rows",
    "preflight_commands",
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
MEASUREMENT_REQUIRED_TEXT_FIELDS = {
    "primary_metric",
    "comparison_artifact",
    "promoted_when",
}
STAGE_ORDER = {
    "train_control": 0,
    "endpoint_comparison": 1,
    "train_and_compare": 2,
    "train_and_rollout": 3,
    "transfer_gate": 4,
}


def _path_status(repo_root: Path, paths: tuple[str, ...]) -> dict[str, bool]:
    return {
        path: "<" in path or ">" in path or (repo_root / path).exists()
        for path in paths
    }


def _method_map(path: Path) -> dict[str, dict[str, Any]]:
    return {row["method"]: row for row in load_method_configs(path)}


def _claim_ids(path: Path) -> set[str]:
    with path.open() as handle:
        return {
            str(json.loads(line)["claim_id"])
            for line in handle
            if line.strip()
        }


def _normalize_measurement(step_id: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{step_id}: missing measurement")
    measurement = dict(value)
    missing_text_fields = [
        field
        for field in sorted(MEASUREMENT_REQUIRED_TEXT_FIELDS)
        if not str(measurement.get(field) or "").strip()
    ]
    if missing_text_fields:
        raise ValueError(
            f"{step_id}: measurement missing {', '.join(missing_text_fields)}"
        )
    for field in ("benchmark_metric_refs", "supporting_metrics"):
        values = measurement.get(field)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{step_id}: measurement missing {field}")
        measurement[field] = tuple(str(value) for value in values)
    return measurement


def _validate_benchmark_metric_refs(
    *,
    step_id: str,
    benchmark_protocol_ids: tuple[str, ...],
    measurement: dict[str, Any],
    protocols: dict[str, dict[str, Any]],
) -> None:
    allowed_protocol_ids = set(benchmark_protocol_ids)
    for metric_ref in measurement["benchmark_metric_refs"]:
        protocol_id, separator, metric = metric_ref.partition(":")
        if not separator or not protocol_id or not metric:
            raise ValueError(
                f"{step_id}: measurement benchmark_metric_refs must use "
                "<protocol_id>:<metric>"
            )
        if protocol_id not in allowed_protocol_ids:
            raise ValueError(
                f"{step_id}: benchmark metric ref {metric_ref} does not belong "
                "to the step benchmark_protocol_ids"
            )
        primary_metrics = set(protocols[protocol_id]["primary_metrics"])
        if metric not in primary_metrics:
            raise ValueError(
                f"{step_id}: benchmark metric ref {metric_ref} is not a primary "
                f"metric for protocol {protocol_id}"
            )


def _stage_rank(step_id: str, stage: str) -> int:
    if stage not in STAGE_ORDER:
        raise ValueError(f"{step_id}: unknown stage {stage}")
    return STAGE_ORDER[stage]


def load_finetuning_steps(
    path: Path = DEFAULT_STEP_CONFIG,
    *,
    method_config_path: Path = DEFAULT_METHOD_CONFIG,
    protocol_config_path: Path = DEFAULT_PROTOCOL_CONFIG,
    claim_ledger_path: Path = DEFAULT_CLAIM_LEDGER,
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
    known_claim_ids = _claim_ids(claim_ledger_path)
    normalized = []
    seen_ids = set()
    previous_stage_rank = -1
    previous_step_id = None
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
        stage_rank = _stage_rank(step_id, row["stage"])
        if stage_rank < previous_stage_rank:
            raise ValueError(
                f"{step_id}: stage {row['stage']} appears after "
                f"{previous_step_id} with a later stage"
            )
        previous_stage_rank = stage_rank
        previous_step_id = step_id
        if step_id in seen_ids:
            raise ValueError(f"{step_id}: duplicate finetuning step id")
        seen_ids.add(step_id)
        if row["method"] not in methods:
            raise ValueError(f"{step_id}: unknown method {row['method']}")
        method = methods[row["method"]]
        for field in LIST_FIELDS:
            row[field] = tuple(row.get(field) or ())
        step_claim_ids = set(row["clears_claim_ids"]) | set(row["blocks_claim_ids"])
        unknown_claim_ids = sorted(step_claim_ids - known_claim_ids)
        if unknown_claim_ids:
            raise ValueError(
                f"{step_id}: unknown claim ids: {', '.join(unknown_claim_ids)}"
            )
        method_claim_ids = set(method["supported_claim_ids"]) | set(method["blocking_claim_ids"])
        unowned_claim_ids = sorted(step_claim_ids - method_claim_ids)
        if unowned_claim_ids:
            raise ValueError(
                f"{step_id}: claim ids are not declared by method {row['method']}: "
                f"{', '.join(unowned_claim_ids)}"
            )
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
        if not row["preflight_commands"]:
            raise ValueError(f"{step_id}: missing preflight_commands")
        all_commands = row["preflight_commands"] + row["commands"]
        non_uv_commands = [
            command
            for command in all_commands
            if not command.startswith("uv run --active --no-sync")
        ]
        if non_uv_commands:
            raise ValueError(f"{step_id}: commands must use uv run --active --no-sync")
        if any("reference SQL" in command for command in all_commands):
            raise ValueError(f"{step_id}: command text mentions reference SQL")
        row["measurement"] = _normalize_measurement(step_id, row.get("measurement"))
        _validate_benchmark_metric_refs(
            step_id=step_id,
            benchmark_protocol_ids=row["benchmark_protocol_ids"],
            measurement=row["measurement"],
            protocols=protocols,
        )
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
    claim_ledger_path: Path = DEFAULT_CLAIM_LEDGER,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Return a JSON-serializable summary for docs and review."""

    steps = load_finetuning_steps(
        path,
        method_config_path=method_config_path,
        protocol_config_path=protocol_config_path,
        claim_ledger_path=claim_ledger_path,
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
                "measurement": {
                    "primary_metric": step["measurement"]["primary_metric"],
                    "benchmark_metric_refs": list(
                        step["measurement"]["benchmark_metric_refs"]
                    ),
                    "supporting_metrics": list(step["measurement"]["supporting_metrics"]),
                    "comparison_artifact": step["measurement"]["comparison_artifact"],
                    "promoted_when": step["measurement"]["promoted_when"],
                },
                "clears_claim_ids": list(step["clears_claim_ids"]),
                "blocks_claim_ids": list(step["blocks_claim_ids"]),
                "preflight_command_count": len(step["preflight_commands"]),
                "cheap_preflight_available": all(
                    "<" not in command and ">" not in command
                    for command in step["preflight_commands"]
                ),
                "command_count": len(step["commands"]),
            }
            for step in steps
        ],
    }


def select_step_commands(
    *,
    step_id: str,
    command_group: str,
    path: Path = DEFAULT_STEP_CONFIG,
    method_config_path: Path = DEFAULT_METHOD_CONFIG,
    protocol_config_path: Path = DEFAULT_PROTOCOL_CONFIG,
    claim_ledger_path: Path = DEFAULT_CLAIM_LEDGER,
    repo_root: Path = Path("."),
) -> tuple[str, ...]:
    """Return preflight, run, or all commands for a configured step."""

    steps = load_finetuning_steps(
        path,
        method_config_path=method_config_path,
        protocol_config_path=protocol_config_path,
        claim_ledger_path=claim_ledger_path,
        repo_root=repo_root,
    )
    step_by_id = {step["step_id"]: step for step in steps}
    step = step_by_id.get(step_id)
    if step is None:
        raise ValueError(f"unknown finetuning step id: {step_id}")
    if command_group == "preflight":
        return tuple(step["preflight_commands"])
    if command_group == "run":
        return tuple(step["commands"])
    if command_group == "all":
        return tuple(step["preflight_commands"] + step["commands"])
    raise ValueError(f"unknown command group: {command_group}")


def build_step_command_record(
    *,
    step_id: str,
    command_group: str,
    path: Path = DEFAULT_STEP_CONFIG,
    method_config_path: Path = DEFAULT_METHOD_CONFIG,
    protocol_config_path: Path = DEFAULT_PROTOCOL_CONFIG,
    claim_ledger_path: Path = DEFAULT_CLAIM_LEDGER,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Return a machine-readable checklist for executing one step command group."""

    steps = load_finetuning_steps(
        path,
        method_config_path=method_config_path,
        protocol_config_path=protocol_config_path,
        claim_ledger_path=claim_ledger_path,
        repo_root=repo_root,
    )
    step_by_id = {step["step_id"]: step for step in steps}
    step = step_by_id.get(step_id)
    if step is None:
        raise ValueError(f"unknown finetuning step id: {step_id}")
    commands = select_step_commands(
        step_id=step_id,
        command_group=command_group,
        path=path,
        method_config_path=method_config_path,
        protocol_config_path=protocol_config_path,
        claim_ledger_path=claim_ledger_path,
        repo_root=repo_root,
    )
    return {
        "schema_version": 1,
        "step_id": step["step_id"],
        "method": step["method"],
        "stage": step["stage"],
        "command_group": command_group,
        "commands": list(commands),
        "benchmark_protocol_ids": list(step["benchmark_protocol_ids"]),
        "train_rows": step["train_rows_status"],
        "eval_rows": step["eval_rows_status"],
        "control_rows": step["control_rows_status"],
        "clears_claim_ids": list(step["clears_claim_ids"]),
        "blocks_claim_ids": list(step["blocks_claim_ids"]),
        "evidence_gate": step["evidence_gate"],
        "measurement": {
            "primary_metric": step["measurement"]["primary_metric"],
            "benchmark_metric_refs": list(step["measurement"]["benchmark_metric_refs"]),
            "supporting_metrics": list(step["measurement"]["supporting_metrics"]),
            "comparison_artifact": step["measurement"]["comparison_artifact"],
            "promoted_when": step["measurement"]["promoted_when"],
        },
        "leakage_boundary": step["leakage_boundary"],
        "cheap_preflight_available": all(
            "<" not in command and ">" not in command
            for command in step["preflight_commands"]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps-config", type=Path, default=DEFAULT_STEP_CONFIG)
    parser.add_argument("--method-config", type=Path, default=DEFAULT_METHOD_CONFIG)
    parser.add_argument("--protocol-config", type=Path, default=DEFAULT_PROTOCOL_CONFIG)
    parser.add_argument("--claim-ledger", type=Path, default=DEFAULT_CLAIM_LEDGER)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--step-id", default=None, help="Print commands for one configured step.")
    parser.add_argument(
        "--commands",
        choices=("preflight", "run", "all"),
        default=None,
        help="Command group to print when --step-id is supplied.",
    )
    args = parser.parse_args()

    if args.step_id or args.commands:
        if not args.step_id or not args.commands:
            raise SystemExit("--step-id and --commands must be supplied together")
        record = build_step_command_record(
            step_id=args.step_id,
            command_group=args.commands,
            path=args.steps_config,
            method_config_path=args.method_config,
            protocol_config_path=args.protocol_config,
            claim_ledger_path=args.claim_ledger,
            repo_root=Path.cwd(),
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(f"Wrote {args.commands} command record for {args.step_id} to {args.output}")
        else:
            print("\n".join(record["commands"]))
            if record["commands"]:
                print()
        return 0

    summary = finetuning_step_summary(
        path=args.steps_config,
        method_config_path=args.method_config,
        protocol_config_path=args.protocol_config,
        claim_ledger_path=args.claim_ledger,
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
