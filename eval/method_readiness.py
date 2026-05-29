"""Summarize which fine-tuning method claims are ready to rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from eval.benchmark_protocols import DEFAULT_PROTOCOL_CONFIG, benchmark_protocol_map

DEFAULT_METHOD_CONFIG = Path("configs/finetuning_methods.yaml")
LIST_FIELDS = {
    "supported_claim_ids",
    "blocking_claim_ids",
    "benchmark_protocol_ids",
    "training_rows",
    "control_rows",
    "prediction_input_rows",
    "prediction_input_manifests",
    "evaluator_paths",
}
REQUIRED_TEXT_FIELDS = {
    "finetuning_objective",
    "benchmark_scope",
    "primary_metric",
    "leakage_boundary",
    "evidence_gate",
}


def load_method_configs(path: Path = DEFAULT_METHOD_CONFIG) -> tuple[dict[str, Any], ...]:
    """Load configured finetuning method arms for readiness reporting."""

    payload = yaml.safe_load(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("finetuning method config must use schema_version=1")
    methods = payload.get("methods")
    if not isinstance(methods, list) or not methods:
        raise ValueError("finetuning method config must include non-empty methods")
    normalized = []
    for method in methods:
        row = dict(method)
        missing_text_fields = [
            field
            for field in sorted(REQUIRED_TEXT_FIELDS)
            if not str(row.get(field) or "").strip()
        ]
        if missing_text_fields:
            method_name = row.get("method") or "<unknown method>"
            raise ValueError(
                f"{method_name}: missing {', '.join(missing_text_fields)}"
            )
        for field in LIST_FIELDS:
            row[field] = tuple(row.get(field) or ())
        if not row["benchmark_protocol_ids"]:
            method_name = row.get("method") or "<unknown method>"
            raise ValueError(f"{method_name}: missing benchmark_protocol_ids")
        normalized.append(row)
    return tuple(normalized)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _claim_statuses(ledger_path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["claim_id"]): row for row in _load_jsonl(ledger_path)}


def _existing_claim_ids(claims: dict[str, dict[str, Any]], claim_ids: tuple[str, ...]) -> list[str]:
    return [claim_id for claim_id in claim_ids if claim_id in claims]


def _all_supported(claims: dict[str, dict[str, Any]], claim_ids: list[str]) -> bool:
    supported_statuses = {
        "supported_method_control",
        "supported_metric_dsl_quality",
        "supported_planner_quality",
        "supported_proxy",
        "supported_value_retrieval_coverage",
    }
    return bool(claim_ids) and all(
        claims[claim_id].get("claim_status") in supported_statuses
        for claim_id in claim_ids
    )


def _blocking_reasons(claims: dict[str, dict[str, Any]], claim_ids: list[str]) -> list[str]:
    reasons = []
    for claim_id in claim_ids:
        row = claims.get(claim_id) or {}
        reason = row.get("blocking_reason") or row.get("required_artifact")
        if reason:
            reasons.append(f"{claim_id}: {reason}")
    return reasons


def _open_blocking_claim_ids(
    claims: dict[str, dict[str, Any]],
    claim_ids: tuple[str, ...],
) -> list[str]:
    open_claim_ids = []
    for claim_id in claim_ids:
        row = claims.get(claim_id)
        if row is None:
            open_claim_ids.append(claim_id)
            continue
        if (
            row.get("claim_status") == "pending"
            or row.get("artifact_valid") is False
            or row.get("blocking_reason")
        ):
            open_claim_ids.append(claim_id)
    return open_claim_ids


def _path_status(repo_root: Path, paths: tuple[str, ...]) -> dict[str, bool]:
    return {path: (repo_root / path).exists() for path in paths}


def _paths_ready(path_status: dict[str, bool], *, required: bool) -> bool:
    if not required:
        return True
    return bool(path_status) and all(path_status.values())


def _prediction_inputs_ready(
    *,
    row_status: dict[str, bool],
    manifest_status: dict[str, bool],
    required: bool,
) -> bool:
    if not required:
        return True
    return _paths_ready(row_status, required=True) or _paths_ready(
        manifest_status,
        required=True,
    )


def required_readiness_failures(rows: list[dict[str, Any]]) -> list[str]:
    """Return missing required path checks from method-readiness rows."""

    failures = []
    for row in rows:
        method = row["method"]
        if row["smoke_rows_required"] and not row["smoke_rows_ready"]:
            missing = [
                path
                for path, exists in row["training_rows"].items()
                if not exists
            ]
            failures.append(f"{method}: missing smoke rows: {', '.join(missing)}")
        if row["control_rows_required"] and not row["control_rows_ready"]:
            missing = [
                path
                for path, exists in row["control_rows"].items()
                if not exists
            ]
            failures.append(f"{method}: missing control rows: {', '.join(missing)}")
        if row.get("prediction_input_rows_required") and not row.get(
            "prediction_input_rows_ready",
            True,
        ):
            missing_rows = [
                path
                for path, exists in row.get("prediction_input_rows", {}).items()
                if not exists
            ]
            missing_manifests = [
                path
                for path, exists in row.get("prediction_input_manifests", {}).items()
                if not exists
            ]
            missing = missing_rows + missing_manifests
            failures.append(
                f"{method}: missing prediction input rows or manifests: {', '.join(missing)}"
            )
        if not row["evaluators_ready"]:
            missing = [
                path
                for path, exists in row["evaluator_paths"].items()
                if not exists
            ]
            failures.append(f"{method}: missing evaluators: {', '.join(missing)}")
    return failures


def build_method_readiness(
    *,
    ledger_path: Path,
    repo_root: Path,
    method_config_path: Path | None = None,
    protocol_config_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic method-readiness rows from the claim ledger."""

    claims = _claim_statuses(ledger_path)
    rows = []
    config_path = method_config_path or repo_root / DEFAULT_METHOD_CONFIG
    protocol_path = protocol_config_path or repo_root / DEFAULT_PROTOCOL_CONFIG
    protocols = benchmark_protocol_map(protocol_path)
    for method in load_method_configs(config_path):
        benchmark_protocol_ids = tuple(method["benchmark_protocol_ids"])
        missing_protocol_ids = [
            protocol_id
            for protocol_id in benchmark_protocol_ids
            if protocol_id not in protocols
        ]
        if missing_protocol_ids:
            raise ValueError(
                f"{method['method']}: unknown benchmark protocols: "
                f"{', '.join(missing_protocol_ids)}"
            )
        supported_claim_ids = _existing_claim_ids(
            claims,
            tuple(method["supported_claim_ids"]),
        )
        blocking_claim_ids = _existing_claim_ids(
            claims,
            tuple(method["blocking_claim_ids"]),
        )
        open_blocking_claim_ids = _open_blocking_claim_ids(
            claims,
            tuple(method["blocking_claim_ids"]),
        )
        blocking_reasons = _blocking_reasons(claims, blocking_claim_ids)
        control_ready_now = (
            method["readiness_level"] == "control_ready"
            and _all_supported(claims, supported_claim_ids)
        )
        rankable_now = (
            _all_supported(claims, supported_claim_ids)
            and not open_blocking_claim_ids
        )
        training_row_status = _path_status(
            repo_root,
            tuple(method["training_rows"]),
        )
        control_row_status = _path_status(
            repo_root,
            tuple(method["control_rows"]),
        )
        evaluator_status = _path_status(
            repo_root,
            tuple(method["evaluator_paths"]),
        )
        prediction_input_status = _path_status(
            repo_root,
            tuple(method["prediction_input_rows"]),
        )
        prediction_input_manifest_status = _path_status(
            repo_root,
            tuple(method["prediction_input_manifests"]),
        )
        prediction_input_required = bool(method.get("prediction_input_rows_required"))
        rows.append(
            {
                "method": method["method"],
                "readiness_level": method["readiness_level"],
                "control_ready_now": control_ready_now,
                "rankable_now": rankable_now,
                "smoke_rows_required": method["smoke_rows_required"],
                "smoke_rows_ready": _paths_ready(
                    training_row_status,
                    required=bool(method["smoke_rows_required"]),
                ),
                "control_rows_required": method["control_rows_required"],
                "control_rows_ready": _paths_ready(
                    control_row_status,
                    required=bool(method["control_rows_required"]),
                ),
                "prediction_input_rows_required": prediction_input_required,
                "prediction_input_rows_ready": _prediction_inputs_ready(
                    row_status=prediction_input_status,
                    manifest_status=prediction_input_manifest_status,
                    required=prediction_input_required,
                ),
                "evaluators_ready": _paths_ready(evaluator_status, required=True),
                "training_rows": training_row_status,
                "control_rows": control_row_status,
                "prediction_input_rows": prediction_input_status,
                "prediction_input_manifests": prediction_input_manifest_status,
                "evaluator_paths": evaluator_status,
                "supported_claim_ids": supported_claim_ids,
                "blocking_claim_ids": blocking_claim_ids,
                "open_blocking_claim_ids": open_blocking_claim_ids,
                "blocking_reasons": blocking_reasons,
                "benchmark_protocol_ids": list(benchmark_protocol_ids),
                "benchmark_protocols": [
                    {
                        "protocol_id": protocols[protocol_id]["protocol_id"],
                        "benchmark": protocols[protocol_id]["benchmark"],
                        "role": protocols[protocol_id]["role"],
                        "supports_method_ranking": protocols[protocol_id][
                            "supports_method_ranking"
                        ],
                        "supports_hosted_sota_claim": protocols[protocol_id][
                            "supports_hosted_sota_claim"
                        ],
                    }
                    for protocol_id in benchmark_protocol_ids
                ],
                "module_path": method["module_path"],
                "module_exists": (repo_root / method["module_path"]).exists(),
                "next_command": method["next_command"],
                "next_artifact": method["next_artifact"],
                "claim_boundary": method["claim_boundary"],
                "rankable_when": method["rankable_when"],
                "finetuning_objective": method["finetuning_objective"],
                "benchmark_scope": method["benchmark_scope"],
                "primary_metric": method["primary_metric"],
                "leakage_boundary": method["leakage_boundary"],
                "evidence_gate": method["evidence_gate"],
            }
        )
    return rows


def write_method_readiness_report(
    *,
    output_path: Path,
    ledger_path: Path,
    repo_root: Path,
    method_config_path: Path | None = None,
    protocol_config_path: Path | None = None,
) -> list[dict[str, Any]]:
    rows = build_method_readiness(
        ledger_path=ledger_path,
        repo_root=repo_root,
        method_config_path=method_config_path,
        protocol_config_path=protocol_config_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"schema_version": 1, "methods": rows}, indent=2, sort_keys=True)
        + "\n"
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("docs/claim_ledgers/cosql_dev_100.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/method_readiness_report.json"),
    )
    parser.add_argument(
        "--method-config",
        type=Path,
        default=None,
        help="Configured finetuning method arms to report.",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=None,
        help="Configured benchmark protocols to validate against method arms.",
    )
    parser.add_argument(
        "--fail-on-missing-required",
        action="store_true",
        help="Exit non-zero if a required smoke row, control row, or evaluator path is missing.",
    )
    args = parser.parse_args()
    repo_root = Path.cwd()
    rows = write_method_readiness_report(
        output_path=args.output,
        ledger_path=args.ledger,
        repo_root=repo_root,
        method_config_path=args.method_config,
        protocol_config_path=args.protocol_config,
    )
    print(f"Wrote {len(rows)} method readiness rows to {args.output}")
    if args.fail_on_missing_required:
        failures = required_readiness_failures(rows)
        for failure in failures:
            print(f"Missing required readiness input: {failure}")
        if failures:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
