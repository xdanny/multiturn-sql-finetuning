"""Summarize which fine-tuning method claims are ready to rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METHODS: tuple[dict[str, Any], ...] = (
    {
        "method": "Direct SQL SFT",
        "readiness_level": "control_ready",
        "supported_claim_ids": (
            "qwen35_9b_base_cosql_dev_100turns",
            "multiturn_sql_100_cosql_dev_100turns",
        ),
        "blocking_claim_ids": (
            "hosted_sota_same_protocol",
            "local_beats_hosted_same_protocol",
            "bird_interact_local_vs_hosted",
        ),
        "module_path": "eval/run_eval.py",
        "next_command": (
            "python -m eval.run_eval --input data/processed/eval_cosql_dev_100.jsonl "
            "--output results/direct_sql/<run-id>.jsonl"
        ),
        "next_artifact": "same-protocol direct SQL manifest on fixed proxy and BIRD-Interact rows",
        "claim_boundary": "Control arm only; proxy movement is not a hosted/SOTA win.",
        "rankable_when": (
            "Use as the direct-SQL control now; rank against hosted/SOTA only after "
            "same-protocol hosted and BIRD-Interact blockers clear."
        ),
    },
    {
        "method": "Planner/DSL first, SQL second",
        "readiness_level": "needs_endpoint_comparison",
        "supported_claim_ids": ("planner_lexical_schema_baseline",),
        "blocking_claim_ids": ("predicted_planner_sql_execution",),
        "module_path": "eval/run_predicted_planner_comparison.py",
        "next_command": (
            "python -m eval.planner_optimize ...; python -m eval.planner_predict ...; "
            "python -m eval.run_predicted_planner_comparison --direct-input ... "
            "--predicted-input ..."
        ),
        "next_artifact": "predicted-planner SQL manifest compared against direct SQL",
        "claim_boundary": "Planner F1 is supported; SQL improvement is still pending.",
        "rankable_when": "Rank only after predicted-planner SQL beats direct SQL on identical rows.",
    },
    {
        "method": "Semantic-layer tuning",
        "readiness_level": "needs_value_entity_retrieval",
        "supported_claim_ids": ("semantic_prompt_minimal_executable_cosql_dev_100turns",),
        "blocking_claim_ids": (
            "hosted_sota_same_protocol",
            "local_beats_hosted_same_protocol",
        ),
        "module_path": "data/value_artifacts.py",
        "next_command": (
            "python -m data.value_artifacts ...; python -m data.value_index ...; "
            "python -m eval.classify_errors ..."
        ),
        "next_artifact": "value/entity retrieval manifest and row-matched semantic delta",
        "claim_boundary": "Semantic prompt movement is proxy evidence, not a proven method win.",
        "rankable_when": "Rank after value/entity artifacts improve same-row SQL outcomes.",
    },
    {
        "method": "MEASURE()-preserving metric DSL",
        "readiness_level": "needs_prediction_manifest",
        "supported_claim_ids": (),
        "blocking_claim_ids": (
            "metric_dsl_evaluation_manifest",
            "metric_dsl_beats_direct_sql",
        ),
        "module_path": "eval/metric_dsl_eval.py",
        "next_command": (
            "python -m eval.metric_dsl_eval --input results/metric_dsl/<run-id>.predictions.jsonl "
            "--manifest-output results/metric_dsl/<run-id>.manifest.json; "
            "python -m eval.compare_metric_dsl_direct_sql ..."
        ),
        "next_artifact": "metric-DSL prediction manifest plus direct-SQL comparison",
        "claim_boundary": "Parser/compiler exist; no metric-DSL method win yet.",
        "rankable_when": "Rank after compiled metric DSL beats direct SQL on same metric-heavy rows.",
    },
    {
        "method": "Behavior/recovery tuning",
        "readiness_level": "needs_rollout_manifest",
        "supported_claim_ids": (),
        "blocking_claim_ids": (
            "model_generated_history_rollout",
            "rollout_beats_teacher_forced_history",
        ),
        "module_path": "eval/rollout_eval.py",
        "next_command": (
            "python -m eval.rollout_eval --input data/processed/eval_cosql_dev_100.jsonl "
            "--manifest-output results/rollout/<run-id>.manifest.json; "
            "python -m eval.compare_rollout_history ..."
        ),
        "next_artifact": "generated-history rollout manifest compared with teacher-forced history",
        "claim_boundary": "Teacher-forced proxy scores do not prove recovery behavior.",
        "rankable_when": "Rank after generated-history rollout improves over teacher-forced control.",
    },
    {
        "method": "Hosted and BIRD-Interact comparison",
        "readiness_level": "needs_hosted_protocol_run",
        "supported_claim_ids": (),
        "blocking_claim_ids": (
            "hosted_sota_same_protocol",
            "local_beats_hosted_same_protocol",
            "bird_interact_local_vs_hosted",
        ),
        "module_path": "eval/compare_hosted_baseline.py",
        "next_command": (
            "python -m eval.compare_hosted_baseline --hosted-manifest ... "
            "--local-manifest ... --output ..."
        ),
        "next_artifact": "hosted/local comparison manifest plus BIRD-Interact transfer manifest",
        "claim_boundary": "No hosted/SOTA or BIRD-Interact win is currently supported.",
        "rankable_when": "Rank after local rows beat hosted rows under the same protocol.",
    },
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _claim_statuses(ledger_path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["claim_id"]): row for row in _load_jsonl(ledger_path)}


def _existing_claim_ids(claims: dict[str, dict[str, Any]], claim_ids: tuple[str, ...]) -> list[str]:
    return [claim_id for claim_id in claim_ids if claim_id in claims]


def _all_supported(claims: dict[str, dict[str, Any]], claim_ids: list[str]) -> bool:
    supported_statuses = {"supported_proxy", "supported_planner_quality"}
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


def build_method_readiness(*, ledger_path: Path, repo_root: Path) -> list[dict[str, Any]]:
    """Return deterministic method-readiness rows from the claim ledger."""

    claims = _claim_statuses(ledger_path)
    rows = []
    for method in METHODS:
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
        rows.append(
            {
                "method": method["method"],
                "readiness_level": method["readiness_level"],
                "control_ready_now": control_ready_now,
                "rankable_now": rankable_now,
                "supported_claim_ids": supported_claim_ids,
                "blocking_claim_ids": blocking_claim_ids,
                "open_blocking_claim_ids": open_blocking_claim_ids,
                "blocking_reasons": blocking_reasons,
                "module_path": method["module_path"],
                "module_exists": (repo_root / method["module_path"]).exists(),
                "next_command": method["next_command"],
                "next_artifact": method["next_artifact"],
                "claim_boundary": method["claim_boundary"],
                "rankable_when": method["rankable_when"],
            }
        )
    return rows


def write_method_readiness_report(
    *,
    output_path: Path,
    ledger_path: Path,
    repo_root: Path,
) -> list[dict[str, Any]]:
    rows = build_method_readiness(ledger_path=ledger_path, repo_root=repo_root)
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
    args = parser.parse_args()
    repo_root = Path.cwd()
    rows = write_method_readiness_report(
        output_path=args.output,
        ledger_path=args.ledger,
        repo_root=repo_root,
    )
    print(f"Wrote {len(rows)} method readiness rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
