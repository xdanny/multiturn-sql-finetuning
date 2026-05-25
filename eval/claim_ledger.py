"""
Build a manifest-backed claim ledger for SQL evaluation results.

The ledger is deliberately conservative: it records what a result can support
and marks larger claims as pending until matching artifacts exist.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

PRODUCTION_MODES = {"non_oracle_generation", "predicted_planner"}
HOSTED_ENDPOINT_PREFIXES = ("https://", "anthropic:", "google:")
HOSTED_LATENCY_KEYS = (
    "mean_latency_ms",
    "mean_generation_latency_ms",
    "p50_latency_ms",
    "latency_ms",
)
HOSTED_COST_KEYS = ("total_cost_usd", "estimated_cost_usd", "cost_usd")
COMPARISON_ACCURACY_KEYS = (
    "value_execution_accuracy",
    "strict_execution_accuracy",
    "execution_accuracy",
)
MODEL_GENERATED_SQL_ROLLOUT = "model_generated_sql_rollout"
GOLD_SQL_TEACHER_FORCED = "gold_sql_teacher_forced"
METRIC_DSL = "metric_dsl"
METRIC_DSL_DIRECT_SQL = "metric_dsl_direct_sql"
NON_ORACLE_GENERATION = "non_oracle_generation"
MANDATORY_MANIFEST_FIELDS = (
    "schema_version",
    "run_id",
    "benchmark",
    "model_name",
    "endpoint",
    "evaluation_mode",
    "oracle_allowed",
    "input_path",
    "input_sha256",
    "output_path",
    "output_sha256",
    "row_count",
    "metrics",
    "command",
)
ORACLE_MARKERS = (
    "Oracle SQL planning hints",
    "SQL planning hints:",
)
PENDING_CLAIMS = (
    {
        "claim_id": "predicted_planner_sql_execution",
        "claim_status": "pending",
        "artifact_type": "pending_claim",
        "evaluation_mode": "predicted_planner",
        "allowed_public_claim": "pending predicted-planner SQL execution",
        "blocking_reason": "no predicted_planner result manifest",
        "required_artifact": "same-protocol endpoint SQL result manifest",
    },
    {
        "claim_id": "model_generated_history_rollout",
        "claim_status": "pending",
        "artifact_type": "pending_claim",
        "evaluation_mode": "non_oracle_generation",
        "allowed_public_claim": "no generated-history rollout result yet",
        "blocking_reason": "no model-generated-history rollout manifest",
        "required_artifact": "prepared_rollout result manifest with model_generated_sql_rollout history",
    },
    {
        "claim_id": "rollout_beats_teacher_forced_history",
        "claim_status": "pending",
        "artifact_type": "pending_claim",
        "evaluation_mode": "not_run",
        "allowed_public_claim": "no behavior/recovery improvement claim yet",
        "blocking_reason": "no side-by-side rollout-vs-teacher-forced comparison",
        "required_artifact": "same-model rollout and teacher-forced manifests with comparison metrics",
    },
    {
        "claim_id": "hosted_sota_same_protocol",
        "claim_status": "pending",
        "artifact_type": "pending_claim",
        "evaluation_mode": "not_run",
        "allowed_public_claim": "no hosted/SOTA comparison yet",
        "blocking_reason": "no same-protocol hosted-model manifest",
        "required_artifact": "hosted-model result manifest with cost and latency",
    },
    {
        "claim_id": "bird_interact_local_vs_hosted",
        "claim_status": "pending",
        "artifact_type": "pending_claim",
        "evaluation_mode": "not_run",
        "allowed_public_claim": "no BIRD-Interact claim yet",
        "blocking_reason": "no BIRD-Interact result manifest",
        "required_artifact": "BIRD-Interact local and hosted result manifests",
    },
    {
        "claim_id": "metric_dsl_evaluation_manifest",
        "claim_status": "pending",
        "artifact_type": "pending_claim",
        "evaluation_mode": METRIC_DSL,
        "allowed_public_claim": "no metric-DSL evaluation result yet",
        "blocking_reason": "no valid metric_dsl result manifest",
        "required_artifact": "metric_dsl result manifest with parse, compile, execution, and measure-preservation metrics",
    },
    {
        "claim_id": "metric_dsl_beats_direct_sql",
        "claim_status": "pending",
        "artifact_type": "pending_claim",
        "evaluation_mode": METRIC_DSL,
        "allowed_public_claim": "no metric-DSL vs direct-SQL improvement claim yet",
        "blocking_reason": "no side-by-side metric-DSL-vs-direct-SQL comparison",
        "required_artifact": "compared metric_dsl manifest with direct-SQL baseline",
    },
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _repo_path(repo_root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _hash_matches(path: Path | None, expected: str | None) -> bool | None:
    if path is None or expected is None:
        return None
    actual = sha256_file(path)
    if actual is None:
        return False
    return actual == expected


def _input_contract(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "input_exists": False,
            "input_rows": 0,
            "input_contract_status": "missing",
            "history_policy": None,
            "teacher_forced_history": None,
            "turn_format": None,
        }

    rows = _load_jsonl(path)
    modes = Counter(str(row.get("evaluation_mode") or "") for row in rows)
    histories = Counter(str(row.get("history_policy") or "") for row in rows)
    turn_formats = Counter(str(row.get("turn_format") or "") for row in rows)
    has_contract = all(row.get("evaluation_mode") and row.get("gold_plans") is not None for row in rows)
    history_policy = _most_common_nonempty(histories)
    oracle_rows = sum(1 for row in rows if _row_uses_oracle_plan(row))
    return {
        "input_exists": True,
        "input_rows": len(rows),
        "input_contract_status": "contracted" if has_contract else "legacy_missing_contract_fields",
        "input_evaluation_modes": _counter_without_empty(modes),
        "input_oracle_rows": oracle_rows,
        "history_policy": history_policy,
        "teacher_forced_history": (
            "teacher_forced" in history_policy if history_policy is not None else _has_prior_assistant(rows)
        ),
        "turn_format": _most_common_nonempty(turn_formats),
    }


def _row_uses_oracle_plan(row: dict[str, Any]) -> bool:
    if (
        row.get("uses_oracle_planning_hints")
        or row.get("semantic_context_pruned_by_oracle_labels")
        or row.get("semantic_model_oracle_derived")
    ):
        return True
    return any(
        marker in str(message.get("content", ""))
        for message in row.get("messages", [])
        for marker in ORACLE_MARKERS
    )


def _output_contract(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "output_exists": False,
            "output_rows": 0,
            "output_evaluation_modes": {},
            "output_oracle_rows": 0,
            "output_value_scored_rows": 0,
            "output_strict_scored_rows": 0,
        }
    rows = _load_jsonl(path)
    modes = Counter(str(row.get("evaluation_mode") or "") for row in rows)
    return {
        "output_exists": True,
        "output_rows": len(rows),
        "output_evaluation_modes": _counter_without_empty(modes),
        "output_oracle_rows": sum(1 for row in rows if _row_uses_oracle_plan(row)),
        "output_value_scored_rows": sum(
            1 for row in rows if row.get("value_execution_score") is not None
        ),
        "output_strict_scored_rows": sum(
            1 for row in rows if row.get("strict_execution_score") is not None
        ),
    }


def _has_prior_assistant(rows: list[dict[str, Any]]) -> bool:
    for row in rows:
        assistant_seen = False
        for message in row.get("messages", []):
            if message.get("role") == "assistant":
                assistant_seen = True
            elif message.get("role") == "user" and assistant_seen:
                return True
    return False


def _counter_without_empty(counter: Counter[str]) -> dict[str, int]:
    return {key: value for key, value in sorted(counter.items()) if key}


def _most_common_nonempty(counter: Counter[str]) -> str | None:
    for key, _ in counter.most_common():
        if key:
            return key
    return None


def _classified_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "classified_path": str(path) if path else None,
            "classified_exists": False,
            "classified_rows": 0,
            "primary_error_counts": {},
        }
    rows = _load_jsonl(path)
    return {
        "classified_path": str(path),
        "classified_exists": True,
        "classified_rows": len(rows),
        "primary_error_counts": dict(Counter(row.get("error_primary", "unknown") for row in rows)),
    }


def _classified_path_for_output(repo_root: Path, classified_dir: Path | None, output_path: Path | None) -> Path | None:
    if classified_dir is None or output_path is None:
        return None
    return classified_dir / output_path.name


def _claim_status(manifest: dict[str, Any]) -> str:
    mode = manifest.get("evaluation_mode")
    if mode == "oracle_planner_diagnostic" or manifest.get("oracle_allowed"):
        return "diagnostic_upper_bound"
    if mode == METRIC_DSL and manifest.get("benchmark") == METRIC_DSL:
        return "supported_metric_dsl_quality"
    if mode in PRODUCTION_MODES:
        return "supported_proxy"
    return "pending"


def _allowed_public_claim(status: str) -> str:
    return {
        "diagnostic_upper_bound": "oracle planner diagnostic only",
        "supported_metric_dsl_quality": "metric-DSL quality only, not direct SQL superiority",
        "supported_proxy": "local proxy result only",
        "pending": "pending until required artifacts are present",
    }[status]


def _missing_manifest_fields(manifest: dict[str, Any]) -> list[str]:
    missing = []
    for field in MANDATORY_MANIFEST_FIELDS:
        if field not in manifest or manifest[field] in (None, ""):
            missing.append(field)
    metrics = manifest.get("metrics")
    if (not isinstance(metrics, dict) or not metrics) and "metrics" not in missing:
        missing.append("metrics")
    return missing


def _predicted_output_has_mode(output_contract: dict[str, Any]) -> bool:
    modes = output_contract.get("output_evaluation_modes") or {}
    return bool(modes) and set(modes) == {"predicted_planner"}


def _metric_dsl_blocking_reason(
    manifest: dict[str, Any],
    output_contract: dict[str, Any],
) -> str | None:
    if manifest.get("evaluation_mode") != METRIC_DSL and manifest.get("benchmark") != METRIC_DSL:
        return None
    if manifest.get("evaluation_mode") != METRIC_DSL or manifest.get("benchmark") != METRIC_DSL:
        return "metric_dsl manifest must use benchmark=metric_dsl and evaluation_mode=metric_dsl"
    modes = output_contract.get("output_evaluation_modes") or {}
    if modes and set(modes) != {METRIC_DSL}:
        return "metric_dsl output rows missing metric_dsl mode"
    metrics = manifest.get("metrics") or {}
    row_count = _num(manifest.get("row_count"))
    executed_rows = _num(metrics.get("compiled_sql_execution_evaluated_rows"))
    positive_metric_keys = (
        "metric_dsl_parse_rate",
        "metric_dsl_compile_rate",
        "measure_preservation",
    )
    if any((_num(metrics.get(key)) or 0.0) <= 0.0 for key in positive_metric_keys):
        return "metric_dsl manifest quality metrics are incomplete"
    if row_count is None or executed_rows is None or executed_rows != row_count:
        return "metric_dsl manifest quality metrics are incomplete"
    if (
        metrics.get("value_execution_accuracy") is None
        or metrics.get("strict_execution_accuracy") is None
    ):
        return "metric_dsl manifest quality metrics are incomplete"
    return None


def _blocking_reason(
    *,
    manifest: dict[str, Any],
    input_contract: dict[str, Any],
    output_contract: dict[str, Any],
    input_hash_matches: bool | None,
    output_hash_matches: bool | None,
) -> str | None:
    missing = _missing_manifest_fields(manifest)
    if missing:
        return "missing manifest fields: " + ", ".join(missing)
    if input_hash_matches is False or output_hash_matches is False:
        return "manifest hash mismatch"
    if not input_contract.get("input_exists") or not output_contract.get("output_exists"):
        return "manifest artifact missing"
    mode = manifest.get("evaluation_mode")
    if not manifest.get("oracle_allowed") and (
        input_contract.get("input_oracle_rows", 0) > 0
        or output_contract.get("output_oracle_rows", 0) > 0
    ):
        return "oracle planning hints found in non-oracle artifact"
    metric_dsl_blocking_reason = _metric_dsl_blocking_reason(manifest, output_contract)
    if metric_dsl_blocking_reason:
        return metric_dsl_blocking_reason
    if mode == "predicted_planner" and not _predicted_output_has_mode(output_contract):
        return "predicted_planner output rows missing predicted_planner mode"
    return None


def _manifest_row(
    manifest: dict[str, Any],
    *,
    repo_root: Path,
    classified_dir: Path | None,
) -> dict[str, Any]:
    input_path = _repo_path(repo_root, manifest.get("input_path"))
    output_path = _repo_path(repo_root, manifest.get("output_path"))
    metrics = manifest.get("metrics") or {}
    classified_path = _classified_path_for_output(repo_root, classified_dir, output_path)
    endpoint = manifest.get("endpoint")
    input_hash_matches = _hash_matches(input_path, manifest.get("input_sha256"))
    output_hash_matches = _hash_matches(output_path, manifest.get("output_sha256"))
    input_contract = _input_contract(input_path)
    output_contract = _output_contract(output_path)
    blocking_reason = _blocking_reason(
        manifest=manifest,
        input_contract=input_contract,
        output_contract=output_contract,
        input_hash_matches=input_hash_matches,
        output_hash_matches=output_hash_matches,
    )
    base_status = _claim_status(manifest)
    status = "pending" if blocking_reason else base_status
    row = {
        "claim_id": manifest.get("run_id"),
        "artifact_type": "result_manifest",
        "claim_status": status,
        "allowed_public_claim": _allowed_public_claim(status),
        "production_claim_allowed": status == "supported_proxy",
        "can_support_sota_claim": False,
        "artifact_valid": blocking_reason is None,
        "blocking_reason": blocking_reason,
        "required_artifact": None if blocking_reason is None else "valid result manifest and matching artifacts",
        "benchmark": manifest.get("benchmark"),
        "model_name": manifest.get("model_name"),
        "endpoint": endpoint,
        "prompt_variant": manifest.get("prompt_variant"),
        "evaluation_mode": manifest.get("evaluation_mode"),
        "oracle_allowed": bool(manifest.get("oracle_allowed")),
        "command": manifest.get("command") or [],
        "row_count": manifest.get("row_count"),
        "dialog_count": metrics.get("dialog_count"),
        "metric_keys": sorted(metrics),
        "result_history_policy": metrics.get("history_policy"),
        "teacher_forced_comparison_run_id": metrics.get("teacher_forced_comparison_run_id"),
        "teacher_forced_input_sha256": metrics.get("teacher_forced_input_sha256"),
        "teacher_forced_model_name": metrics.get("teacher_forced_model_name"),
        "teacher_forced_value_execution_accuracy": metrics.get(
            "teacher_forced_value_execution_accuracy"
        ),
        "teacher_forced_comparable_row_count": metrics.get(
            "teacher_forced_comparable_row_count"
        ),
        "rollout_value_delta_vs_teacher_forced": metrics.get(
            "rollout_value_delta_vs_teacher_forced"
        ),
        "direct_sql_comparison_run_id": metrics.get("direct_sql_comparison_run_id"),
        "direct_sql_model_name": metrics.get("direct_sql_model_name"),
        "direct_sql_input_sha256": metrics.get("direct_sql_input_sha256"),
        "direct_sql_output_sha256": metrics.get("direct_sql_output_sha256"),
        "direct_sql_value_execution_accuracy": metrics.get(
            "direct_sql_value_execution_accuracy"
        ),
        "direct_sql_strict_execution_accuracy": metrics.get(
            "direct_sql_strict_execution_accuracy"
        ),
        "predicted_planner_value_delta_vs_direct_sql": metrics.get(
            "predicted_planner_value_delta_vs_direct_sql"
        ),
        "direct_sql_comparable_row_count": metrics.get("direct_sql_comparable_row_count"),
        "metric_dsl_parse_rate": metrics.get("metric_dsl_parse_rate"),
        "metric_dsl_compile_rate": metrics.get("metric_dsl_compile_rate"),
        "compiled_sql_execution_evaluated_rows": metrics.get(
            "compiled_sql_execution_evaluated_rows"
        ),
        "measure_preservation": metrics.get("measure_preservation"),
        "metric_dsl_measure_preservation": metrics.get("metric_dsl_measure_preservation"),
        "metric_dsl_value_delta_vs_direct_sql": metrics.get(
            "metric_dsl_value_delta_vs_direct_sql"
        ),
        "metric_dsl_strict_delta_vs_direct_sql": metrics.get(
            "metric_dsl_strict_delta_vs_direct_sql"
        ),
        "metric_dsl_comparable_row_count": metrics.get("metric_dsl_comparable_row_count"),
        "strict_execution_accuracy": metrics.get("strict_execution_accuracy"),
        "value_execution_accuracy": metrics.get("value_execution_accuracy")
        if metrics.get("value_execution_accuracy") is not None
        else metrics.get("execution_accuracy"),
        "syntax_accuracy": metrics.get("syntax_accuracy"),
        "input_path": manifest.get("input_path"),
        "input_sha256": manifest.get("input_sha256"),
        "input_sha256_matches": input_hash_matches,
        "output_path": manifest.get("output_path"),
        "output_sha256": manifest.get("output_sha256"),
        "output_sha256_matches": output_hash_matches,
    }
    row.update(input_contract)
    if row.get("result_history_policy"):
        row["history_policy"] = row["result_history_policy"]
        row["teacher_forced_history"] = "teacher_forced" in str(row["result_history_policy"])
    row.update(output_contract)
    row.update(_classified_summary(classified_path))
    return row


def _planner_row(planner_summary_path: Path | None) -> dict[str, Any] | None:
    if planner_summary_path is None or not planner_summary_path.exists():
        return None
    summary = _load_json(planner_summary_path)
    oracle_prompt_rows = int(summary.get("oracle_prompt_rows") or 0)
    artifact_valid = oracle_prompt_rows == 0
    return {
        "claim_id": "planner_lexical_schema_baseline",
        "artifact_type": "planner_summary",
        "claim_status": "supported_planner_quality" if artifact_valid else "pending",
        "allowed_public_claim": "planner quality only, not SQL execution",
        "production_claim_allowed": False,
        "can_support_sota_claim": False,
        "artifact_valid": artifact_valid,
        "blocking_reason": None
        if artifact_valid
        else "planner summary contains oracle prompt rows",
        "required_artifact": None if artifact_valid else "non-oracle planner summary",
        "evaluation_mode": "planner_scoring",
        "planner_summary_path": str(planner_summary_path),
        "row_count": summary.get("rows"),
        "dialog_count": summary.get("dialog_count"),
        "oracle_prompt_rows": oracle_prompt_rows,
        "prediction_sources": summary.get("prediction_sources") or {},
        "macro_planner_score": summary.get("macro_planner_score"),
        "table_f1": summary.get("table_f1"),
        "column_f1": summary.get("column_f1"),
        "selected_count_match": summary.get("selected_count_match"),
    }


def _pending_rows(existing_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = list(existing_rows)
    has_predicted_sql = any(_row_has_predicted_sql_claim_support(row, rows) for row in rows)
    has_valid_predicted_manifest = any(_row_is_valid_predicted_sql_manifest(row) for row in rows)
    has_rollout = any(_row_has_rollout_claim_support(row) for row in rows)
    has_rollout_teacher_forced_comparison = any(
        _row_has_rollout_teacher_forced_comparison(row, rows) for row in rows
    )
    has_metric_dsl_eval = any(_row_has_metric_dsl_quality_support(row) for row in rows)
    has_metric_dsl_direct_sql_comparison = any(
        _row_has_metric_dsl_direct_sql_comparison(row, rows) for row in rows
    )
    has_hosted = any(_row_has_hosted_claim_support(row, rows) for row in rows)
    has_bird_interact = any(
        row.get("artifact_type") == "result_manifest"
        and "bird_interact" in str(row.get("benchmark") or "").lower()
        for row in rows
    )
    keep = {
        "predicted_planner_sql_execution": not has_predicted_sql,
        "model_generated_history_rollout": not has_rollout,
        "rollout_beats_teacher_forced_history": not has_rollout_teacher_forced_comparison,
        "hosted_sota_same_protocol": not has_hosted,
        "bird_interact_local_vs_hosted": not has_bird_interact,
        "metric_dsl_evaluation_manifest": not has_metric_dsl_eval,
        "metric_dsl_beats_direct_sql": not has_metric_dsl_direct_sql_comparison,
    }
    pending = []
    for row in PENDING_CLAIMS:
        if keep[row["claim_id"]]:
            row = dict(row)
            if row["claim_id"] == "predicted_planner_sql_execution" and has_valid_predicted_manifest:
                row["blocking_reason"] = (
                    "no same-model direct-SQL comparison with positive value delta"
                )
                row["required_artifact"] = (
                    "compared predicted_planner manifest with direct-SQL baseline"
                )
            if row["claim_id"] == "metric_dsl_beats_direct_sql" and has_metric_dsl_eval:
                row["blocking_reason"] = (
                    "no non-oracle direct-SQL comparison with positive metric-DSL value delta"
                )
                row["required_artifact"] = (
                    "compared metric_dsl manifest plus matching direct-SQL manifest"
                )
            pending.append(
                {
                    **row,
                    "production_claim_allowed": False,
                    "can_support_sota_claim": False,
                    "artifact_valid": False,
                }
            )
    return pending


def _row_is_valid_predicted_sql_manifest(row: dict[str, Any]) -> bool:
    return (
        row.get("artifact_type") == "result_manifest"
        and row.get("artifact_valid")
        and row.get("production_claim_allowed")
        and row.get("evaluation_mode") == "predicted_planner"
    )


def _row_is_valid_metric_dsl_manifest(row: dict[str, Any]) -> bool:
    return (
        row.get("artifact_type") == "result_manifest"
        and row.get("artifact_valid")
        and row.get("claim_status") == "supported_metric_dsl_quality"
        and row.get("benchmark") == METRIC_DSL
        and row.get("evaluation_mode") == METRIC_DSL
        and not row.get("oracle_allowed")
    )


def _row_has_metric_dsl_quality_support(row: dict[str, Any]) -> bool:
    if not _row_is_valid_metric_dsl_manifest(row):
        return False
    row_count = _num(row.get("row_count"))
    executed_rows = _num(row.get("compiled_sql_execution_evaluated_rows"))
    parse_rate = _num(row.get("metric_dsl_parse_rate"))
    compile_rate = _num(row.get("metric_dsl_compile_rate"))
    measure_preservation = _num(row.get("measure_preservation"))
    if row_count is None or executed_rows is None or executed_rows != row_count:
        return False
    return (
        parse_rate is not None
        and parse_rate > 0
        and compile_rate is not None
        and compile_rate > 0
        and measure_preservation is not None
        and measure_preservation > 0
    )


def _row_has_metric_dsl_direct_sql_comparison(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    if not _row_has_metric_dsl_quality_support(row):
        return False
    if not row.get("direct_sql_comparison_run_id"):
        return False
    command = [str(item) for item in row.get("command") or []]
    if "# compared-with-direct-sql" not in command:
        return False
    if str(row.get("direct_sql_comparison_run_id")) not in command:
        return False
    if not row.get("direct_sql_model_name"):
        return False
    if not row.get("direct_sql_input_sha256") or not row.get("direct_sql_output_sha256"):
        return False
    row_count = _num(row.get("row_count"))
    comparable_rows = _num(row.get("metric_dsl_comparable_row_count"))
    executed_rows = _num(row.get("compiled_sql_execution_evaluated_rows"))
    if (
        row_count is None
        or comparable_rows is None
        or executed_rows is None
        or comparable_rows != row_count
        or executed_rows != row_count
    ):
        return False
    if not _has_matching_metric_dsl_direct_sql_row(row, rows):
        return False
    direct_score = _num(row.get("direct_sql_value_execution_accuracy"))
    metric_score = _num(row.get("value_execution_accuracy"))
    delta = _num(row.get("metric_dsl_value_delta_vs_direct_sql"))
    measure_preservation = _num(row.get("metric_dsl_measure_preservation"))
    if measure_preservation is None:
        measure_preservation = _num(row.get("measure_preservation"))
    if direct_score is None or metric_score is None or delta is None:
        return False
    if measure_preservation is None or measure_preservation <= 0:
        return False
    return delta > 0 and metric_score > direct_score


def _has_matching_metric_dsl_direct_sql_row(
    metric_dsl_row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    direct_id = metric_dsl_row.get("direct_sql_comparison_run_id")
    for row in rows:
        if row.get("claim_id") != direct_id:
            continue
        if row.get("artifact_type") != "result_manifest" or not row.get("artifact_valid"):
            return False
        if not row.get("production_claim_allowed"):
            return False
        if row.get("evaluation_mode") != NON_ORACLE_GENERATION:
            return False
        if row.get("benchmark") != METRIC_DSL_DIRECT_SQL:
            return False
        if row.get("oracle_allowed"):
            return False
        if _num(row.get("row_count")) != _num(metric_dsl_row.get("metric_dsl_comparable_row_count")):
            return False
        if _num(row.get("output_value_scored_rows")) != _num(
            metric_dsl_row.get("metric_dsl_comparable_row_count")
        ):
            return False
        if _num(row.get("output_strict_scored_rows")) != _num(
            metric_dsl_row.get("metric_dsl_comparable_row_count")
        ):
            return False
        if row.get("model_name") != metric_dsl_row.get("direct_sql_model_name"):
            return False
        if row.get("input_sha256") != metric_dsl_row.get("direct_sql_input_sha256"):
            return False
        if row.get("output_sha256") != metric_dsl_row.get("direct_sql_output_sha256"):
            return False
        return _num(row.get("value_execution_accuracy")) == _num(
            metric_dsl_row.get("direct_sql_value_execution_accuracy")
        )
    return False


def _row_has_predicted_sql_claim_support(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    if not _row_is_valid_predicted_sql_manifest(row):
        return False
    if row.get("direct_sql_model_name") != row.get("model_name"):
        return False
    if not row.get("direct_sql_comparison_run_id"):
        return False
    if not row.get("direct_sql_input_sha256") or not row.get("direct_sql_output_sha256"):
        return False
    comparable_rows = _num(row.get("direct_sql_comparable_row_count"))
    row_count = _num(row.get("row_count"))
    if comparable_rows is None or row_count is None or comparable_rows != row_count:
        return False
    if not _has_matching_direct_sql_row(row, rows):
        return False
    direct_score = _num(row.get("direct_sql_value_execution_accuracy"))
    predicted_score = _num(row.get("value_execution_accuracy"))
    delta = _num(row.get("predicted_planner_value_delta_vs_direct_sql"))
    if direct_score is None or predicted_score is None or delta is None:
        return False
    return delta > 0 and predicted_score > direct_score


def _has_matching_direct_sql_row(
    predicted_row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    direct_id = predicted_row.get("direct_sql_comparison_run_id")
    for row in rows:
        if row.get("claim_id") != direct_id:
            continue
        if row.get("artifact_type") != "result_manifest" or not row.get("artifact_valid"):
            return False
        if not row.get("production_claim_allowed"):
            return False
        if row.get("evaluation_mode") != "non_oracle_generation":
            return False
        if row.get("benchmark") != "prepared":
            return False
        if row.get("oracle_allowed"):
            return False
        if row.get("model_name") != predicted_row.get("model_name"):
            return False
        if row.get("input_sha256") != predicted_row.get("direct_sql_input_sha256"):
            return False
        if row.get("output_sha256") != predicted_row.get("direct_sql_output_sha256"):
            return False
        return _num(row.get("value_execution_accuracy")) == _num(
            predicted_row.get("direct_sql_value_execution_accuracy")
        )
    return False


def _row_has_hosted_claim_support(row: dict[str, Any], rows: list[dict[str, Any]]) -> bool:
    if row.get("artifact_type") != "result_manifest" or not row.get("artifact_valid"):
        return False
    if not row.get("production_claim_allowed") or row.get("evaluation_mode") not in PRODUCTION_MODES:
        return False
    if row.get("oracle_allowed"):
        return False
    if not _is_hosted_endpoint(row.get("endpoint")):
        return False
    metric_keys = set(row.get("metric_keys") or [])
    if not (metric_keys & set(HOSTED_LATENCY_KEYS)) or not (metric_keys & set(HOSTED_COST_KEYS)):
        return False
    if not _row_accuracy_keys(row):
        return False
    return any(_row_matches_local_protocol(row, candidate) for candidate in rows)


def _is_hosted_endpoint(endpoint: Any) -> bool:
    return str(endpoint or "").startswith(HOSTED_ENDPOINT_PREFIXES)


def _row_accuracy_keys(row: dict[str, Any]) -> set[str]:
    return {key for key in COMPARISON_ACCURACY_KEYS if row.get(key) is not None}


def _row_matches_local_protocol(hosted: dict[str, Any], candidate: dict[str, Any]) -> bool:
    if candidate.get("claim_id") == hosted.get("claim_id"):
        return False
    if candidate.get("artifact_type") != "result_manifest" or not candidate.get("artifact_valid"):
        return False
    if not candidate.get("production_claim_allowed"):
        return False
    if candidate.get("oracle_allowed"):
        return False
    if _is_hosted_endpoint(candidate.get("endpoint")):
        return False
    if not (_row_accuracy_keys(hosted) & _row_accuracy_keys(candidate)):
        return False
    return (
        candidate.get("benchmark") == hosted.get("benchmark")
        and candidate.get("evaluation_mode") == hosted.get("evaluation_mode")
        and candidate.get("input_sha256") == hosted.get("input_sha256")
        and _num(candidate.get("row_count")) == _num(hosted.get("row_count"))
    )


def _row_has_rollout_claim_support(row: dict[str, Any]) -> bool:
    return (
        row.get("artifact_type") == "result_manifest"
        and row.get("artifact_valid")
        and row.get("production_claim_allowed")
        and row.get("benchmark") == "prepared_rollout"
        and row.get("history_policy") == MODEL_GENERATED_SQL_ROLLOUT
    )


def _row_has_rollout_teacher_forced_comparison(
    row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    if not _row_has_rollout_claim_support(row):
        return False
    if row.get("teacher_forced_model_name") != row.get("model_name"):
        return False
    if row.get("teacher_forced_input_sha256") != row.get("input_sha256"):
        return False
    if not row.get("teacher_forced_comparison_run_id"):
        return False
    comparable_rows = _num(row.get("teacher_forced_comparable_row_count"))
    row_count = _num(row.get("row_count"))
    if comparable_rows is None or row_count is None or comparable_rows != row_count:
        return False
    command = [str(item) for item in row.get("command") or []]
    if "# compared-with" not in command:
        return False
    if str(row.get("teacher_forced_comparison_run_id")) not in command:
        return False
    if not _has_matching_teacher_forced_row(row, rows):
        return False
    teacher_forced_score = _num(row.get("teacher_forced_value_execution_accuracy"))
    rollout_score = _num(row.get("value_execution_accuracy"))
    delta = _num(row.get("rollout_value_delta_vs_teacher_forced"))
    if teacher_forced_score is None or rollout_score is None or delta is None:
        return False
    return delta > 0 and rollout_score > teacher_forced_score


def _has_matching_teacher_forced_row(
    rollout_row: dict[str, Any],
    rows: list[dict[str, Any]],
) -> bool:
    teacher_id = rollout_row.get("teacher_forced_comparison_run_id")
    comparable_rows = _num(rollout_row.get("teacher_forced_comparable_row_count"))
    for row in rows:
        if row.get("claim_id") != teacher_id:
            continue
        if row.get("artifact_type") != "result_manifest" or not row.get("artifact_valid"):
            return False
        if not row.get("production_claim_allowed"):
            return False
        if row.get("evaluation_mode") != NON_ORACLE_GENERATION:
            return False
        if row.get("benchmark") != "prepared":
            return False
        if row.get("oracle_allowed"):
            return False
        if row.get("history_policy") != GOLD_SQL_TEACHER_FORCED:
            return False
        if row.get("model_name") != rollout_row.get("teacher_forced_model_name"):
            return False
        if row.get("input_sha256") != rollout_row.get("teacher_forced_input_sha256"):
            return False
        if _num(row.get("row_count")) != comparable_rows:
            return False
        if _num(row.get("output_value_scored_rows")) != comparable_rows:
            return False
        return _num(row.get("value_execution_accuracy")) == _num(
            rollout_row.get("teacher_forced_value_execution_accuracy")
        )
    return False


def build_claim_ledger(
    *,
    manifest_path: Path,
    repo_root: Path = Path("."),
    classified_dir: Path | None = Path("results/classified"),
    planner_summary_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build claim ledger rows from tracked manifests and analysis artifacts."""

    repo_root = repo_root.resolve()
    manifests = _load_json(manifest_path)
    if isinstance(manifests, dict):
        manifests = [manifests]
    resolved_classified_dir = _repo_path(repo_root, str(classified_dir)) if classified_dir else None
    rows = [
        _manifest_row(manifest, repo_root=repo_root, classified_dir=resolved_classified_dir)
        for manifest in manifests
    ]
    planner = _planner_row(_repo_path(repo_root, str(planner_summary_path)) if planner_summary_path else None)
    if planner:
        rows.append(planner)
    for row in rows:
        if row.get("artifact_type") == "result_manifest":
            row["can_support_sota_claim"] = _row_has_hosted_claim_support(row, rows)
    rows.extend(_pending_rows(rows))
    return rows


def write_claim_ledger(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_num(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{value:.12g}"


def write_claim_summary(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        fieldnames = [
            "claim_id",
            "claim_status",
            "evaluation_mode",
            "allowed_public_claim",
            "blocking_reason",
            "required_artifact",
            "value_execution_accuracy",
            "strict_execution_accuracy",
            "row_count",
            "dialog_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item.get("claim_id") or "")):
            writer.writerow(
                {
                    "claim_id": row.get("claim_id") or "",
                    "claim_status": row.get("claim_status") or "",
                    "evaluation_mode": row.get("evaluation_mode") or "",
                    "allowed_public_claim": row.get("allowed_public_claim") or "",
                    "blocking_reason": row.get("blocking_reason") or "",
                    "required_artifact": row.get("required_artifact") or "",
                    "value_execution_accuracy": _fmt_num(_num(row.get("value_execution_accuracy"))),
                    "strict_execution_accuracy": _fmt_num(_num(row.get("strict_execution_accuracy"))),
                    "row_count": str(row.get("row_count")) if row.get("row_count") is not None else "",
                    "dialog_count": str(row.get("dialog_count"))
                    if row.get("dialog_count") is not None
                    else "",
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/result_manifests/cosql_dev_100_proxy.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--classified-dir", type=Path, default=Path("results/classified"))
    parser.add_argument(
        "--planner-summary",
        type=Path,
        default=Path("docs/planner_baseline_cosql_dev_100_summary.json"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("docs/claim_ledgers/cosql_dev_100.jsonl"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("docs/claim_ledgers/cosql_dev_100_summary.csv"),
    )
    args = parser.parse_args()

    rows = build_claim_ledger(
        manifest_path=args.manifest,
        repo_root=args.repo_root,
        classified_dir=args.classified_dir,
        planner_summary_path=args.planner_summary,
    )
    write_claim_ledger(rows, args.output_jsonl)
    write_claim_summary(rows, args.output_summary)
    print(f"Wrote {len(rows)} claim ledger rows to {args.output_jsonl}")
    print(f"Wrote claim summary to {args.output_summary}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
