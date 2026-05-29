"""Build a compact rollup of hypothesis-arm evidence artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval.result_manifest import sha256_file

DEFAULT_EVIDENCE_PATHS = (
    Path("docs/training_runs/metric_dsl_prompt_baseline.json"),
    Path("docs/training_runs/metric_dsl_arm_5steps.json"),
    Path("docs/training_runs/behavior_recovery_5steps.json"),
    Path("docs/training_runs/value_schema_repair_prompt.json"),
    Path("docs/training_runs/value_choice_consistency_prompt.json"),
    Path("docs/training_runs/alias_column_validity_prompt.json"),
    Path("docs/training_runs/alias_column_context_prompt_limit8.json"),
)

RUN_SUMMARIES = {
    "metric_dsl_prompt_baseline": {
        "hypothesis_id": "metric_dsl",
        "arm_type": "prompt_only_baseline",
        "control": "direct_sql_prompt",
        "decision": "failed",
        "primary_failure_mode": "The model did not follow the metric DSL output contract.",
        "next_action": "Separate constrained-output obedience from metric semantics before scaling.",
    },
    "metric_dsl_arm_5steps": {
        "hypothesis_id": "metric_dsl",
        "arm_type": "five_step_lora",
        "control": "direct_sql_five_step_lora",
        "decision": "failed",
        "primary_failure_mode": "The adapter still emitted invalid DSL while direct SQL solved one fixture.",
        "next_action": "Do not claim DSL benefit until parse and compile rates are nonzero.",
    },
    "behavior_recovery_5steps": {
        "hypothesis_id": "behavior_recovery",
        "arm_type": "five_step_lora",
        "control": "direct_sql_five_step_lora",
        "decision": "no_delta",
        "primary_failure_mode": (
            "Recovery and direct-SQL adapters made the same value-normalization "
            "and invalid-column errors."
        ),
        "next_action": "Test value and schema repair context before another adapter run.",
    },
    "value_schema_repair_prompt": {
        "hypothesis_id": "value_schema_repair",
        "arm_type": "prompt_only_diagnostic",
        "control": "teacher_forced_reference",
        "decision": "failed_changed_failure_mode",
        "primary_failure_mode": (
            "The prompt fixed schema validity but chose the wrong storage value."
        ),
        "next_action": "Add a value-choice consistency artifact and score value choice separately.",
    },
    "value_choice_consistency_prompt": {
        "hypothesis_id": "value_schema_repair",
        "arm_type": "prompt_only_value_choice_diagnostic",
        "control": "teacher_forced_reference",
        "decision": "partial_pass_value_choice_failed_sql",
        "primary_failure_mode": (
            "The model chose FR correctly but reintroduced the invalid customers.customer_id column."
        ),
        "next_action": "Add alias/column-validity supervision after value choice is correct.",
    },
    "alias_column_validity_prompt": {
        "hypothesis_id": "value_schema_repair",
        "arm_type": "prompt_only_alias_column_diagnostic",
        "control": "teacher_forced_reference",
        "decision": "passed_single_synthetic_diagnostic",
        "primary_failure_mode": (
            "The explicit column-role constraints fixed the known invalid-column failure."
        ),
        "next_action": "Run a row-matched prompt-only CoSQL validation comparison.",
    },
    "alias_column_context_prompt_limit8": {
        "hypothesis_id": "value_schema_repair",
        "arm_type": "prompt_only_validation_slice",
        "control": "direct_sql_prompt",
        "decision": "no_delta_on_limit8_validation_slice",
        "primary_failure_mode": (
            "Column-valid SQL still missed value execution; remaining failures are query-shape/semantic, not invalid-column."
        ),
        "next_action": "Run a larger row-matched slice and classify remaining failures before training.",
    },
}

METRIC_KEYS = (
    "rows",
    "dialog_count",
    "syntax_accuracy",
    "value_execution_accuracy",
    "strict_execution_accuracy",
    "metric_dsl_parse_rate",
    "metric_dsl_compile_rate",
    "metric_dsl_value_delta_vs_direct_sql",
    "direct_sql_value_execution_accuracy",
    "behavior_recovery_value_delta_vs_direct_sql",
    "rollout_value_delta_vs_teacher_forced",
    "value_choice_accuracy",
    "column_validity_accuracy",
    "schema_valid_sql_rate",
    "alias_resolution_success_rate",
    "alias_column_context_value_delta_vs_direct_sql",
    "direct_sql_value_execution_accuracy",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _selected_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: metrics[key] for key in METRIC_KEYS if key in metrics}


def _observed_outputs(payload: dict[str, Any]) -> list[dict[str, Any]]:
    for key in (
        "observed_output",
        "observed_outputs",
        "observed_recovery_output",
        "observed_failures",
        "direct_sql_control_outputs",
    ):
        rows = payload.get(key)
        if isinstance(rows, list):
            return rows[:3]
    return []


def build_hypothesis_rollup(
    *,
    repo_root: Path,
    evidence_paths: tuple[Path, ...] = DEFAULT_EVIDENCE_PATHS,
) -> dict[str, Any]:
    arms = []
    for relative_path in evidence_paths:
        path = repo_root / relative_path
        payload = _load_json(path)
        run_id = str(payload["run_id"])
        summary = RUN_SUMMARIES[run_id]
        arms.append(
            {
                "run_id": run_id,
                "hypothesis_id": summary["hypothesis_id"],
                "arm_type": summary["arm_type"],
                "evidence_path": str(relative_path),
                "evidence_sha256": sha256_file(path),
                "claim_boundary": payload.get("claim_boundary"),
                "model": payload.get("model"),
                "control": summary["control"],
                "decision": summary["decision"],
                "primary_failure_mode": summary["primary_failure_mode"],
                "selected_metrics": _selected_metrics(payload.get("metrics", {})),
                "observed_outputs": _observed_outputs(payload),
                "next_action": summary["next_action"],
            }
        )

    return {
        "schema_version": 1,
        "artifact_type": "hypothesis_arm_rollup",
        "claim_boundary": (
            "This rollup summarizes tiny diagnostic arms. It is not a benchmark, "
            "a SOTA claim, or evidence that any method is promoted."
        ),
        "evidence_paths": [str(path) for path in evidence_paths],
        "arms": arms,
        "overall_decision": {
            "status": "no_method_promoted",
            "reason": (
                "The alias/column context passed one synthetic diagnostic but "
                "showed no value delta on the first eight CoSQL validation turns. "
                "No method has passed broader validation or a locked proxy gate."
            ),
            "next_repo_step": (
                "Run a larger row-matched alias/column validation slice and "
                "classify remaining failures before another GPU fine-tuning run "
                "or blog claim."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", default="docs/training_runs/hypothesis_arm_rollup.json")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    output_path = repo_root / args.output
    payload = build_hypothesis_rollup(repo_root=repo_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
