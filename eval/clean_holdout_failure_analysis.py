"""Build clean-holdout failure-analysis artifacts for Roadmap Checkpoint 4."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from eval.classify_errors import classify_file
from eval.compare_failures import (
    load_classified_files,
    write_model_summary,
    write_pairwise_comparison,
)
from eval.result_manifest import sha256_file

DEFAULT_CONFIG = Path("configs/direct_sql_full_non_oracle.yaml")
DEFAULT_ARTIFACT_DIR_NAME = "clean_holdout_failure_analysis"
LABEL_TO_METHOD_HINTS = {
    "schema_link": ["planner_schema_linking"],
    "join_path": ["planner_schema_linking"],
    "aggregation": ["planner_schema_linking", "metric_definitions"],
    "grain_fanout": ["planner_schema_linking", "metric_definitions"],
    "projection": ["planner_schema_linking"],
    "value_grounding": ["semantic_value_grounding"],
    "invalid_sql": ["recovery"],
    "execution_error": ["recovery"],
    "history_resolution": ["recovery"],
    "ordering_limit": ["planner_schema_linking"],
    "other": ["unresolved_other"],
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve(base_dir: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _run_label(manifest: dict[str, Any]) -> str:
    model_name = str(manifest.get("model_name") or "")
    prompt_variant = manifest.get("prompt_variant")
    return f"{model_name}[{prompt_variant}]" if prompt_variant else model_name


def _clean_holdout_result_entries(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = (config.get("checkpoint3_artifacts") or {}).get("result_manifests") or {}
    required = {}
    for name in ("base_clean_holdout", "lora_clean_holdout"):
        entry = entries.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"checkpoint3_artifacts.result_manifests.{name} is required")
        required[name] = entry
    return required


def _assert_same_rows(
    *,
    base_manifest: dict[str, Any],
    lora_manifest: dict[str, Any],
) -> None:
    base_metrics = base_manifest.get("metrics") or {}
    lora_metrics = lora_manifest.get("metrics") or {}
    for field in ("split_row_ids_sha256", "split_eval_turn_ids_sha256"):
        if base_metrics.get(field) != lora_metrics.get(field):
            raise ValueError(f"clean-holdout base/LoRA {field} mismatch")


def _method_hint_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        primary = str(row.get("error_primary") or "other")
        if primary == "correct":
            continue
        for hint in LABEL_TO_METHOD_HINTS.get(primary, ["unresolved_other"]):
            counts[hint] += 1
    return dict(sorted(counts.items()))


def _primary_label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("error_primary") or "other") for row in rows).items()))


def _default_output_dir(base_manifest_path: Path) -> Path:
    return base_manifest_path.parent / DEFAULT_ARTIFACT_DIR_NAME


def build_clean_holdout_failure_analysis(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_dir: Path | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Classify clean-holdout base/LoRA outputs and write comparison artifacts."""

    base_dir = config_path.resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    entries = _clean_holdout_result_entries(config)
    base_manifest_path = _resolve(base_dir, entries["base_clean_holdout"]["manifest"])
    lora_manifest_path = _resolve(base_dir, entries["lora_clean_holdout"]["manifest"])
    base_manifest = _load_json(base_manifest_path)
    lora_manifest = _load_json(lora_manifest_path)
    _assert_same_rows(base_manifest=base_manifest, lora_manifest=lora_manifest)

    output_dir = output_dir or _default_output_dir(base_manifest_path)
    base_output = _resolve(base_dir, base_manifest["output_path"])
    lora_output = _resolve(base_dir, lora_manifest["output_path"])
    base_classified = output_dir / "base_clean_holdout.classified.jsonl"
    lora_classified = output_dir / "lora_clean_holdout.classified.jsonl"
    base_summary = output_dir / "base_clean_holdout.error_summary.csv"
    lora_summary = output_dir / "lora_clean_holdout.error_summary.csv"
    model_summary = output_dir / "model_error_summary.csv"
    pairwise_summary = output_dir / "pairwise_vs_base.csv"
    analysis_manifest_path = output_dir / "manifest.json"

    base_count = classify_file(base_output, base_classified, base_summary)
    lora_count = classify_file(lora_output, lora_classified, lora_summary)
    runs = load_classified_files([base_classified, lora_classified])
    baseline_label = _run_label(base_manifest)
    write_model_summary(runs, model_summary)
    write_pairwise_comparison(runs, baseline_label=baseline_label, output=pairwise_summary)

    base_rows = _load_jsonl(base_classified)
    lora_rows = _load_jsonl(lora_classified)
    manifest = {
        "schema_version": 1,
        "artifact_type": "clean_holdout_failure_analysis",
        "config_path": str(config_path),
        "base_result_manifest_path": str(base_manifest_path.relative_to(base_dir)),
        "base_result_manifest_sha256": sha256_file(base_manifest_path),
        "lora_result_manifest_path": str(lora_manifest_path.relative_to(base_dir)),
        "lora_result_manifest_sha256": sha256_file(lora_manifest_path),
        "split_id": entries["base_clean_holdout"].get("split_id"),
        "split_role": entries["base_clean_holdout"].get("split_role"),
        "baseline_label": baseline_label,
        "classified_outputs": {
            "base": str(base_classified),
            "lora": str(lora_classified),
        },
        "summaries": {
            "base_error_summary": str(base_summary),
            "lora_error_summary": str(lora_summary),
            "model_error_summary": str(model_summary),
            "pairwise_vs_base": str(pairwise_summary),
        },
        "row_counts": {
            "base": base_count,
            "lora": lora_count,
        },
        "primary_error_counts": {
            "base": _primary_label_counts(base_rows),
            "lora": _primary_label_counts(lora_rows),
        },
        "roadmap_method_hint_counts": {
            "base": _method_hint_counts(base_rows),
            "lora": _method_hint_counts(lora_rows),
        },
        "label_to_method_hints": LABEL_TO_METHOD_HINTS,
        "command": command or [],
    }
    analysis_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    manifest = build_clean_holdout_failure_analysis(
        config_path=args.config,
        output_dir=args.output_dir,
        command=[
            "uv",
            "run",
            "--active",
            "--no-sync",
            "python",
            "-m",
            "eval.clean_holdout_failure_analysis",
            "--config",
            str(args.config),
            *(["--output-dir", str(args.output_dir)] if args.output_dir else []),
        ],
    )
    print(f"Wrote clean-holdout failure analysis to {manifest['summaries']['model_error_summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
