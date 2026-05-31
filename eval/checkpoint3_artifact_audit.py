"""Audit Roadmap Checkpoint 3 direct-SQL control artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AuditResult:
    ok: bool
    issues: tuple[str, ...]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base_dir: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base_dir / candidate


def _has_only_non_oracle_modes(manifest: dict[str, Any]) -> bool:
    modes = manifest.get("evaluation_modes") or {}
    return bool(modes) and set(modes) == {"non_oracle_generation"}


def _audit_prepared_data(config: dict[str, Any], *, base_dir: Path) -> list[str]:
    issues = []
    expected_roles = {
        "train": "train",
        "proxy_dev_seen": "proxy_dev_seen",
        "clean_local_holdout": "clean_local_holdout",
    }
    prepared = config.get("prepared_data") or {}
    for key, expected_role in expected_roles.items():
        entry = prepared.get(key)
        if not isinstance(entry, dict):
            issues.append(f"prepared_data.{key}: missing entry")
            continue
        manifest_path = _resolve(base_dir, entry.get("manifest", ""))
        output_path = _resolve(base_dir, entry.get("path", ""))
        if not manifest_path.exists():
            issues.append(f"prepared_data.{key}: missing manifest {manifest_path}")
            continue
        if not output_path.exists():
            issues.append(f"prepared_data.{key}: missing prepared JSONL {output_path}")
        manifest = _load_json(manifest_path)
        if manifest.get("split_id") != entry.get("split_id"):
            issues.append(f"prepared_data.{key}: split_id mismatch")
        if manifest.get("split_role") != expected_role:
            issues.append(f"prepared_data.{key}: expected split_role {expected_role}")
        if not int(manifest.get("row_count") or 0):
            issues.append(f"prepared_data.{key}: row_count must be positive")
        if manifest.get("oracle_policy") != "non_oracle_generation":
            issues.append(f"prepared_data.{key}: oracle_policy must be non_oracle_generation")
        if not _has_only_non_oracle_modes(manifest):
            issues.append(f"prepared_data.{key}: evaluation_modes must be non_oracle_generation only")
        if manifest.get("uses_oracle_planning_hints"):
            issues.append(f"prepared_data.{key}: uses_oracle_planning_hints must be false")
        if manifest.get("semantic_context_pruned_by_oracle_labels"):
            issues.append(
                f"prepared_data.{key}: semantic_context_pruned_by_oracle_labels must be false"
            )
    return issues


def _metric_missing(metrics: dict[str, Any], name: str) -> bool:
    return metrics.get(name) in (None, "")


def _audit_result_manifest(
    *,
    name: str,
    entry: dict[str, Any],
    base_dir: Path,
    required_metrics: list[str],
) -> tuple[list[str], dict[str, Any] | None]:
    issues = []
    manifest_path = _resolve(base_dir, entry.get("manifest", ""))
    if not manifest_path.exists():
        return [f"{name}: missing result manifest {manifest_path}"], None
    manifest = _load_json(manifest_path)
    metrics = manifest.get("metrics") or {}
    if manifest.get("schema_version") != 1:
        issues.append(f"{name}: manifest schema_version must be 1")
    if manifest.get("benchmark") != "prepared":
        issues.append(f"{name}: benchmark must be prepared")
    if manifest.get("oracle_allowed") is not False:
        issues.append(f"{name}: oracle_allowed must be false")
    if manifest.get("evaluation_mode") != "non_oracle_generation":
        issues.append(f"{name}: evaluation_mode must be non_oracle_generation")
    if not int(manifest.get("row_count") or 0):
        issues.append(f"{name}: row_count must be positive")
    if not manifest.get("output_sha256"):
        issues.append(f"{name}: output_sha256 is required")
    for metric in required_metrics:
        if _metric_missing(metrics, metric):
            issues.append(f"{name}: missing metric {metric}")
    split_id = entry.get("split_id")
    split_role = entry.get("split_role")
    if split_id and split_id not in (metrics.get("split_ids") or {}):
        issues.append(f"{name}: missing split_id {split_id} in metrics")
    if split_role and split_role not in (metrics.get("split_roles") or {}):
        issues.append(f"{name}: missing split_role {split_role} in metrics")
    if not metrics.get("split_row_ids_sha256"):
        issues.append(f"{name}: missing split_row_ids_sha256")
    if not metrics.get("split_eval_turn_ids_sha256"):
        issues.append(f"{name}: missing split_eval_turn_ids_sha256")
    return issues, manifest


def _audit_eval_manifests(config: dict[str, Any], *, base_dir: Path) -> list[str]:
    artifact_config = config.get("checkpoint3_artifacts") or {}
    required_metrics = list(artifact_config.get("required_eval_metrics") or [])
    entries = artifact_config.get("result_manifests") or {}
    issues = []
    loaded: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            issues.append(f"{name}: result manifest entry must be an object")
            continue
        entry_issues, manifest = _audit_result_manifest(
            name=f"result_manifests.{name}",
            entry=entry,
            base_dir=base_dir,
            required_metrics=required_metrics,
        )
        issues.extend(entry_issues)
        if manifest is not None:
            loaded[name] = (entry, manifest)

    by_split_role: dict[str, dict[str, dict[str, Any]]] = {}
    for entry, manifest in loaded.values():
        split_role = str(entry.get("split_role") or "")
        model_role = str(entry.get("model_role") or "")
        if split_role and model_role:
            by_split_role.setdefault(split_role, {})[model_role] = manifest
    for split_role, manifests in sorted(by_split_role.items()):
        base = manifests.get("base")
        lora = manifests.get("lora")
        if not base or not lora:
            issues.append(f"{split_role}: both base and lora result manifests are required")
            continue
        base_metrics = base.get("metrics") or {}
        lora_metrics = lora.get("metrics") or {}
        for field in ("split_row_ids_sha256", "split_eval_turn_ids_sha256"):
            if base_metrics.get(field) != lora_metrics.get(field):
                issues.append(f"{split_role}: base/lora {field} mismatch")
    return issues


def _audit_rollout_manifests(config: dict[str, Any], *, base_dir: Path) -> list[str]:
    artifact_config = config.get("checkpoint3_artifacts") or {}
    entries = artifact_config.get("generated_history_rollout_manifests") or {}
    issues = []
    seen_roles = set()
    for name, entry in entries.items():
        if not isinstance(entry, dict):
            issues.append(f"generated_history_rollout_manifests.{name}: entry must be an object")
            continue
        manifest_path = _resolve(base_dir, entry.get("manifest", ""))
        model_role = entry.get("model_role")
        if model_role:
            seen_roles.add(model_role)
        if not manifest_path.exists():
            issues.append(
                f"generated_history_rollout_manifests.{name}: missing manifest {manifest_path}"
            )
            continue
        manifest = _load_json(manifest_path)
        metrics = manifest.get("metrics") or {}
        if manifest.get("oracle_allowed") is not False:
            issues.append(f"generated_history_rollout_manifests.{name}: oracle_allowed must be false")
        if not int(manifest.get("row_count") or 0):
            issues.append(f"generated_history_rollout_manifests.{name}: row_count must be positive")
        for metric in entry.get("required_metrics") or []:
            if _metric_missing(metrics, metric):
                issues.append(f"generated_history_rollout_manifests.{name}: missing metric {metric}")
    if entries and seen_roles != {"base", "lora"}:
        issues.append("generated_history_rollout_manifests: base and lora manifests are required")
    return issues


def audit_checkpoint3_artifacts(
    config_path: Path = Path("configs/direct_sql_full_non_oracle.yaml"),
) -> AuditResult:
    """Return whether the direct-SQL full-control evidence contract is satisfied."""

    base_dir = config_path.resolve().parents[1]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    issues = [
        *_audit_prepared_data(config, base_dir=base_dir),
        *_audit_eval_manifests(config, base_dir=base_dir),
        *_audit_rollout_manifests(config, base_dir=base_dir),
    ]
    return AuditResult(ok=not issues, issues=tuple(issues))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/direct_sql_full_non_oracle.yaml"))
    args = parser.parse_args()

    result = audit_checkpoint3_artifacts(args.config)
    if result.ok:
        print("Checkpoint 3 direct-SQL artifact contract is satisfied.")
        return 0
    print("Checkpoint 3 direct-SQL artifact contract is incomplete:")
    for issue in result.issues:
        print(f"- {issue}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
