"""Load compact experiment registry rows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_EXPERIMENT_REGISTRY = Path("configs/experiments.yaml")

REQUIRED_TEXT_FIELDS = {
    "experiment_id",
    "hypothesis_id",
    "hypothesis",
    "train_split_id",
    "validation_split_id",
    "test_split_id",
    "dataset_role",
    "benchmark_protocol_id",
    "model",
    "method",
    "oracle_policy",
    "scorer",
    "output_path",
    "primary_metric",
    "claim_boundary",
    "status",
}

ALLOWED_DATASET_ROLES = {
    "train",
    "validation",
    "proxy_dev_seen",
    "clean_local_holdout",
    "external_target",
}


def load_experiment_registry(path: Path = DEFAULT_EXPERIMENT_REGISTRY) -> tuple[dict[str, Any], ...]:
    """Return normalized experiment registry rows."""

    payload = yaml.safe_load(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("experiment registry must use schema_version=1")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("experiment registry must include non-empty experiments")

    normalized = []
    seen_ids = set()
    for experiment in experiments:
        row = dict(experiment)
        experiment_id = row.get("experiment_id") or "<unknown experiment>"
        missing_text_fields = [
            field
            for field in sorted(REQUIRED_TEXT_FIELDS)
            if not str(row.get(field) or "").strip()
        ]
        if missing_text_fields:
            raise ValueError(f"{experiment_id}: missing {', '.join(missing_text_fields)}")
        if experiment_id in seen_ids:
            raise ValueError(f"{experiment_id}: duplicate experiment id")
        seen_ids.add(experiment_id)
        if row["dataset_role"] not in ALLOWED_DATASET_ROLES:
            raise ValueError(f"{experiment_id}: invalid dataset_role {row['dataset_role']!r}")
        if not isinstance(row.get("checkpoint"), int):
            raise ValueError(f"{experiment_id}: checkpoint must be an integer")
        if row.get("adapter") is None:
            row["adapter"] = ""
        if row.get("control_experiment_id") is None:
            row["control_experiment_id"] = ""
        output_path = str(row["output_path"])
        if not output_path.startswith("results/"):
            raise ValueError(f"{experiment_id}: output_path must be under results/")
        normalized.append(row)
    experiment_ids = {row["experiment_id"] for row in normalized}
    for row in normalized:
        experiment_id = row["experiment_id"]
        control_id = row["control_experiment_id"]
        if row["method"] != "direct_sql" and not control_id:
            raise ValueError(f"{experiment_id}: comparison experiments must name control_experiment_id")
        if control_id:
            if control_id == experiment_id:
                raise ValueError(f"{experiment_id}: control_experiment_id cannot reference itself")
            if control_id not in experiment_ids:
                raise ValueError(f"{experiment_id}: unknown control_experiment_id {control_id!r}")
    return tuple(normalized)


def experiment_registry_map(path: Path = DEFAULT_EXPERIMENT_REGISTRY) -> dict[str, dict[str, Any]]:
    """Return experiment rows keyed by stable id."""

    return {row["experiment_id"]: row for row in load_experiment_registry(path)}
