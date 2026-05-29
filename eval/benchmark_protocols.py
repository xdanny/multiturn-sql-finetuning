"""Load benchmark protocol claim boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROTOCOL_CONFIG = Path("configs/benchmark_protocols.yaml")

LIST_FIELDS = {
    "primary_metrics",
    "required_artifacts",
    "allowed_claims",
    "blocked_claims",
}
REQUIRED_TEXT_FIELDS = {
    "protocol_id",
    "benchmark",
    "role",
    "turn_policy",
    "row_identity_policy",
    "scorer_policy",
    "leakage_boundary",
}


def load_benchmark_protocols(path: Path = DEFAULT_PROTOCOL_CONFIG) -> tuple[dict[str, Any], ...]:
    """Return normalized benchmark protocol rows."""

    payload = yaml.safe_load(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("benchmark protocol config must use schema_version=1")
    protocols = payload.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("benchmark protocol config must include non-empty protocols")

    normalized = []
    seen_ids = set()
    for protocol in protocols:
        row = dict(protocol)
        missing_text_fields = [
            field
            for field in sorted(REQUIRED_TEXT_FIELDS)
            if not str(row.get(field) or "").strip()
        ]
        protocol_id = row.get("protocol_id") or "<unknown protocol>"
        if missing_text_fields:
            raise ValueError(f"{protocol_id}: missing {', '.join(missing_text_fields)}")
        if protocol_id in seen_ids:
            raise ValueError(f"{protocol_id}: duplicate benchmark protocol id")
        seen_ids.add(protocol_id)
        for field in LIST_FIELDS:
            row[field] = tuple(row.get(field) or ())
            if not row[field] and field != "blocked_claims":
                raise ValueError(f"{protocol_id}: missing non-empty {field}")
        for field in ("supports_method_ranking", "supports_hosted_sota_claim"):
            if not isinstance(row.get(field), bool):
                raise ValueError(f"{protocol_id}: {field} must be a boolean")
        normalized.append(row)
    return tuple(normalized)


def benchmark_protocol_map(path: Path = DEFAULT_PROTOCOL_CONFIG) -> dict[str, dict[str, Any]]:
    """Return protocols keyed by stable id."""

    return {row["protocol_id"]: row for row in load_benchmark_protocols(path)}
