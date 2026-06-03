"""
Compare a local SQL result manifest against a same-protocol hosted baseline.

This does not run either model. It verifies that both result manifests were
produced on the same rows under the same non-oracle protocol, then writes an
augmented local manifest carrying the hosted comparison metrics required for a
hosted-baseline claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

NON_ORACLE_GENERATION = "non_oracle_generation"
PRODUCTION_MODES = {NON_ORACLE_GENERATION}
HOSTED_ENDPOINT_PREFIXES = ("https://", "anthropic:", "google:")
HOSTED_LATENCY_KEYS = (
    "mean_latency_ms",
    "mean_generation_latency_ms",
    "p50_latency_ms",
    "latency_ms",
)
HOSTED_COST_KEYS = ("total_cost_usd", "estimated_cost_usd", "cost_usd")


def _metric(manifest: dict[str, Any], name: str) -> float:
    metrics = manifest.get("metrics") or {}
    value = metrics.get(name)
    if value is None:
        raise ValueError(f"manifest {manifest.get('run_id')} is missing metric {name}")
    return float(value)


def _first_metric(manifest: dict[str, Any], keys: tuple[str, ...], *, label: str) -> float:
    metrics = manifest.get("metrics") or {}
    for key in keys:
        value = metrics.get(key)
        if value is not None:
            return float(value)
    raise ValueError(f"hosted manifest must include {label}")


def _is_hosted_endpoint(endpoint: Any) -> bool:
    return str(endpoint or "").startswith(HOSTED_ENDPOINT_PREFIXES)


def _validate_non_oracle(manifest: dict[str, Any], *, label: str) -> None:
    if manifest.get("oracle_allowed"):
        raise ValueError(f"{label} manifest must be non-oracle")


def _validate_local_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="local")
    if manifest.get("evaluation_mode") not in PRODUCTION_MODES:
        raise ValueError("local manifest must use a production evaluation mode")
    if _is_hosted_endpoint(manifest.get("endpoint")):
        raise ValueError("local manifest must not use a hosted endpoint")


def _validate_hosted_manifest(manifest: dict[str, Any]) -> None:
    _validate_non_oracle(manifest, label="hosted")
    if manifest.get("evaluation_mode") not in PRODUCTION_MODES:
        raise ValueError("hosted manifest must use a production evaluation mode")
    if not _is_hosted_endpoint(manifest.get("endpoint")):
        raise ValueError("hosted manifest must use a hosted endpoint")
    try:
        _first_metric(manifest, HOSTED_LATENCY_KEYS, label="cost and latency metrics")
        _first_metric(manifest, HOSTED_COST_KEYS, label="cost and latency metrics")
    except ValueError as exc:
        raise ValueError("hosted manifest must include cost and latency metrics") from exc


def _row_uses_oracle(row: dict[str, Any]) -> bool:
    return bool(row.get("semantic_model_oracle_derived"))


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str, str]:
    missing = [
        field
        for field in ("dialog_id", "turn_index", "database_id", "reference_sql")
        if field not in row
    ]
    if missing:
        raise ValueError("result row is missing identity field(s): " + ", ".join(missing))
    return (
        str(row.get("dialog_id")),
        str(row.get("turn_index")),
        str(row.get("database_id")),
        str(row.get("reference_sql")),
    )


def _validate_manifest_row_count(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    expected = int(manifest.get("row_count") or 0)
    if expected != len(rows):
        raise ValueError(f"{label} manifest row_count does not match output rows")


def _reject_duplicate_identities(
    identities: list[tuple[str, str, str, str]],
    *,
    label: str,
) -> None:
    seen = set()
    for identity in identities:
        if identity in seen:
            raise ValueError(f"duplicate {label} row identity")
        seen.add(identity)


def _validate_same_protocol(
    *,
    local_manifest: dict[str, Any],
    hosted_manifest: dict[str, Any],
) -> None:
    if local_manifest.get("benchmark") != hosted_manifest.get("benchmark"):
        raise ValueError("local and hosted manifests must use the same benchmark")
    if local_manifest.get("evaluation_mode") != hosted_manifest.get("evaluation_mode"):
        raise ValueError("local and hosted manifests must use the same evaluation mode")
    if local_manifest.get("input_sha256") != hosted_manifest.get("input_sha256"):
        raise ValueError("local and hosted manifests must use the same input")
    if int(local_manifest.get("row_count") or 0) != int(hosted_manifest.get("row_count") or 0):
        raise ValueError("local and hosted manifests must have matching row_count")


def _validate_rows(
    *,
    local_manifest: dict[str, Any],
    hosted_manifest: dict[str, Any],
    local_rows: list[dict[str, Any]],
    hosted_rows: list[dict[str, Any]],
) -> None:
    if not local_rows or not hosted_rows:
        raise ValueError("comparison requires non-empty output rows")
    _validate_manifest_row_count(local_manifest, local_rows, label="local")
    _validate_manifest_row_count(hosted_manifest, hosted_rows, label="hosted")
    mode = local_manifest.get("evaluation_mode")
    local_modes = {row.get("evaluation_mode") for row in local_rows}
    hosted_modes = {row.get("evaluation_mode") for row in hosted_rows}
    if local_modes != {mode} or hosted_modes != {mode}:
        raise ValueError("comparison output rows must use the manifest evaluation mode")
    if any(_row_uses_oracle(row) for row in local_rows + hosted_rows):
        raise ValueError("oracle-derived rows found in hosted comparison")

    local_identities = [_row_identity(row) for row in local_rows]
    hosted_identities = [_row_identity(row) for row in hosted_rows]
    _reject_duplicate_identities(local_identities, label="local")
    _reject_duplicate_identities(hosted_identities, label="hosted")
    if local_identities != hosted_identities:
        raise ValueError("local and hosted row identity mismatch")


def compare_hosted_baseline_manifests(
    *,
    local_manifest: dict[str, Any],
    hosted_manifest: dict[str, Any],
    local_rows: list[dict[str, Any]],
    hosted_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return an augmented local manifest with hosted-baseline comparison metrics."""

    _validate_local_manifest(local_manifest)
    _validate_hosted_manifest(hosted_manifest)
    _validate_same_protocol(local_manifest=local_manifest, hosted_manifest=hosted_manifest)
    _validate_rows(
        local_manifest=local_manifest,
        hosted_manifest=hosted_manifest,
        local_rows=local_rows,
        hosted_rows=hosted_rows,
    )

    local_value = _metric(local_manifest, "value_execution_accuracy")
    local_strict = _metric(local_manifest, "strict_execution_accuracy")
    hosted_value = _metric(hosted_manifest, "value_execution_accuracy")
    hosted_strict = _metric(hosted_manifest, "strict_execution_accuracy")
    hosted_latency = _first_metric(
        hosted_manifest,
        HOSTED_LATENCY_KEYS,
        label="cost and latency metrics",
    )
    hosted_cost = _first_metric(
        hosted_manifest,
        HOSTED_COST_KEYS,
        label="cost and latency metrics",
    )

    compared = dict(local_manifest)
    compared_metrics = dict(local_manifest.get("metrics") or {})
    compared_metrics.update(
        {
            "hosted_comparison_run_id": hosted_manifest.get("run_id"),
            "hosted_model_name": hosted_manifest.get("model_name"),
            "hosted_input_sha256": hosted_manifest.get("input_sha256"),
            "hosted_output_sha256": hosted_manifest.get("output_sha256"),
            "hosted_value_execution_accuracy": hosted_value,
            "hosted_strict_execution_accuracy": hosted_strict,
            "hosted_mean_latency_ms": hosted_latency,
            "hosted_total_cost_usd": hosted_cost,
            "local_value_delta_vs_hosted": local_value - hosted_value,
            "local_strict_delta_vs_hosted": local_strict - hosted_strict,
            "hosted_comparable_row_count": len(hosted_rows),
        }
    )
    compared["metrics"] = compared_metrics
    compared["command"] = list(local_manifest.get("command") or []) + [
        "# compared-with-hosted",
        str(hosted_manifest.get("run_id")),
    ]
    return compared


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_path(repo_root: Path, path_value: str | None) -> Path:
    if not path_value:
        raise ValueError("manifest is missing output_path")
    path = Path(path_value)
    return path if path.is_absolute() else repo_root / path


def _write_manifest(manifest: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def compare_hosted_baseline_manifest_files(
    *,
    local_manifest_path: Path,
    hosted_manifest_path: Path,
    output_path: Path,
    repo_root: Path = Path("."),
) -> dict[str, Any]:
    """Compare manifest files and write an augmented local manifest."""

    local_manifest = _load_manifest(local_manifest_path)
    hosted_manifest = _load_manifest(hosted_manifest_path)
    repo_root = repo_root.resolve()
    compared = compare_hosted_baseline_manifests(
        local_manifest=local_manifest,
        hosted_manifest=hosted_manifest,
        local_rows=_load_jsonl(_resolve_path(repo_root, local_manifest.get("output_path"))),
        hosted_rows=_load_jsonl(_resolve_path(repo_root, hosted_manifest.get("output_path"))),
    )
    _write_manifest(compared, output_path)
    return compared


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-manifest", type=Path, required=True)
    parser.add_argument("--hosted-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args()

    compared = compare_hosted_baseline_manifest_files(
        local_manifest_path=args.local_manifest,
        hosted_manifest_path=args.hosted_manifest,
        output_path=args.output,
        repo_root=args.repo_root,
    )
    metrics = compared["metrics"]
    print(f"Wrote local-vs-hosted comparison manifest to {args.output}")
    print(
        "Value delta vs hosted: "
        f"{metrics['local_value_delta_vs_hosted']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
