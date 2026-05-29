from __future__ import annotations

from pathlib import Path

from eval.benchmark_protocols import benchmark_protocol_map, load_benchmark_protocols

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_protocols_define_claim_boundaries() -> None:
    protocols = load_benchmark_protocols(REPO_ROOT / "configs" / "benchmark_protocols.yaml")

    protocol_ids = [row["protocol_id"] for row in protocols]
    assert protocol_ids == [
        "cosql_dev_100_teacher_forced_proxy",
        "cosql_generated_history_rollout",
        "sparc_context_transfer",
        "synthetic_schema_rich_method_fixture",
        "bird_interact_same_protocol_transfer",
    ]

    cosql = protocols[0]
    assert cosql["supports_method_ranking"] is True
    assert cosql["supports_hosted_sota_claim"] is False
    assert "BIRD-Interact transfer" in cosql["blocked_claims"]
    assert "future turns" in cosql["leakage_boundary"]

    synthetic = next(
        row for row in protocols if row["protocol_id"] == "synthetic_schema_rich_method_fixture"
    )
    assert synthetic["supports_method_ranking"] is False
    assert "real benchmark improvement" in synthetic["blocked_claims"]

    bird = protocols[-1]
    assert bird["supports_hosted_sota_claim"] is True
    assert bird["blocked_claims"] == ()
    assert "hosted result manifest" in bird["required_artifacts"]


def test_benchmark_protocol_loader_rejects_missing_contract_field(tmp_path) -> None:
    config = tmp_path / "protocols.yaml"
    config.write_text(
        """
schema_version: 1
protocols:
  - protocol_id: bad_protocol
    benchmark: Broken
    role: test
    turn_policy: fixed
    row_identity_policy: fixed
    scorer_policy: local
    primary_metrics:
      - value_accuracy
    required_artifacts:
      - manifest
    allowed_claims:
      - something
    blocked_claims: []
    supports_method_ranking: true
    supports_hosted_sota_claim: false
""",
        encoding="utf-8",
    )

    try:
        load_benchmark_protocols(config)
    except ValueError as exc:
        assert "bad_protocol: missing leakage_boundary" in str(exc)
    else:
        raise AssertionError("missing leakage boundary should fail")


def test_benchmark_protocol_map_is_keyed_by_stable_id() -> None:
    protocols = benchmark_protocol_map(REPO_ROOT / "configs" / "benchmark_protocols.yaml")

    assert protocols["sparc_context_transfer"]["benchmark"] == "SParC context-dependent SQL"
