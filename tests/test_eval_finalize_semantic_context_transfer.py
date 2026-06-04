from __future__ import annotations

import json
from pathlib import Path

from eval.finalize_semantic_context_transfer import finalize_semantic_context_transfer


def test_finalize_semantic_context_transfer_writes_all_comparisons_and_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    local_normal = tmp_path / "local.normal.manifest.json"
    local_semantic = tmp_path / "local.semantic.manifest.json"
    hosted_normal = tmp_path / "hosted.normal.manifest.json"
    hosted_semantic = tmp_path / "hosted.semantic.manifest.json"
    preflight = tmp_path / "preflight.json"
    for path in (local_normal, local_semantic, hosted_normal, hosted_semantic, preflight):
        path.write_text("{}\n")
    output_dir = tmp_path / "results"
    semantic_calls: list[dict] = []
    hosted_calls: list[dict] = []
    summary_calls: list[dict] = []

    def fake_semantic_compare(**kwargs):
        semantic_calls.append(kwargs)
        kwargs["output_path"].write_text(
            json.dumps({"metrics": {"semantic_context_transfer_blockers": []}}) + "\n"
        )
        return {"metrics": {"semantic_context_transfer_blockers": []}}

    def fake_hosted_compare(**kwargs):
        hosted_calls.append(kwargs)
        kwargs["output_path"].write_text(
            json.dumps({"metrics": {"local_value_delta_vs_hosted": -0.1}}) + "\n"
        )
        return {"metrics": {"local_value_delta_vs_hosted": -0.1}}

    def fake_summary(**kwargs):
        summary_calls.append(kwargs)
        summary = {
            "artifact_type": "semantic_context_transfer_evidence_summary",
            "promotion_status": "negative_or_inconclusive",
        }
        kwargs["output_path"].write_text(json.dumps(summary) + "\n")
        return summary

    monkeypatch.setattr(
        "eval.finalize_semantic_context_transfer.compare_semantic_context_transfer_manifest_files",
        fake_semantic_compare,
    )
    monkeypatch.setattr(
        "eval.finalize_semantic_context_transfer.compare_hosted_baseline_manifest_files",
        fake_hosted_compare,
    )
    monkeypatch.setattr(
        "eval.finalize_semantic_context_transfer.summarize_semantic_context_transfer_evidence_files",
        fake_summary,
    )

    summary = finalize_semantic_context_transfer(
        local_normal_manifest=local_normal,
        local_semantic_manifest=local_semantic,
        hosted_normal_manifest=hosted_normal,
        hosted_semantic_manifest=hosted_semantic,
        preflight_manifest=preflight,
        output_dir=output_dir,
        run_id="cp10",
        repo_root=tmp_path,
        command=["finalize-cp10"],
    )

    local_context_path = output_dir / "cp10.local.semantic_context_comparison.manifest.json"
    hosted_context_path = (
        output_dir / "cp10.hosted.semantic_context_comparison.manifest.json"
    )
    normal_gap_path = output_dir / "cp10.normal.local_vs_hosted.manifest.json"
    semantic_gap_path = output_dir / "cp10.semantic.local_vs_hosted.manifest.json"
    summary_path = output_dir / "cp10.evidence_summary.json"

    assert [call["normal_manifest_path"] for call in semantic_calls] == [
        local_normal,
        hosted_normal,
    ]
    assert [call["semantic_manifest_path"] for call in semantic_calls] == [
        local_semantic,
        hosted_semantic,
    ]
    assert [call["comparison_role"] for call in semantic_calls] == [
        "best_local",
        "hosted_sonnet",
    ]
    assert all(call["preflight_manifest_path"] == preflight for call in semantic_calls)
    assert [call["output_path"] for call in semantic_calls] == [
        local_context_path,
        hosted_context_path,
    ]
    assert hosted_calls == [
        {
            "local_manifest_path": local_normal,
            "hosted_manifest_path": hosted_normal,
            "output_path": normal_gap_path,
            "repo_root": tmp_path,
        },
        {
            "local_manifest_path": local_semantic,
            "hosted_manifest_path": hosted_semantic,
            "output_path": semantic_gap_path,
            "repo_root": tmp_path,
        },
    ]
    assert summary_calls == [
        {
            "local_context_comparison_path": local_context_path,
            "hosted_context_comparison_path": hosted_context_path,
            "normal_local_vs_hosted_comparison_path": normal_gap_path,
            "semantic_local_vs_hosted_comparison_path": semantic_gap_path,
            "output_path": summary_path,
        }
    ]
    assert summary["finalizer"]["command"] == ["finalize-cp10"]
    assert summary["finalizer"]["summary_path"] == str(summary_path)
    assert json.loads(summary_path.read_text()) == summary
