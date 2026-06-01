from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_docs_record_inconclusive_evidence_as_writeup_input() -> None:
    evidence_contract = (REPO_ROOT / "docs" / "evidence_contract.md").read_text(
        encoding="utf-8"
    )
    roadmap = (REPO_ROOT / "docs" / "research_roadmap.md").read_text(encoding="utf-8")
    blog_readme = (REPO_ROOT / "docs" / "blog" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "Inconclusive Evidence And Writeups" in evidence_contract
    assert "what was learned and what remains unproven" in evidence_contract
    assert "Inconclusive checkpoint evidence should stay visible" in roadmap
    assert "Inconclusive results should be cited" in blog_readme
