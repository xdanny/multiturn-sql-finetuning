from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_finetuning_ladder_documents_structured_training_program() -> None:
    ladder = (REPO_ROOT / "docs" / "finetuning_ladder.md").read_text()

    for phrase in [
        "# Finetuning Ladder",
        "Stage 0: Direct SQL control",
        "Stage 1: Planner supervision",
        "Stage 2: Predicted-planner SQL",
        "Stage 3: Semantic-layer tuning",
        "Stage 4: MEASURE()-preserving metric DSL",
        "Stage 5: Generated-history recovery",
        "Stage 6: Hosted and BIRD-Interact comparison",
        "Training target",
        "Prepared data",
        "Trainer invocation",
        "Evaluation gate",
        "Claim boundary",
        "uv run python -m train.finetune",
        "uv run python -m eval.run_eval",
        "uv run python -m eval.run_predicted_planner_comparison",
        "uv run python -m eval.metric_dsl_eval",
        "uv run python -m eval.compare_rollout_history",
    ]:
        assert phrase in ladder


def test_root_docs_reference_finetuning_ladder() -> None:
    readme = (REPO_ROOT / "README.md").read_text()
    goal = (REPO_ROOT / "docs" / "research_goal.md").read_text()
    methodology = (REPO_ROOT / "docs" / "methodology.md").read_text()

    assert "docs/finetuning_ladder.md" in readme
    assert "docs/finetuning_ladder.md" in goal
    assert "docs/finetuning_ladder.md" in methodology
    assert "data.finetuning_program_registry" in readme
    assert "docs/data_artifacts/finetuning_program_registry.json" in methodology


def test_agents_guidance_exists_at_repo_and_domain_levels() -> None:
    required_agents = {
        REPO_ROOT / "AGENTS.md": [
            "docs/research_goal.md",
            "docs/finetuning_ladder.md",
            "same rows",
            "oracle",
        ],
        REPO_ROOT / "data" / "AGENTS.md": [
            "prepared data",
            "artifact",
            "oracle",
            "semantic",
            "finetuning_program_registry",
        ],
        REPO_ROOT / "eval" / "AGENTS.md": [
            "manifest",
            "comparison",
            "claim",
            "same-model",
        ],
        REPO_ROOT / "train" / "AGENTS.md": [
            "train.finetune",
            "allow-oracle-diagnostic-data",
            "run naming",
            "mixture",
        ],
        REPO_ROOT / "notebooks" / "AGENTS.md": [
            "lab",
            "reader",
            "generated evidence",
            "not source of truth",
        ],
        REPO_ROOT / "docs" / "AGENTS.md": [
            "research_goal",
            "methodology",
            "finetuning ladder",
            "claim boundary",
            "finetuning_program_registry",
        ],
    }

    for path, phrases in required_agents.items():
        content = path.read_text()
        for phrase in phrases:
            assert phrase in content, f"{phrase!r} missing from {path}"
