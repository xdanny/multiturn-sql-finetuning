# Repo AGENTS

This repo is a finetuning and evaluation workspace for multi-turn analytical
SQL. Keep changes oriented around measurable method comparisons, not isolated
demo artifacts.

Primary responsibilities:

- Preserve the research ladder from direct SQL control toward planner,
  semantic-layer, metric-DSL, recovery, and hosted/BIRD-style comparisons.
- Keep data, training, evaluation, and docs aligned on the same leakage policy.
- Prefer small PRs that land one coherent improvement at a time.
- Keep checked-in artifacts small and explainable; use `outputs/` and
  `results/` for run-specific byproducts.

Rules:

- Use `uv run ...` for Python commands in docs and local verification.
- Do not expose future turns, reference SQL, gold plans, gold metric DSL, repair
  labels, or expected rows to model prompts unless the run is explicitly labeled
  as an oracle diagnostic.
- Keep contract tests focused. Add tests for shared behavior or leakage
  prevention, not broad generated-file freshness checks.
- Before adding files under `docs/data_artifacts/`, read
  `docs/data_artifacts/README.md`.

When editing broadly, inspect:

- `README.md`
- `docs/research_goal.md`
- `docs/methodology.md`
- `docs/finetuning_ladder.md`
- `docs/finetuning_measurement_plan.md`
- `docs/data_artifacts/README.md`
- `docs/AGENTS.md`
- `configs/AGENTS.md`
- `train/AGENTS.md`
- `eval/AGENTS.md`
- `data/AGENTS.md`
- `tests/AGENTS.md`
- `notebooks/AGENTS.md`
- `scripts/AGENTS.md`
- `serve/AGENTS.md`
