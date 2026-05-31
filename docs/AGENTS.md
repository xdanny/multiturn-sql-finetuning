# Docs AGENTS

This subtree owns the research narrative, runbooks, evidence contracts, and
small checked-in documentation artifacts for the SQL finetuning program.

Primary responsibilities:

- Keep blog-derived ideas grounded in runnable repo steps: direct SQL controls,
  planner work, semantic context, metric DSL, behavior recovery, and hosted or
  BIRD-style comparisons.
- Separate smoke-run viability from scored evaluation and benchmark claims.
- Keep claim language human-readable, specific, and tied to visible artifacts.
- Make docs point to the exact code, data artifact, command, manifest, or
  comparison file they rely on.

Rules:

- Do not describe a method as better until the required comparison manifest
  exists and shows the right same-row delta.
- Do not turn temporary outputs into checked-in evidence. Runtime outputs belong
  under `outputs/` or `results/`; small canonical inputs belong under
  `docs/data_artifacts/`.
- Do not expose future turns, reference SQL, gold plans, gold metric DSL,
  expected rows, or repair labels as production-style model inputs.
- When adding a method doc, state the control arm, primary metric, supporting
  metrics, leakage boundary, and comparison artifact.
- Prefer short prose and compact bullet lists over large markdown tables.
- If a doc cites CoSQL, SParC, BIRD, or BIRD-Interact, explain the role of that
  benchmark in this repo instead of assuming the reader already knows it.
- Use `uv run ...` in documented Python commands.

When editing here, inspect:

- `docs/research_goal.md`
- `docs/methodology.md`
- `docs/evidence_contract.md`
- `docs/finetuning_ladder.md`
- `docs/finetuning_method_runbook.md`
- `docs/finetuning_smoke_matrix.md`
- `docs/finetuning_measurement_plan.md`
- `docs/data_artifacts/README.md`
- `README.md`
