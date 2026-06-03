# Docs AGENTS

This subtree owns the research narrative and small checked-in documentation
artifacts for the SQL fine-tuning program.

Primary responsibilities:

- Keep docs grounded in runnable repo steps: data preparation, direct SQL
  controls, semantic/value context, Metric DSL smoke tests, behavior recovery,
  and hosted comparisons.
- Keep claim language human-readable, specific, and tied to visible artifacts.
- Make `docs/research_roadmap.md` the canonical roadmap.

Rules:

- Do not describe a method as better until a same-row comparison supports it.
- Do not turn temporary outputs into checked-in evidence. Runtime outputs belong
  under `outputs/` or `results/`; small canonical inputs belong under
  `docs/data_artifacts/`.
- Do not expose future turns, reference SQL, expected rows, repair labels, or
  hidden evaluation answers as model inputs.
- Prefer short prose and compact bullet lists over large markdown tables.
- If a doc cites CoSQL, SParC, BIRD, BIRD-Interact, LiveSQLBench, or Spider 2.0,
  explain the role of that benchmark in this repo.
- Use `uv run ...` in documented Python commands.

When editing here, inspect:

- `README.md`
- `docs/research_roadmap.md`
- `docs/data_artifacts/README.md`
- `docs/blog/README.md`
