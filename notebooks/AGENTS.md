# Notebooks AGENTS

This subtree owns shareable analysis notebooks. Notebooks are companion
material, not the source of truth for benchmark claims.

Primary responsibilities:

- Keep notebooks small, deterministic, and runnable without a live endpoint by
  default.
- Use notebooks to explain measured evidence, failure examples, and reproduction
  paths.
- Keep reusable logic in normal Python modules when it is part of the repo
  workflow.

Rules:

- Notebook-only results are not benchmark evidence.
- Do not expose reference SQL, expected rows, repair labels, or future turns as
  model inputs for validation, holdout, or hosted comparisons.
- If a notebook cites a result, point to the manifest or compact evidence
  artifact that supports it.
- Avoid rigid article scaffolds. Let blog prose and notebooks follow the
  research question and measured results.

When editing here, inspect:

- `README.md`
- `docs/research_roadmap.md`
- `docs/blog/README.md`
