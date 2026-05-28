# Notebooks AGENTS

This subtree owns shareable labs and notebook support code. Notebooks are
companion material for a specific public explanation, not the source of truth
for benchmark claims.

Primary responsibilities:

- Keep labs small enough for readers to run on CPU by default, with accelerator
  use treated as optional.
- Make notebook examples explain method shape: direct SQL, planner/DSL first,
  semantic-layer state, `MEASURE()` preservation, and behavior recovery.
- Keep reusable logic in Python support modules instead of large notebook cells.
- Make published HTML labs discoverable without turning every old blog post
  into a notebook project.

Rules:

- Use notebook code to illustrate the repo contracts; do not let notebook-only
  results become benchmark evidence.
- Do not expose reference SQL, gold plans, gold metric DSL, expected rows,
  repair labels, or future turns as production-style model inputs.
- Prefer static, deterministic examples over live endpoint calls.
- If a lab cites a result, point to the manifest, artifact, or generated evidence
  file that supports it.
- Keep notebook exports synchronized only when the source notebook intentionally
  changes. Avoid regenerating unrelated notebooks in the same PR.

When editing here, inspect:

- `notebooks/labs/local_multiturn_sql_lab.py`
- `notebooks/labs/local_multiturn_sql_lab_support.py`
- `docs/blog/README.md`
- `docs/finetuning_measurement_plan.md`
- `docs/evidence_contract.md`
