# Notebooks AGENTS

This subtree owns reader-facing lab surfaces.

Primary responsibilities:

- Explain repo evidence through a runnable lab.
- Keep the notebook and lab readable for a reader who is not running the full
  training stack.
- Reflect generated evidence and claim boundaries without becoming the source of
  truth.

Rules:

- The lab is for the reader. It should not require a GPU or local serving path
  just to understand the argument.
- Generated evidence may be displayed here, but notebooks are not source of truth
  for manifests, comparisons, or claims.
- Keep a clean separation between lab narrative, generated evidence, and
  training/eval internals.
- If a notebook changes a claim, the backing docs and manifests must change
  too.

When editing here, inspect:

- `notebooks/labs/local_multiturn_sql_lab.py`
- `notebooks/labs/local_multiturn_sql_lab_support.py`
- `docs/blog/README.md`
- `docs/blog/generated/`
