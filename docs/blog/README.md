# Blog-Attached Code Lab

The public post links to this repository as an attached codebase and to a
published HTML lab at `/labs/local-multiturn-sql-finetuning/`. The same lab can
be rerun from the source code with Marimo:

```bash
marimo edit notebooks/labs/local_multiturn_sql_lab.py
```

A portable Jupyter export is also available:

```bash
jupyter lab notebooks/labs/local_multiturn_sql_lab.ipynb
```

The lab is CPU-safe and auto-selects CUDA, MPS, or XPU
when PyTorch can detect an accelerator, then falls back to CPU. It compares direct SQL,
planner-first SQL, semantic-layer state, `MEASURE()`-preserving DSL, and
behavior/recovery tuning on a small four-turn scenario. It should stay a
portable lab, not a serving or dependency-installation guide.

The post should use the lab as the code attachment and cite run manifests or
small canonical data artifacts for any larger endpoint, planner, metric-DSL, or
rollout result. It should not turn local setup, vLLM serving, generated evidence
bundles, or environment notes into the public reader path.

The intended pattern is:

1. The post frames a claim.
2. The matching lab section runs the smallest executable version of the claim or
   diagnostic.
3. Run manifests or canonical data artifacts support larger endpoint and planner
   results.
4. The post states what the artifact proves and what it does not prove.
