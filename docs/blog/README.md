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

The post should use the lab as the code attachment and generated evidence assets
for larger endpoint, planner, and claim-ledger results. It should not turn local
setup, vLLM serving, or
environment notes into the public reader path.

Publishable evidence assets are generated from the same support loaders:

```bash
python -m notebooks.blog_support --output-dir docs/blog/generated
```

Tracked outputs:

- `docs/blog/generated/accuracy-ladder.svg`
- `docs/blog/generated/planner-baseline.svg`
- `docs/blog/generated/claim-table.md`
- `docs/blog/generated/dataset-role-matrix.md`
- `docs/blog/generated/metric-dsl-contract.md`
- `docs/blog/generated/shareable-lab.md`
- `docs/blog/generated/lab-reader-flow.md`
- `docs/blog/generated/method-decision-rules.md`
- `docs/blog/generated/method-priority-backlog.md`
- `docs/blog/generated/lab-method-scores.md`
- `docs/blog/generated/lab-failure-trace.md`
- `docs/blog/generated/data-engineering-gates.md`
- `docs/blog/generated/synthetic-method-fixtures.md`
- `docs/blog/generated/value-grounding-labels.md`
- `docs/blog/generated/value-index.md`
- `docs/blog/generated/planner-readiness.md`
- `docs/blog/generated/data-artifact-contract.md`
- `docs/blog/generated/prompt-optimization-findings.md`
- `docs/blog/generated/target-comparison.md`
- `docs/blog/generated/target-evidence-matrix.md`
- `docs/blog/generated/endpoint-run-scorecard.md`
- `docs/blog/generated/failure-taxonomy-delta.md`
- `docs/blog/generated/schema-validation-findings.md`
- `docs/blog/generated/gpu-finetuning-evidence.md`
- `docs/blog/generated/manifest.json`

The intended pattern is:

1. The post frames a claim.
2. The matching lab section runs the smallest executable version of the claim or
   diagnostic.
3. Generated evidence assets render the larger endpoint and planner results.
4. The post states what the artifact proves and what it does not prove.
