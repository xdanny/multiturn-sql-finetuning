# Blog Companion Lab

The public post has one reader-facing lab notebook:

```bash
marimo edit notebooks/labs/local_multiturn_sql_lab.py
```

That notebook is intentionally small. It runs an in-memory SQLite multi-turn SQL
experiment and compares five training targets: direct SQL, planner-first SQL,
semantic value grounding before SQL, a `MEASURE()`-preserving DSL before SQL,
and behavior/recovery tuning from failed execution feedback. It defaults to CPU
and has an `auto` runtime option that reports CUDA, MPS, or XPU availability when
PyTorch can see one. The SQLite lab computation remains CPU-safe. It does not
require the full GPU training setup or a model download.

For non-interactive verification, the marimo lab is a plain Python file and can be
compiled with:

```bash
python -m py_compile notebooks/labs/local_multiturn_sql_lab.py
```

Publishable evidence assets are generated from the same notebook support loaders:

```bash
python -m notebooks.blog_support --output-dir docs/blog/generated
```

Tracked outputs:

- `docs/blog/generated/accuracy-ladder.svg`
- `docs/blog/generated/planner-baseline.svg`
- `docs/blog/generated/claim-table.md`
- `docs/blog/generated/metric-dsl-contract.md`
- `docs/blog/generated/shareable-lab.md`
- `docs/blog/generated/lab-reader-flow.md`
- `docs/blog/generated/data-engineering-gates.md`
- `docs/blog/generated/target-comparison.md`
- `docs/blog/generated/endpoint-run-scorecard.md`
- `docs/blog/generated/manifest.json`

`tests/test_blog_notebooks.py` compares the checked-in generated files against a
fresh export, so the public blog cannot silently drift away from the current
claim ledger and planner artifacts.

The intended pattern is:

1. The post frames the question.
2. The companion lab runs the smallest executable version of the method comparison.
3. Generated evidence assets render the larger endpoint and planner results.
4. The post interprets what each artifact does and does not prove.

This keeps the series tied to evidence instead of disconnected prose.
The public site includes `shareable-lab.md` so readers can start with the lab
instead of reconstructing the experiment from prose.
