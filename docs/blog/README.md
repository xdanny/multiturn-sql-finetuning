# Blog Lab And Notebook Map

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

The chapter notebooks below are internal evidence checkpoints for the repo, not the
reader-facing lab. Run one from the repository root only when editing the deeper
claim ledger or generated assets:

```bash
marimo edit notebooks/blog/01_problem_and_result.py
```

For non-interactive verification, marimo notebooks are plain Python files and can be
compiled with:

```bash
python -m py_compile notebooks/blog/01_problem_and_result.py
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
- `docs/blog/generated/target-comparison.md`
- `docs/blog/generated/endpoint-run-scorecard.md`
- `docs/blog/generated/manifest.json`

`tests/test_blog_notebooks.py` compares the checked-in generated files against a
fresh export, so the public blog cannot silently drift away from the current
claim ledger and planner artifacts.

| Chapter | Markdown | Notebook |
| --- | --- | --- |
| 01 | `docs/blog/01_problem_and_result.md` | `notebooks/blog/01_problem_and_result.py` |
| 02 | `docs/blog/02_wsl_5090_setup.md` | `notebooks/blog/02_wsl_5090_setup.py` |
| 03 | `docs/blog/03_data_and_eval.md` | `notebooks/blog/03_data_and_eval.py` |
| 04 | `docs/blog/04_training_iterations.md` | `notebooks/blog/04_training_iterations.py` |
| 05 | `docs/blog/05_vllm_blackwell_deep_dive.md` | `notebooks/blog/05_vllm_blackwell_deep_dive.py` |
| 06 | `docs/blog/06_data_engineering_for_multiturn_sql_eval.md` | `notebooks/blog/06_data_engineering_for_multiturn_sql_eval.py` |

The intended pattern is:

1. The post frames the question.
2. The notebook loads the current artifact.
3. The notebook renders the table or graph used in the post.
4. The post interprets what the artifact does and does not prove.

This keeps the series tied to evidence instead of disconnected prose.
The public site includes `shareable-lab.md` so readers can start with the lab
instead of reading a setup notebook or reconstructing the experiment from prose.
