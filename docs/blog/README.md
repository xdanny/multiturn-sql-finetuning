# Blog Lab And Notebook Map

The shareable companion lab for the public post is:

```bash
marimo edit notebooks/labs/local_multiturn_sql_lab.py
```

That notebook is intentionally small. It runs an in-memory SQLite multi-turn SQL
experiment and compares four training targets: direct SQL, planner-first SQL,
semantic value grounding before SQL, and a `MEASURE()`-preserving DSL before SQL.
It defaults to CPU while detecting CUDA, MPS, or XPU when PyTorch can see them.
It does not require the full GPU training setup or a model download.

Each blog chapter has a matching marimo notebook. Run a notebook from the repository
root with:

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
