# Eval AGENTS

This subtree owns local and endpoint evaluation, scoring, and comparison
manifests.

Primary responsibilities:

- Compare methods on the same rows, same databases, same scorer, and same
  prompt-visible inputs.
- Keep direct-SQL controls explicit for semantic context, Metric DSL, value
  retrieval, recovery, and hosted comparisons.
- Separate input compatibility checks from measured claims.
- Write run outputs to `results/` unless a file is a small canonical benchmark
  input or compact evidence artifact.

Rules:

- A method claim needs a same-row comparison against its control arm.
- Hosted or external benchmark claims need frozen inputs, result manifests, and
  clear cost/model metadata.
- Keep new tests focused on scoring correctness, row matching, and leakage
  prevention. Do not add broad repo-shape tests for every new artifact.

When editing here, inspect:

- `eval/run_eval.py`
- `eval/local_benchmark.py`
- `eval/metric_dsl_eval.py`
- `eval/run_semantic_value_retrieval_comparison.py`
- `eval/compare_hosted_baseline.py`
- `docs/research_roadmap.md`
- `docs/data_artifacts/README.md`
