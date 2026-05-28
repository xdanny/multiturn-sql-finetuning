# Eval AGENTS

This subtree owns local and endpoint evaluation, scoring, comparison manifests,
and claim ledger inputs.

Primary responsibilities:

- Compare methods on the same rows, same databases, and same oracle policy.
- Keep direct-SQL controls explicit for planner, semantic-layer, metric-DSL, and
  recovery experiments.
- Separate readiness checks from measured claims.
- Write run outputs to `results/` unless a file is a small canonical benchmark
  input or claim artifact.

Rules:

- A preflight proves inputs are comparable; it is not a model win.
- A method claim needs a same-row comparison against its control arm.
- Hosted or BIRD-style claims need frozen inputs, result manifests, and clear
  cost/model metadata.
- Keep new tests focused on scoring correctness, row matching, and leakage
  prevention. Do not add broad repo-shape tests for every new artifact.

When editing here, inspect:

- `eval/run_eval.py`
- `eval/local_benchmark.py`
- `eval/metric_dsl_eval.py`
- `eval/claim_ledger.py`
- `docs/evidence_contract.md`
- `docs/data_artifacts/README.md`
