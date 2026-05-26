# Eval AGENTS

This subtree owns scoring, manifests, method comparison, readiness gates, and
claim enforcement.

Primary responsibilities:

- Write manifest-producing evaluation code before broadening any public claim.
- Keep each comparison explicit about benchmark, same rows pairing, oracle
  policy, and claim boundary.
- Reject incomplete or misleading comparisons even when a result looks good.

Rules:

- A manifest without a comparison is usually not enough for a method claim.
- For improvement claims, require same-model or same-method pairing exactly as
  the contract says; do not blur same-model and cross-method comparisons.
- Keep comparison artifacts keyed to claim-ledger rows.
- Prefer narrow, machine-checkable comparison outputs over prose summaries.
- When a method comparison depends on prior training-stage contracts, validate
  the training manifests before trusting the eval filenames.
- For local checkpoint experiments, keep the generation step separate from the
  scorer, then feed the scored manifests into the same comparison path used by
  offline or endpoint variants.

When editing here, inspect:

- `docs/research_goal.md`
- `docs/methodology.md`
- `docs/finetuning_ladder.md`
- `eval/result_manifest.py`
- `eval/compare_predicted_planner.py`
- `eval/compare_metric_dsl_direct_sql.py`
- `eval/run_metric_dsl_comparison.py`
- `eval/local_metric_dsl_benchmark.py`
- `eval/run_local_metric_dsl_comparison.py`
- `eval/compare_rollout_history.py`
