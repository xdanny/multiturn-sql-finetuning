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
- If a stage adds or changes a comparison gate, update
  `data.finetuning_program_registry` so the cross-stage contract stays current.
- Prefer narrow, machine-checkable comparison outputs over prose summaries.
- When a method comparison depends on prior training-stage contracts, validate
  the training manifests before trusting the eval filenames.
- For local checkpoint experiments, keep the generation step separate from the
  scorer, then feed the scored manifests into the same comparison path used by
  offline or endpoint variants.
- For SQL-generating method-vs-direct-control pairs, prefer
  `eval.local_sql_pair` plus a small method spec over writing a new local
  runner from scratch.
- For mixed-output method-vs-control pairs, prefer
  `eval.local_generation_pair` so the orchestration stays shared even when the
  method side uses a different scorer than direct SQL.
- For paired training-manifest metadata checks, prefer
  `eval.training_manifest_pair` instead of repeating stage, benchmark, and
  evaluation-mode validation inline.
- For same-row and non-empty paired-input checks, prefer
  `eval.pair_input_contract` instead of duplicating row-identity validation in
  each comparison runner.
- For paired local `run_local_benchmark` flows, prefer
  `eval.local_benchmark_pair` so SQL-vs-SQL benchmark orchestration stays
  shared across stages like predicted planner and semantic proxy.
- If a local comparison runner exists, it should reuse `eval.local_benchmark`
  or another manifest-writing generator rather than inventing a second result
  format.
- Planner-quality loops should stop at planner manifests and predicted prepared
  artifacts unless they explicitly continue into a separate SQL-generation
  stage.

When editing here, inspect:

- `docs/research_goal.md`
- `docs/methodology.md`
- `docs/finetuning_ladder.md`
- `data/finetuning_program_registry.py`
- `eval/result_manifest.py`
- `eval/compare_predicted_planner.py`
- `eval/compare_metric_dsl_direct_sql.py`
- `eval/compare_semantic_layer_direct_sql.py`
- `eval/compare_semantic_proxy_direct_sql.py`
- `eval/compare_behavior_recovery_direct_sql.py`
- `eval/local_generation_pair.py`
- `eval/local_benchmark_pair.py`
- `eval/local_sql_pair.py`
- `eval/pair_input_contract.py`
- `eval/training_manifest_pair.py`
- `eval/run_metric_dsl_comparison.py`
- `eval/local_metric_dsl_benchmark.py`
- `eval/local_text_benchmark.py`
- `eval/run_local_behavior_recovery_comparison.py`
- `eval/run_local_metric_dsl_comparison.py`
- `eval/run_local_semantic_layer_comparison.py`
- `eval/run_local_semantic_proxy_comparison.py`
- `eval/run_local_predicted_planner_comparison.py`
- `eval/local_rollout_benchmark.py`
- `eval/run_local_rollout_comparison.py`
- `eval/run_hosted_baseline_comparison.py`
- `eval/run_bird_interact_comparison.py`
- `eval/local_planner_benchmark.py`
- `eval/run_local_planner_eval.py`
- `eval/compare_rollout_history.py`
