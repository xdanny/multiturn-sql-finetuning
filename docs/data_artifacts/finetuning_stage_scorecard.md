# Finetuning Stage Scorecard

This is the short companion to the finetuning ladder and the JSON registry.
Each stage answers the same practical questions: what the model learns,
what it is compared against, what benchmark surface it uses, and what
would count as a real win.

No wide summary table appears here on purpose.

## Stage 0: Direct SQL control

- Learns: direct SQL behavior with no planner or semantic shortcuts.
- Benchmark surfaces: prepared.
- Control arm: none; this is the control arm.
- Prepared artifacts: data.prepare, data/processed/eval_cosql_dev_100.jsonl.
- Comparison gate: eval.run_eval, eval.claim_ledger.
- Win condition: establish a stable non-oracle control arm on the fixed prepared slice.
- Leakage to forbid: no_future_turn_content, no_oracle_planner_hints, no_oracle_pruned_semantic_context.
- Current evidence: measured proxy results exist; best checked-in direct-SQL proxy value accuracy is 0.63.
- Claim boundary: Proxy-only control arm until later same-row comparisons or Stage 6 benchmarks beat hosted baselines.

## Stage 1: Planner supervision

- Learns: tables, columns, joins, projection shape, and duplicate policy before SQL generation.
- Benchmark surfaces: prepared.
- Control arm: direct_sql_control.
- Prepared artifacts: data.prepare, eval.planner_eval, gold_plan.
- Comparison gate: eval.planner_eval, eval.planner_optimize, eval.run_local_planner_eval.
- Win condition: planner-quality metrics improve before SQL generation is promoted.
- Leakage to forbid: no_future_turn_content, gold_plan_allowed_only_as_label, no_reference_sql_in_prompt.
- Current evidence: planner-quality evidence exists; current checked-in macro planner score is 0.571.
- Claim boundary: Planner quality is a prerequisite surface, not a SQL win by itself.

## Stage 2: Predicted-planner SQL

- Learns: SQL generation conditioned on a non-oracle predicted plan.
- Benchmark surfaces: prepared.
- Control arm: direct_sql_control.
- Prepared artifacts: eval.planner_eval, data/processed/eval_cosql_dev_predicted_planner_100.jsonl.
- Comparison gate: eval.run_predicted_planner_comparison, eval.run_local_predicted_planner_comparison.
- Win condition: same-row SQL execution beats the direct SQL control.
- Leakage to forbid: no_future_turn_content, predicted_plan_must_be_non_oracle, same_row_pairing_against_direct_sql.
- Current evidence: prepared artifacts exist, but pending claim because no predicted_planner result manifest.
- Next evidence: same-protocol endpoint SQL result manifest via `uv run python -m eval.run_predicted_planner_comparison`.
- Claim boundary: Only a positive same-row comparison against direct SQL can clear this method claim.

## Stage 3: Semantic-layer tuning

- Learns: governed entities, joins, grain, and value meaning that raw schema text misses.
- Benchmark surfaces: synthetic_semantic_layer.
- Control arm: direct_sql_control.
- Prepared artifacts: data.semantic_layer_dataset, data.semantic_proxy_dataset, docs/data_artifacts/value_index_cosql_dev_100.jsonl.
- Comparison gate: eval.run_local_semantic_layer_comparison, eval.run_local_semantic_proxy_comparison.
- Win condition: same-row comparison beats the direct SQL control without oracle pruning.
- Leakage to forbid: no_future_turn_content, no_reference_sql_in_prompt, no_expected_rows_in_prompt, no_gold_metric_dsl_in_prompt.
- Current evidence: prepared semantic artifacts exist, but no checked-in same-row semantic comparison result manifest yet.
- Next evidence: same-row semantic comparison result manifest via `uv run python -m eval.run_local_semantic_layer_comparison`.
- Claim boundary: Semantic artifacts only matter if same-row comparison beats the direct control without oracle pruning.

## Stage 4: MEASURE()-preserving metric DSL

- Learns: preserve governed metric intent before SQL compilation.
- Benchmark surfaces: synthetic_metric_dsl_bootstrap, metric_dsl_direct_sql.
- Control arm: direct_sql_control.
- Prepared artifacts: data.metric_dsl_dataset, data.metric_dsl_direct_sql_dataset, docs/data_artifacts/metric_dsl_training_rows.jsonl.
- Comparison gate: eval.metric_dsl_eval, eval.run_metric_dsl_comparison, eval.run_local_metric_dsl_comparison.
- Win condition: same-row compiled SQL beats the direct SQL control on metric-heavy rows.
- Leakage to forbid: no_future_turn_content, no_reference_sql_in_prompt, no_compiled_sql_in_prompt, measure_preservation_scored_separately_from_sql.
- Current evidence: prepared artifacts exist, but pending claim because no valid metric_dsl result manifest.
- Next evidence: compared metric_dsl manifest with direct-SQL baseline via `uv run python -m eval.run_local_metric_dsl_comparison`.
- Claim boundary: A DSL parse/compile win is not enough; compiled SQL must beat same-row direct SQL on metric-heavy rows.

## Stage 5: Generated-history recovery

- Learns: repair, retry, and recovery behavior after the model's own earlier SQL.
- Benchmark surfaces: synthetic_behavior_recovery, prepared.
- Control arm: direct_sql_control.
- Prepared artifacts: data.behavior_recovery_dataset, data.behavior_recovery_proxy_dataset, generated_history_trace.
- Comparison gate: eval.run_local_behavior_recovery_comparison, eval.run_local_rollout_comparison, eval.compare_rollout_history.
- Win condition: rollout comparison beats the same checkpoint under teacher-forced history.
- Leakage to forbid: no_future_turn_content, no_reference_sql_in_prompt, no_repair_labels_in_prompt, generated_history_claims_require_rollout_eval.
- Current evidence: prepared artifacts exist, but pending claim because no model-generated-history rollout manifest.
- Next evidence: same-model rollout and teacher-forced comparison manifests via `uv run python -m eval.run_local_rollout_comparison`.
- Claim boundary: Teacher-forced history cannot support a recovery claim; rollout comparison is required.

## Stage 6: Hosted and BIRD-Interact comparison

- Learns: none; this stage validates whether the best earlier local method transfers to real benchmark gates.
- Benchmark surfaces: prepared, bird_interact_transfer.
- Control arm: best_local_candidate_from_stage_0_to_5.
- Prepared artifacts: data.hosted_baseline_dataset, data.bird_interact_transfer_dataset.
- Comparison gate: eval.run_hosted_baseline_comparison, eval.run_bird_interact_comparison.
- Win condition: the best local candidate holds up against hosted or BIRD-Interact baselines on frozen contracts.
- Leakage to forbid: no_future_turn_content, non_oracle_generation_only, result_manifests_must_match_frozen_input_contract_hash.
- Current evidence: prepared artifacts exist, but pending claim because no same-protocol hosted-model manifest.
- Next evidence: hosted-model and BIRD-Interact result manifests on frozen contracts via `uv run python -m eval.run_hosted_baseline_comparison`.
- Claim boundary: This is the only stage that can support local-vs-hosted or BIRD-Interact competitiveness language.
