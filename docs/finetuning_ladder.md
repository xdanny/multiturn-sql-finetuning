# Finetuning Ladder

This document turns the repo's research question into a structured finetuning
program. The purpose is not to list every possible experiment. The purpose is to
name the next trainable target, the data it needs, the evaluation gate it must
clear, and the claim boundary that keeps the result honest.

Every stage below should be treated as a same-repo method comparison, not a
free-form prompt hunt. If a stage cannot produce a prepared artifact, a trainer
run, and a comparison manifest, it is not ready to be called a finetuning step.

## Stage 0: Direct SQL control

- Training target:
  chat-format SQL rows with no oracle planner hints and no oracle-pruned
  semantic context.
- Prepared data:
  `data.prepare` output with `evaluation_mode=non_oracle_generation`.
- Trainer invocation:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data <train.jsonl> --eval-data <eval.jsonl>`
- Evaluation gate:
  `uv run python -m eval.run_eval` on the fixed CoSQL proxy slice, then a manifest in
  the claim ledger.
- Claim boundary:
  this is the control arm. It proves local proxy movement only. It does not
  support a hosted or BIRD-Interact win by itself.

## Stage 1: Planner supervision

- Training target:
  planner labels for relevant tables, relevant columns, query skeleton,
  projection shape, grouping, and duplicate policy.
- Prepared data:
  planner-prepared JSONL with `gold_plan` labels used as training or scoring
  targets, never as non-oracle prompt input.
- Trainer invocation:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data <planner-train.jsonl> --eval-data <planner-eval.jsonl>`
- Evaluation gate:
  `uv run python -m eval.planner_eval` and `uv run python -m eval.planner_optimize` must show
  planner-quality movement before SQL generation is promoted. For local
  checkpoint experiments, the repo now also has
  `uv run python -m eval.run_local_planner_eval`, which generates planner JSON
  locally, scores it with `planner_eval`, and can write predicted prepared JSON
  for Stage 2.
- Claim boundary:
  planner quality is not SQL quality. A better planner score alone does not beat
  the direct SQL control.

## Stage 2: Predicted-planner SQL

- Training target:
  SQL generation conditioned on a non-oracle `predicted_plan`.
- Prepared data:
  `predicted_planner` records written by `eval.planner_eval` after planner
  prediction and normalization.
- Trainer invocation:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data <predicted-planner-train.jsonl> --eval-data <predicted-planner-eval.jsonl>`
- Evaluation gate:
  `uv run python -m eval.run_predicted_planner_comparison` against the direct control
  on the same rows. For local checkpoint experiments, the repo also has
  `uv run python -m eval.run_local_predicted_planner_comparison`, which reuses
  the paired prepared inputs, writes local result manifests for both sides, and
  then writes the comparison manifest.
- Claim boundary:
  only a same rows comparison with a positive value delta can clear the planner
  method claim.

## Stage 3: Semantic-layer tuning

- Training target:
  governed entities, value aliases, grain, dimensions, joins, and measures
  represented as semantic state instead of buried in one SQL string.
- Prepared data:
  non-oracle semantic artifacts such as the value index, semantic retrieval
  context, and entity-resolution labels that do not leak reference SQL. The
  first runnable finetuning surface is
  `uv run python -m data.semantic_layer_dataset`, which derives semantic-aware
  SQL rows from the curated synthetic fixtures without copying scorer-only
  fields such as `expected_rows`, `reference_sql`, or gold metric DSL strings
  into the user prompt.
  The first proxy package on real CoSQL rows is
  `uv run python -m data.semantic_proxy_dataset`, which packages the fixed
  semantic training and eval slices together with the non-oracle value-label and
  value-index summaries. Its direct control companion is
  `uv run python -m data.semantic_proxy_direct_sql_dataset`.
- Trainer invocation:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data docs/data_artifacts/semantic_layer_training_rows.jsonl --eval-data docs/data_artifacts/semantic_layer_training_rows.jsonl --expected-training-target semantic_layer --expected-evaluation-mode non_oracle_generation --expected-benchmark synthetic_semantic_layer`
- Training artifact:
  each run should emit a training manifest for the exact semantic rows it saw,
  for example with `--run-id semantic_layer_bootstrap --training-manifest-output results/train/semantic_layer_bootstrap.manifest.json`.
- Evaluation gate:
  value/entity retrieval artifacts must improve metric and join behavior on the
  same evaluation rows before SQL gains are called causal. The local checkpoint
  loop is `uv run python -m eval.run_local_semantic_layer_comparison`, which
  pairs a `semantic_layer` manifest with a row-matched
  `synthetic_semantic_layer_direct_sql` control and writes the compared
  manifest only after both sides are scored on the same fixture rows. This
  runner now uses the shared `eval.local_sql_pair` contract rather than a
  stage-specific local workflow.
  The corresponding proxy loop on prepared CoSQL rows is
  `uv run python -m eval.run_local_semantic_proxy_comparison`, which reuses the
  packaged semantic/direct eval slices and compares the resulting prepared
  manifests on the same dialog turns.
- Claim boundary:
  semantic prompt growth alone is not enough. The semantic path has to justify
  itself with artifact quality and same-row SQL outcomes. The direct control
  for that comparison should be trained with
  `docs/data_artifacts/semantic_layer_direct_sql_training_rows.jsonl`, which
  keeps the same synthetic rows and SQL targets but omits semantic model
  context so the comparison isolates the semantic-layer signal itself.
  On the CoSQL proxy package, the semantic training pack is filtered to rows
  that actually contain semantic context; mixed source files are not allowed to
  silently blur the stage boundary.

## Stage 4: MEASURE()-preserving metric DSL

- Training target:
  a DSL that preserves governed `MEASURE()` intent until compilation.
- Prepared data:
  metric-heavy rows with `reference_metric_dsl` or `gold_dsl`, semantic model
  context, and optional database-backed execution targets. The first bootstrap
  surface is `uv run python -m data.metric_dsl_dataset`, which derives
  finetuning rows from the curated synthetic fixtures.
- Trainer invocation:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data docs/data_artifacts/metric_dsl_training_rows.jsonl --eval-data docs/data_artifacts/metric_dsl_training_rows.jsonl --expected-training-target metric_dsl --expected-evaluation-mode metric_dsl --expected-benchmark synthetic_metric_dsl_bootstrap`
- Training artifact:
  each run should emit a training manifest with the prepared-data hashes,
  expected metadata contract, output directory, and final checkpoint path, for
  example via `--run-id metric_dsl_bootstrap --training-manifest-output results/train/metric_dsl_bootstrap.manifest.json`.
- Evaluation gate:
  `uv run python -m eval.metric_dsl_eval` followed by
  `uv run python -m eval.compare_metric_dsl_direct_sql`. The repo also has
  `uv run python -m eval.run_metric_dsl_comparison` to validate the paired
  training manifests, score both offline sides, and write the comparison
  manifest in one path, plus `uv run python -m eval.run_local_metric_dsl_comparison`
  for actual local checkpoint generation on the Stage 4 pair.
- Claim boundary:
  a parseable DSL manifest is only a quality claim. The method wins only if the
  compiled SQL beats the direct SQL baseline on matching metric-heavy rows.
  The direct control for that comparison should be trained on
  `docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl` with
  `--expected-training-target direct_sql_control`,
  `--expected-evaluation-mode non_oracle_generation`, and
  `--expected-benchmark metric_dsl_direct_sql`.

## Stage 5: Generated-history recovery

- Training target:
  model behavior after its own earlier SQL, empty results, and repair steps.
- Prepared data:
  the first checked-in gate is the synthetic recovery pair:
  `data.behavior_recovery_dataset` and
  `data.behavior_recovery_direct_sql_dataset`. Those rows keep prior SQL and
  observed empty rows visible, but they do not copy scorer-only repair labels
  into the prompt. The prepared-dialog proxy package is
  `data.behavior_recovery_proxy_dataset`, which packages the fixed semantic
  CoSQL slice as `benchmark=prepared` with `training_target=behavior_recovery`
  so the same checkpoint can be compared under teacher-forced and
  generated-history evaluation.
- Trainer invocation:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data docs/data_artifacts/behavior_recovery_training_rows.jsonl --eval-data docs/data_artifacts/behavior_recovery_training_rows.jsonl --expected-training-target behavior_recovery --expected-evaluation-mode non_oracle_generation --expected-benchmark synthetic_behavior_recovery`
  for the synthetic gate, or:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data docs/data_artifacts/behavior_recovery_proxy_train.jsonl --eval-data docs/data_artifacts/behavior_recovery_proxy_eval.jsonl --expected-training-target behavior_recovery --expected-evaluation-mode non_oracle_generation --expected-benchmark prepared`
  for the prepared-dialog proxy gate.
- Evaluation gate:
  first, `uv run python -m eval.run_local_behavior_recovery_comparison` for the
  synthetic same-row pair. Then, `uv run python -m eval.run_local_rollout_comparison`
  for the same-checkpoint teacher-forced vs generated-history comparison on
  prepared dialogs. The endpoint path remains `uv run python -m eval.rollout_eval`
  followed by `uv run python -m eval.compare_rollout_history`.
  The synthetic pair now also uses the shared `eval.local_sql_pair` contract so
  recovery tuning follows the same local method-vs-control structure as
  semantic-layer tuning.
- Claim boundary:
  the synthetic pair can justify a narrow recovery-method comparison only. A
  behavior/recovery claim on the CoSQL proxy still requires same-model rollout
  to beat the teacher-forced baseline.

## Stage 6: Hosted and BIRD-Interact comparison

- Training target:
  none yet. This stage is the benchmark gate for whichever earlier training path
  becomes the strongest local candidate.
- Prepared data:
  the first checked-in input contract is `data.hosted_baseline_dataset`, which
  freezes the non-oracle prepared CoSQL proxy rows for same-protocol hosted
  comparison. The next checked-in transfer gate is
  `data.bird_interact_transfer_dataset`, which freezes a BIRD-Interact-style
  benchmark contract under a distinct `bird_interact_transfer` benchmark name.
- Trainer invocation:
  use the best earlier finetuned candidate from Stages 0 through 5.
- Evaluation gate:
  `uv run python -m eval.run_hosted_baseline_comparison` validates the local
  candidate manifest and then writes the same compared local-vs-hosted manifest
  through `eval.compare_hosted_baseline`. After that, a BIRD-Interact transfer
  run goes through `uv run python -m eval.run_bird_interact_comparison`, which
  requires both manifests to already declare a `bird_interact` benchmark plus
  the same oracle policy, latency accounting, cost accounting, and frozen-input
  contract hash.
- Claim boundary:
  this is the only stage that can support “local model competes with hosted
  SOTA” language.

## Common rules

- Training target:
  every run should name which stage it belongs to and which control arm it is
  expected to beat.
- Prepared data:
  keep oracle and non-oracle data split explicitly. Do not hide oracle planner
  hints inside renamed fields or copied prompt text.
- Trainer invocation:
  `uv run python -m train.finetune` should reject oracle diagnostic rows unless
  `--allow-oracle-diagnostic-data` is passed for an explicitly diagnostic run.
- Evaluation gate:
  manifests, comparisons, and claim-ledger rows are required. Notebook prose is
  not evidence.
- Claim boundary:
  same rows, same scorer, same oracle boundary, and explicit benchmark naming
  are mandatory before calling one finetuning step better than another.
