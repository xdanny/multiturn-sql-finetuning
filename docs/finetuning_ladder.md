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
  planner-quality movement before SQL generation is promoted.
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
  on the same rows.
- Claim boundary:
  only a same rows comparison with a positive value delta can clear the planner
  method claim.

## Stage 3: Semantic-layer tuning

- Training target:
  governed entities, value aliases, grain, dimensions, joins, and measures
  represented as semantic state instead of buried in one SQL string.
- Prepared data:
  non-oracle semantic artifacts such as the value index, semantic retrieval
  context, and entity-resolution labels that do not leak reference SQL.
- Trainer invocation:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data <semantic-train.jsonl> --eval-data <semantic-eval.jsonl>`
- Evaluation gate:
  value/entity retrieval artifacts must improve metric and join behavior on the
  same evaluation rows before SQL gains are called causal.
- Claim boundary:
  semantic prompt growth alone is not enough. The semantic path has to justify
  itself with artifact quality and same-row SQL outcomes.

## Stage 4: MEASURE()-preserving metric DSL

- Training target:
  a DSL that preserves governed `MEASURE()` intent until compilation.
- Prepared data:
  metric-heavy rows with `reference_metric_dsl` or `gold_dsl`, semantic model
  context, and optional database-backed execution targets. The first bootstrap
  surface is `uv run python -m data.metric_dsl_dataset`, which derives
  finetuning rows from the curated synthetic fixtures.
- Trainer invocation:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data docs/data_artifacts/metric_dsl_training_rows.jsonl --eval-data docs/data_artifacts/metric_dsl_training_rows.jsonl`
- Evaluation gate:
  `uv run python -m eval.metric_dsl_eval` followed by
  `uv run python -m eval.compare_metric_dsl_direct_sql`.
- Claim boundary:
  a parseable DSL manifest is only a quality claim. The method wins only if the
  compiled SQL beats the direct SQL baseline on matching metric-heavy rows.

## Stage 5: Generated-history recovery

- Training target:
  model behavior after its own earlier SQL, empty results, and repair steps.
- Prepared data:
  rollout-style rows or synthetic recovery fixtures that encode the difference
  between teacher-forced clean history and model-generated history.
- Trainer invocation:
  `uv run python -m train.finetune --config configs/qwen35_9b_5090.yaml --data <recovery-train.jsonl> --eval-data <recovery-eval.jsonl>`
- Evaluation gate:
  `uv run python -m eval.rollout_eval` followed by
  `uv run python -m eval.compare_rollout_history`.
- Claim boundary:
  recovery is not proven on teacher-forced rows. The same-model rollout must
  beat the teacher-forced baseline.

## Stage 6: Hosted and BIRD-Interact comparison

- Training target:
  none yet. This stage is the benchmark gate for whichever earlier training path
  becomes the strongest local candidate.
- Prepared data:
  fixed hosted-baseline rows, local rows, and BIRD-Interact-style transfer rows
  with shared scoring and manifest metadata.
- Trainer invocation:
  use the best earlier finetuned candidate from Stages 0 through 5.
- Evaluation gate:
  `uv run python -m eval.compare_hosted_baseline` plus a BIRD-Interact transfer
  manifest with the same oracle policy, latency accounting, and cost accounting.
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
