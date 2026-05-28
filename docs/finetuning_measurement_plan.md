# Finetuning Measurement Plan

This file separates three things that are easy to mix up:

- a smoke run proves the training target can load and produce an adapter;
- an evaluation run scores generated outputs;
- a benchmark claim compares the right method against the right control on the
  same rows.

Do not call a method better because its smoke run completed. A method becomes
evidence only after generated outputs are scored and compared against the
control arm listed here.

## Shared Rules

- Use the same input rows for a method and its control.
- Keep reference SQL, gold plans, gold metric DSL, expected rows, repair labels,
  and future turns out of model prompts unless the run is explicitly an oracle
  diagnostic.
- Put adapters and trainer byproducts under `outputs/experiments/`.
- Put generations, scored rows, manifests, and comparison manifests under
  `results/`.
- Prefer value-only execution accuracy for SQL outcome comparisons, but keep
  strict execution accuracy, syntax validity, and normalized match visible.
- Treat CoSQL as the local proxy slice. Do not describe CoSQL proxy gains as
  BIRD-Interact or hosted-SOTA wins.

## Direct SQL Control

Training input:

- `data/processed/train_smoke.jsonl` for smoke checks.
- larger prepared CoSQL rows when doing a real proxy run.

Primary measurement:

- value-only execution accuracy on the fixed non-oracle CoSQL proxy slice.

Supporting measurements:

- strict execution accuracy,
- syntax validity,
- normalized SQL match,
- latency and generation count.

Evidence gate:

- a result manifest from `eval.run_eval` or `eval.local_benchmark` that records
  input path, output path, model, endpoint, evaluation mode, command, and row
  count.

## Semantic Context

Training input:

- `data/processed/train_semantic_smoke.jsonl` for smoke checks.

Control:

- direct SQL on the same rows without extra semantic context, or the same model
  evaluated with the matching direct-SQL prompt.

Primary measurement:

- value-only execution accuracy delta against the same-row direct SQL control.

Supporting measurements:

- strict execution accuracy,
- syntax validity,
- value-grounding failure counts,
- join/path failure counts when classified.

Evidence gate:

- same-row result manifests for semantic-context and direct-SQL runs, plus a
  comparison artifact that records the delta and the prompt/evaluation policy.

## Metric DSL

Training inputs:

- `docs/data_artifacts/metric_dsl_training_rows.jsonl`
- `docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl`

Generation inputs:

- `docs/data_artifacts/metric_dsl_prediction_inputs.jsonl`
- `docs/data_artifacts/metric_dsl_direct_sql_prediction_inputs.jsonl`

Control:

- direct SQL trained and evaluated on the same metric-heavy fixtures.

Primary measurement:

- compiled metric-DSL value-only execution accuracy delta against direct SQL.

Supporting measurements:

- metric DSL parse rate,
- compile rate,
- `measure_preservation`,
- `measure_f1`,
- `dimension_f1`,
- `filter_f1`,
- strict execution accuracy for compiled SQL.

Evidence gate:

- generated prediction JSONL for both arms from
  `eval.generate_metric_dsl_predictions`,
- `eval.run_metric_dsl_comparison` output under `results/metric_dsl/`,
  containing the metric-DSL result manifest, the direct-SQL control manifest,
  and the same-row comparison manifest.

What it means:

- If parse or compile rate fails, the issue is not SQL execution yet.
- If parse and compile succeed but execution loses, the DSL preserved intent but
  did not improve the generated query path.
- If compiled DSL beats direct SQL while preserving `MEASURE(...)`, the method
  has a real proxy win on the metric-heavy slice.

## Behavior Recovery

Training inputs:

- `docs/data_artifacts/behavior_recovery_training_rows.jsonl`
- `docs/data_artifacts/behavior_recovery_direct_sql_training_rows.jsonl`

Rollout input:

- `docs/data_artifacts/behavior_recovery_rollout_inputs.jsonl`

Control:

- direct SQL trained on the same generated-history repair fixture.
- teacher-forced evaluation is useful as a diagnostic, but it is not the
  recovery win condition.

Primary measurement:

- generated-history rollout value-only execution accuracy delta.

Supporting measurements:

- strict rollout accuracy,
- syntax validity after a previous generated error,
- empty-result repair rate,
- value-normalization repair rate,
- whether later turns contain generated SQL history instead of reference SQL.

Evidence gate:

- generated-history rollout input from `data.behavior_recovery_rollout_inputs`,
- `eval.rollout_eval` manifest with
  `history_policy=model_generated_sql_rollout`,
- `eval.behavior_recovery_teacher_forced` manifest on the same input when
  checking rollout-vs-teacher-forced behavior,
- `eval.compare_rollout_history` comparison manifest.

What it means:

- A recovery adapter can load and train without proving recovery.
- Recovery only matters when a later turn sees prior generated SQL and observed
  results, then repairs the next query.

## Planner And Semantic-State Work

Training input:

- planner or semantic-state rows must expose only user-visible history, schema,
  and non-oracle artifacts.

Control:

- direct SQL generation without planner context, or a lexical/predicted-planner
  baseline on the same row identities.

Primary measurement:

- planner field quality before SQL generation, then value-only execution
  accuracy after SQL generation.

Supporting measurements:

- table F1,
- column F1,
- join-path match,
- projection-shape match,
- duplicate-row policy match,
- value/entity grounding accuracy.

Evidence gate:

- planner evaluation manifest before using the planner as SQL-generation
  context,
- predicted-planner SQL result manifest,
- same-row comparison against direct SQL.

## Minimum Run Record

Every real benchmark run should leave enough evidence for another developer to
answer these questions:

- Which adapter or endpoint produced the generations?
- Which exact input rows were used?
- Which prompt or method policy was used?
- Was the run non-oracle, oracle diagnostic, or synthetic curated supervision?
- Which control arm used the same rows?
- Which metric is primary for the claim?
- Where are the scored rows and manifests?
- Which comparison artifact proves the delta?

If any of those answers are missing, the run can still be useful exploration,
but it should not clear a claim in the ledger.

To see the current method readiness state from the claim ledger and checked-in
row artifacts, run:

```bash
uv run --active --no-sync python -m eval.method_readiness \
  --output results/method_readiness/<run-id>.json \
  --fail-on-missing-required
```

This report is a status summary. It does not replace scored generations or the
comparison manifests listed above.

The method arms in that report are loaded from `configs/finetuning_methods.yaml`.
When adding a new finetuning approach, put its training rows, control rows,
evaluator paths, next command, and claim boundary there first. Then update the
matching section in this plan only if the measurement rule itself changes.
