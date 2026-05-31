# Finetuning Method Runbook

This runbook explains how each research idea becomes a concrete finetuning or
evaluation comparison.

Use this when deciding what to run next. A method is not ranked because it has a
nice story or a completed smoke run. It becomes rankable only after the right
control has been evaluated on the same rows and the comparison manifest shows
the primary metric moved in the right direction.

## Shared Contract

Every method step needs the same minimum shape:

- **Hypothesis**: what behavior should the model learn?
- **Rows**: which canonical input rows define the method and its control?
- **Control**: what direct-SQL or teacher-forced baseline must it beat?
- **Primary metric**: usually value-only execution accuracy for SQL outcomes,
  with method-specific supporting metrics.
- **Leakage boundary**: which fields are scorer-only and must not enter prompts?
- **Evidence artifact**: which manifest or comparison artifact makes the result
  interpretable?

Runtime outputs belong under `results/` or `outputs/experiments/`. Checked-in
files under `docs/data_artifacts/` should be small canonical inputs, summaries,
or manifests that another developer can regenerate and inspect.

The durable checklist is:

- keep train, validation, proxy, and test split roles explicit;
- keep reference SQL, expected rows, future turns, gold plans, gold DSL, and
  repair labels out of production-style prompts;
- compare method and control on the same row identities;
- retain run manifests for scored generations and comparisons.

Use this checklist to keep smoke runs, benchmark proxies, synthetic fixtures,
and hosted/BIRD-style transfer claims separate. If the comparison artifact is
missing, the method may be wired, but it is not a supported win.

## Direct SQL SFT

Hypothesis: ordinary supervised SQL chat finetuning is the control arm for the
program.

Implementation path: train on prepared chat-format rows such as
`data/processed/train_smoke.jsonl` for smoke checks, then larger non-oracle
prepared rows for proxy runs.

Control: none. This is the baseline other methods must beat.

Evidence artifact: `eval.run_eval` or `eval.local_benchmark` writes a result
manifest with model, input hash, output hash, endpoint or local runner, and
value/strict/syntax metrics.

Next useful movement: keep this arm boring and stable. Do not add method-specific
context here unless it is also present in the method control.

## Planner Or DSL First, SQL Second

Hypothesis: SQL generation improves if the model first predicts a compact plan:
relevant tables, columns, joins, projection shape, grouping, duplicate policy,
and value/entity hints.

Implementation path: improve planner predictions before spending endpoint time on
SQL generation. Score planner outputs with `eval.planner_eval` and
`eval.planner_readiness`.

Control: direct SQL on the same row identities, same model, same scorer, and
same database root.

Evidence artifact: `eval.run_predicted_planner_comparison` produces direct-SQL,
predicted-planner, and comparison manifests. The comparison must show a positive
value-accuracy delta before `predicted_planner_sql_execution` can clear.

Latest movement: train-split planner-SFT rows from `data.planner_sft` produced a
1000-step planner adapter that passed bounded proxy and clean-holdout planner
readiness, but the 24-turn clean-holdout same-row SQL pair regressed by
`-0.0417` value accuracy versus the direct-SQL control.

Next useful movement: analyze why the predicted plans did not help downstream
SQL generation, then improve the non-oracle planner path before running a
broader endpoint pair.

## Semantic-Layer Tuning

Hypothesis: governed semantic context helps the model ground entities,
dimensions, measures, joins, grain, and values that raw DDL does not explain.

Implementation path: train or prompt with semantic context that is available at
inference time. `data.semantic_value_retrieval_inputs` turns the database-derived
value index into prepared prompt context by matching aliases against user text
seen up to each turn. It does not retrieve from reference SQL, assistant SQL,
expected rows, or future turns.

Control: same rows without the semantic-layer context, or the same model under a
matching direct-SQL prompt.

Evidence artifact: `eval.run_semantic_value_retrieval_comparison` runs the direct
SQL control and semantic value-retrieval arm as one endpoint pair, annotates the
semantic manifest with the database-derived value-index SHA, and calls
`eval.compare_semantic_value_retrieval`. The comparison verifies matching row
identities, non-oracle output rows, execution scores, and value-only and strict
deltas against direct SQL. Semantic prompt gains on the CoSQL proxy are useful,
but they are not a hosted or BIRD-Interact claim.

Preflight outputs from this runner are run-specific and should be written under
`results/`. They prove input compatibility only; they are not evidence that the
method improved SQL execution.

Next useful movement: generate the semantic prepared input on the fixed CoSQL
rows, run the paired endpoint experiment versus direct SQL, then inspect whether
the remaining misses are value lookup, entity resolution, join path, or query
shape failures.

## `MEASURE()`-Preserving Metric DSL

Hypothesis: for metric-heavy analysis, the model should preserve governed metric
intent as `MEASURE(name)` and let a compiler expand it to SQL.

Implementation path: use `docs/data_artifacts/metric_dsl_training_rows.jsonl` for
the DSL target and `metric_dsl_direct_sql_training_rows.jsonl` as the direct-SQL
control. Use `metric_dsl_prediction_inputs.jsonl` and
`metric_dsl_direct_sql_prediction_inputs.jsonl` only for generation-time
prediction prompts.

Control: direct SQL trained and evaluated on the same metric-heavy fixture
identities.

Primary metrics: DSL parse rate, compile rate, measure preservation,
measure/dimension/filter F1, and compiled-SQL value accuracy where database
execution is available.

Evidence artifact: `eval.run_metric_dsl_comparison` writes metric-DSL, direct-SQL,
and comparison manifests. A method win needs a positive compiled value delta and
must preserve `MEASURE(...)`; executable SQL alone is not enough.

Next useful movement: generate predictions from actual adapters for both arms
and run the paired comparison under `results/metric_dsl/`.

## Behavior And Recovery

Hypothesis: a useful multi-turn SQL model must continue after its own earlier
SQL, empty results, or bad value grounding. Clean teacher-forced history hides
this failure mode.

Implementation path: use `behavior_recovery_training_rows.jsonl` for the recovery
target and `behavior_recovery_direct_sql_training_rows.jsonl` as the same-fixture
control.

Control: direct SQL trained on the same generated-history repair fixture.
Teacher-forced history is diagnostic; it shows how much clean history hides
rollout failures, but it is not the recovery method's win condition.

Evidence artifact: `eval.run_behavior_recovery_comparison` writes rollout,
teacher-forced, and comparison manifests from one run id. That clears only the
generated-history diagnostic comparison. A recovery-tuning win still needs generated
predictions from the recovery adapter and the direct-SQL control adapter on the
same row identities, then a positive value delta under generated-history
rollout.

Next useful movement: run the paired recovery comparison for a served adapter,
then expand from the tiny synthetic repair row to CoSQL generated-history proxy
dialogs.

## Hosted And BIRD-Interact Comparison

Hypothesis: the final question is whether the best local 9B candidate can
compete with larger hosted systems under the same interactive data-analysis
protocol.

Implementation path: do not start here. Promote only a method that has already
cleared same-row proxy comparisons.

Control: hosted baseline manifests with the same inputs, scorer, latency, cost,
and model metadata.

Evidence artifact: `eval.compare_hosted_baseline` plus a frozen BIRD-Interact or
BIRD-style transfer manifest. CoSQL proxy movement alone cannot support this
claim.

Next useful movement: keep hosted and BIRD-Interact blockers explicit in the
roadmap until a same-protocol run exists.

## Adding A New Method Arm

Start with one small PR:

- Add or identify canonical rows.
- Add the direct control or explain why the control is teacher-forced.
- Add an evaluator or comparison runner only if existing runners do not cover
  the method.
- Update this runbook only when the method's operational contract changes.

Do not start by committing large generated outputs. Make the row identity,
leakage boundary, and comparison artifact clear first.
