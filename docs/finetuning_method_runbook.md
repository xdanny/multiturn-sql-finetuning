# Finetuning Method Runbook

This is the operational companion to `configs/finetuning_methods.yaml`. The
YAML file is the machine-readable source of truth; this file explains how each
blog-derived idea becomes a concrete finetuning or evaluation step.

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
- **Evidence gate**: which manifest or comparison artifact makes the result
  interpretable?

Runtime outputs belong under `results/` or `outputs/experiments/`. Checked-in
files under `docs/data_artifacts/` should be small canonical inputs, summaries,
or manifests that another developer can regenerate and inspect.

`configs/finetuning_methods.yaml` carries the machine-readable version of this
contract. Every method arm must define:

- `finetuning_objective`
- `benchmark_scope`
- `primary_metric`
- `leakage_boundary`
- `evidence_gate`

Those fields are emitted by `eval.method_readiness` so method status can be
reviewed without reading this runbook first. Keep them concrete: name the
behavior being trained, the benchmark rows or fixture family, the metric that
settles the comparison, the fields that must not enter prompts, and the artifact
that would clear the gate.

`configs/finetuning_steps.yaml` is the concrete sequence of train, evaluation,
and comparison commands. Validate it with:

```bash
uv run --active --no-sync python -m train.finetuning_steps
```

The step summary is not a benchmark result. It is a checklist for which row
artifacts, controls, protocols, and claim gates the next run must use.

Print the cheap preflight commands for one step before spending GPU or endpoint
time:

```bash
uv run --active --no-sync python -m train.finetuning_steps \
  --step-id metric_dsl_vs_direct_sql \
  --commands preflight
```

Use `--commands run` for the full training/evaluation commands and
`--commands all` when preparing a complete run checklist.

To hand a step to another process or save an execution checklist, add
`--output`:

```bash
uv run --active --no-sync python -m train.finetuning_steps \
  --step-id semantic_value_retrieval_pair \
  --commands preflight \
  --output results/step_records/<run-id>.semantic_preflight.json
```

That JSON record includes the selected commands, row readiness, protocol ids,
claim ids, evidence gate, and leakage boundary.

## Direct SQL SFT

Hypothesis: ordinary supervised SQL chat finetuning is the control arm for the
program.

Finetuning step: train on prepared chat-format rows such as
`data/processed/train_smoke.jsonl` for smoke checks, then larger non-oracle
prepared rows for proxy runs.

Control: none. This is the baseline other methods must beat.

Evidence gate: `eval.run_eval` or `eval.local_benchmark` writes a result
manifest with model, input hash, output hash, endpoint or local runner, and
value/strict/syntax metrics.

Next useful movement: keep this arm boring and stable. Do not add method-specific
context here unless it is also present in the method control.

## Planner Or DSL First, SQL Second

Hypothesis: SQL generation improves if the model first predicts a compact plan:
relevant tables, columns, joins, projection shape, grouping, duplicate policy,
and value/entity hints.

Finetuning step: improve planner predictions before spending endpoint time on
SQL generation. Score planner outputs with `eval.planner_eval` and
`eval.planner_readiness`.

Control: direct SQL on the same row identities, same model, same scorer, and
same database root.

Evidence gate: `eval.run_predicted_planner_comparison` produces direct-SQL,
predicted-planner, and comparison manifests. The comparison must show a positive
value-accuracy delta before `predicted_planner_sql_execution` can clear.

Next useful movement: replace weak lexical planner output with a non-oracle
planner that improves column linking and projection shape before running the
full SQL pair.

## Semantic-Layer Tuning

Hypothesis: governed semantic context helps the model ground entities,
dimensions, measures, joins, grain, and values that raw DDL does not explain.

Finetuning step: train or prompt with semantic context that is available at
inference time. `data.semantic_value_retrieval_inputs` turns the database-derived
value index into prepared prompt context by matching aliases against user text
seen up to each turn. It does not retrieve from reference SQL, assistant SQL,
expected rows, or future turns.

Control: same rows without the semantic-layer context, or the same model under a
matching direct-SQL prompt.

Evidence gate: `eval.run_semantic_value_retrieval_comparison` runs the direct
SQL control and semantic value-retrieval arm as one endpoint pair, annotates the
semantic manifest with the database-derived value-index SHA, and calls
`eval.compare_semantic_value_retrieval`. The comparison verifies matching row
identities, non-oracle output rows, execution scores, and value-only and strict
deltas against direct SQL. Semantic prompt gains on the CoSQL proxy are useful,
but they are not a hosted or BIRD-Interact claim.

Preflight: `docs/semantic_value_retrieval_comparison_preflight.json` proves the
current direct and semantic prepared inputs align before endpoint generation.
It is readiness evidence only.

Next useful movement: generate the semantic prepared input on the fixed CoSQL
rows, run the paired endpoint experiment versus direct SQL, then inspect whether
the remaining misses are value lookup, entity resolution, join path, or query
shape failures.

## `MEASURE()`-Preserving Metric DSL

Hypothesis: for metric-heavy analysis, the model should preserve governed metric
intent as `MEASURE(name)` and let a compiler expand it to SQL.

Finetuning step: use `docs/data_artifacts/metric_dsl_training_rows.jsonl` for
the DSL target and `metric_dsl_direct_sql_training_rows.jsonl` as the direct-SQL
control. Use `metric_dsl_prediction_inputs.jsonl` and
`metric_dsl_direct_sql_prediction_inputs.jsonl` only for generation-time
prediction prompts.

Control: direct SQL trained and evaluated on the same metric-heavy fixture
identities.

Primary metrics: DSL parse rate, compile rate, measure preservation,
measure/dimension/filter F1, and compiled-SQL value accuracy where database
execution is available.

Evidence gate: `eval.run_metric_dsl_comparison` writes metric-DSL, direct-SQL,
and comparison manifests. A method win needs a positive compiled value delta and
must preserve `MEASURE(...)`; executable SQL alone is not enough.

Next useful movement: generate predictions from actual adapters for both arms
and run the paired comparison under `results/metric_dsl/`.

## Behavior And Recovery

Hypothesis: a useful multi-turn SQL model must continue after its own earlier
SQL, empty results, or bad value grounding. Clean teacher-forced history hides
this failure mode.

Finetuning step: use `behavior_recovery_training_rows.jsonl` for the recovery
target and `behavior_recovery_direct_sql_training_rows.jsonl` as the same-fixture
control.

Control: direct SQL trained on the same generated-history repair fixture.
Teacher-forced history is diagnostic; it shows how much clean history hides
rollout failures, but it is not the recovery method's win condition.

Evidence gate: `eval.run_behavior_recovery_comparison` writes rollout,
teacher-forced, and comparison manifests from one run id. That clears only the
generated-history diagnostic gate. A recovery-tuning win still needs generated
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

Finetuning step: do not start here. Promote only a method that has already
cleared same-row proxy comparisons.

Control: hosted baseline manifests with the same inputs, scorer, latency, cost,
and model metadata.

Evidence gate: `eval.compare_hosted_baseline` plus a frozen BIRD-Interact or
BIRD-style transfer manifest. CoSQL proxy movement alone cannot support this
claim.

Next useful movement: keep hosted and BIRD-Interact blockers explicit in the
claim ledger until a same-protocol run exists.

## Adding A New Method Arm

Start with one small PR:

- Add or identify canonical rows.
- Add the direct control or explain why the control is teacher-forced.
- Add an evaluator or comparison runner only if existing runners do not cover
  the method.
- Update `configs/finetuning_methods.yaml`.
- Update this runbook only when the method's operational contract changes.
- Run `uv run --active --no-sync python -m eval.method_readiness
  --fail-on-missing-required`.

Do not start by committing large generated outputs. Make the row identity,
leakage boundary, and comparison artifact clear first.
