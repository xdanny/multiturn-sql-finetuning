# Data Artifacts

This directory holds small, versioned inputs that make the finetuning program
reproducible. It is not meant to hold every output from every experiment.

The rule is:

- Commit canonical inputs that define a benchmark, fixture pack, or data
  contract.
- Commit a summary or manifest only when it explains provenance or supports a
  claim.
- Keep checkpoints, model generations, local comparison outputs, and ad hoc run
  manifests under `outputs/` or `results/`.

## Current Files

The current checked-in artifacts fall into three groups.

### Synthetic Method Fixtures

- `synthetic_method_fixtures.jsonl`
- `synthetic_method_fixtures_summary.json`
- `synthetic_method_fixtures.manifest.json`

These are tiny schema-rich rows used to isolate failure modes before spending
GPU or endpoint time. They cover value normalization, entity resolution,
grain/fanout, `MEASURE()` preservation, and recovery behavior.

The JSONL file is the useful input. The summary and manifest explain what was
generated and how to verify that the fixture file has not drifted.

### Metric DSL Finetuning Rows

- `metric_dsl_training_rows.jsonl`
- `metric_dsl_direct_sql_training_rows.jsonl`
- `metric_dsl_training_rows_summary.json`
- `metric_dsl_training_rows.manifest.json`

These rows project the metric-heavy synthetic fixtures into two same-fixture
training targets: metric DSL output and a direct SQL control. They are tiny on
purpose. Their job is to make the first DSL-vs-SQL finetuning smoke test
runnable before the project spends GPU time on a larger benchmark slice.

The model input contains schema, conversation, and semantic-model context. It
does not include reference SQL in the metric-DSL prompt. The labels are
synthetic curated answers, so use them for method smoke tests and controls, not
for claims about CoSQL, SParC, or BIRD performance.

### Metric DSL Prediction Inputs

- `metric_dsl_prediction_inputs.jsonl`
- `metric_dsl_direct_sql_prediction_inputs.jsonl`
- `metric_dsl_prediction_inputs_summary.json`
- `metric_dsl_prediction_inputs.manifest.json`

These rows are the paired generation inputs for the first metric-DSL comparison
gate. The prompt messages contain only schema, conversation, and semantic-model
context. The scorer fields, including reference SQL and gold DSL, are held out
from the prompt and carried only so generated outputs can be scored later by
`eval.run_metric_dsl_comparison`.

Use these files to generate same-row predictions for the metric-DSL adapter and
the direct-SQL control. The generated outputs still belong under `results/`, not
in this directory.

### Behavior Recovery Finetuning Rows

- `behavior_recovery_training_rows.jsonl`
- `behavior_recovery_direct_sql_training_rows.jsonl`
- `behavior_recovery_training_rows_summary.json`
- `behavior_recovery_training_rows.manifest.json`

These rows project the generated-history recovery fixture into two same-fixture
training targets: a recovery-specific repair prompt and a direct SQL control.
The recovery prompt includes the previous generated SQL and observed empty rows,
but not the repaired reference SQL, expected rows, or future turns.

Use these rows to smoke-test whether the training path can learn from generated
history and repair context. A behavior/recovery claim still needs generated
predictions evaluated under rollout, not teacher-forced history.

### Behavior Recovery Rollout Inputs

- `behavior_recovery_rollout_inputs.jsonl`
- `behavior_recovery_rollout_inputs_summary.json`
- `behavior_recovery_rollout_inputs.manifest.json`

These rows seed a tiny generated-history recovery dialog. The previous failed
SQL is visible as prior assistant history, while the repair SQL remains a
held-out assistant label for the final turn. Use this artifact to smoke the
rollout path before spending endpoint time on the larger CoSQL proxy slice.

### Value/Schema Repair Rollout Inputs

- `value_schema_repair_rollout_inputs.jsonl`
- `value_schema_repair_rollout_inputs_summary.json`
- `value_schema_repair_rollout_inputs.manifest.json`

These rows are the next diagnostic artifact after the failed behavior-recovery
arm. They keep the same generated-history repair fixture, but add two
inference-time artifacts that are allowed in production:

- a database-derived value index mapping display values such as `France` to
  stored values such as `FR`,
- schema validation guardrails listing allowed table columns and the invalid
  column pattern observed in the failed arm.

The repair SQL remains scorer-side. Use this artifact to test whether explicit
value/schema context changes the failure mode before training another adapter.

### Value-Choice Consistency Rollout Inputs

- `value_choice_consistency_rollout_inputs.jsonl`
- `value_choice_consistency_rollout_inputs_summary.json`
- `value_choice_consistency_rollout_inputs.manifest.json`

These rows narrow the failed value/schema repair prompt to one question: can the
model choose the storage value that matches the visible user mention? The prompt
shows a matched value-choice record derived from database contents and visible
text, including the `France` to `FR` candidate and a `US` distractor. The
expected storage value remains scorer-side so `eval.value_choice_consistency`
can score value choice separately from full SQL execution.

### Alias/Column-Validity Rollout Inputs

- `alias_column_validity_rollout_inputs.jsonl`
- `alias_column_validity_rollout_inputs_summary.json`
- `alias_column_validity_rollout_inputs.manifest.json`

These rows continue the same recovery fixture after value choice succeeds but
SQL execution still fails. They show column-role constraints from schema
introspection and the visible failed pattern, then let
`eval.alias_column_validity` score whether generated SQL uses only valid
table-column references and resolvable aliases. This keeps schema validity
separate from value choice and result matching.

### Alias/Column Context Prepared Inputs

- `alias_column_context_inputs_summary.json`
- `alias_column_context_inputs.manifest.json`

`data.alias_column_context_inputs` writes
`data/processed/eval_cosql_dev_100_alias_column_context.jsonl`, a fixed CoSQL
proxy-slice variant with schema-derived column-role constraints appended to the
first user turn. The context comes from SQLite schema introspection only:
allowed columns, primary keys, and foreign-key join keys. It does not use
reference SQL, gold plans, expected rows, assistant SQL, or future turns. Use it
for row-matched prompt comparisons before treating the single synthetic
alias/column pass as a scalable method.

### Value Grounding Labels

- `value_grounding_labels_cosql_dev_100.jsonl`
- `value_grounding_labels_cosql_dev_100_summary.json`
- `value_grounding_labels_cosql_dev_100.manifest.json`

These labels come from reference SQL on the fixed 100-turn CoSQL proxy slice.
They are useful for analysis and scorer-side supervision, but they are not
production prompt context. Treat them as diagnostic unless a run is explicitly
marked as oracle-supervised.

### Non-Oracle Value Index

- `value_index_cosql_dev_100.jsonl`
- `value_index_cosql_dev_100_summary.json`
- `value_index_cosql_dev_100.manifest.json`

This index comes from database contents, not from the answer SQL. It is the
right kind of artifact for retrieval experiments because it can be available at
inference time.

### Semantic Value-Retrieval Inputs

`data.semantic_value_retrieval_inputs` writes the prepared input used by
`eval.run_semantic_value_retrieval_comparison`. The default JSONL output belongs
under `data/processed/` because it is a generated benchmark input. The summary
and manifest may be written here to record provenance.

The builder defaults to matching value-index aliases against the current
user-authored turn only, capping retrieval volume at 4 matches per turn and
pruning short ambiguous aliases while retaining numeric aliases. It intentionally
does not use reference SQL, gold planner labels, expected rows, assistant SQL, or
future user turns for retrieval matching. History-scope retrieval is available
only for explicitly labeled diagnostics.

## What Should Be Added Here

Add files here when they are small, stable inputs that another developer should
be able to regenerate and inspect:

- fixed synthetic fixtures,
- small prepared benchmark slices,
- value/entity indexes,
- canonical training rows for a method comparison,
- manifests that prove the input file and generation command.

Prefer descriptive names:

- `<method>_training_rows.jsonl`
- `<method>_direct_sql_training_rows.jsonl`
- `<benchmark>_rows.jsonl`
- `<artifact>.manifest.json`
- `<artifact>_summary.json`

## What Should Not Be Added Here

Do not add:

- LoRA adapters,
- model checkpoints,
- endpoint generations,
- local benchmark outputs,
- exploratory comparison manifests,
- validate-only training-run manifests,
- one-off preflight files.

Those belong under `outputs/` or `results/`.

## Leakage Boundary

Every artifact should make its oracle boundary clear.

Allowed model inputs:

- user question,
- dialog history up to the current turn,
- schema,
- non-oracle semantic context,
- non-oracle value indexes.

Scorer-only or diagnostic fields:

- `reference_sql`,
- expected rows,
- gold planner labels,
- gold metric DSL,
- repair labels,
- future dialog turns.

Those fields may exist in an artifact for evaluation, but prompt builders must
not expose them as model input unless the run is explicitly labeled as an oracle
diagnostic.
