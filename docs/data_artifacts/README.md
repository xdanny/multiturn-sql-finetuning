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
