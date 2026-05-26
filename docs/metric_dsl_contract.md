# Metric DSL Contract

The project needs to test whether a small model should learn raw SQL directly or
learn a semantic intermediate representation first. This contract defines the
first small version of that intermediate target.

The DSL preserves governed metrics until the semantic layer compiles them:

```text
MEASURE(revenue) BY customer_country
WHERE customer_country = 'FR'
ORDER BY MEASURE(revenue) DESC
LIMIT 5
```

That is not SQL. It is a typed intent representation with four separable claims:

- the requested measure is `revenue`;
- the grouping dimension is `customer_country`;
- the filter is a business-level country constraint;
- the metric should remain `MEASURE(revenue)` until the semantic model expands it.

The compiler expands that intent only after consulting a semantic model:

```sql
SELECT customers.country AS customer_country, SUM(orders.amount) AS revenue
FROM orders
JOIN customers ON orders.customer_id = customers.id
WHERE customers.country = 'FR'
GROUP BY customers.country
ORDER BY revenue DESC
LIMIT 5
```

## Why This Matters

Direct text-to-SQL fine-tuning asks the model to learn business meaning, schema
linking, metric formulas, joins, SQL syntax, and dialect rules in one output string.
That makes every miss look like "bad SQL."

A metric DSL creates a smaller set of targets:

| Target | Example score |
| --- | --- |
| Did the model preserve the governed measure? | `measure_preservation` |
| Did it pick the right measure? | `measure_f1` |
| Did it pick the right grouping dimensions? | `dimension_f1` |
| Did it preserve business-level filters? | `filter_f1` |
| Did the semantic compiler emit executable SQL? | SQL execution accuracy |

This lets the repo compare direct SQL SFT against DSL-first training. A model that
emits `SUM(orders.amount)` instead of `MEASURE(revenue)` may produce executable SQL,
but it bypasses the governed metric contract. That should be scored separately.

## Current Implementation

The implementation lives in `data.metric_dsl`:

- `parse_metric_query(...)` parses compact metric intent.
- `compile_metric_query(...)` expands intent through a semantic model.
- `score_metric_query(...)` scores measure preservation, measure F1, dimension F1,
  and filter F1.

The first finetuning-data surface lives in `data.metric_dsl_dataset`. It turns
the curated synthetic method fixtures into bootstrap chat-format training rows
for Stage 4:

```bash
uv run python -m data.metric_dsl_dataset
uv run python -m data.metric_dsl_direct_sql_dataset
```

Those commands write:

- `docs/data_artifacts/metric_dsl_training_rows.jsonl`
- `docs/data_artifacts/metric_dsl_training_rows_summary.json`
- `docs/data_artifacts/metric_dsl_training_rows.manifest.json`
- `docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl`
- `docs/data_artifacts/metric_dsl_direct_sql_training_rows_summary.json`
- `docs/data_artifacts/metric_dsl_direct_sql_training_rows.manifest.json`

The first artifact is the Stage 4 DSL target. The second is the same-row
direct-SQL control. Keeping both artifacts derived from the same synthetic
fixtures matters because the method comparison only means anything if the rows,
semantic model, and failure modes are matched.

These rows are intentionally small. They are a bootstrap contract for metric
DSL finetuning, not a sufficient dataset for a broad superiority claim.

Train them with explicit metadata checks:

```bash
uv run python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data docs/data_artifacts/metric_dsl_training_rows.jsonl \
  --eval-data docs/data_artifacts/metric_dsl_training_rows.jsonl \
  --expected-training-target metric_dsl \
  --expected-evaluation-mode metric_dsl \
  --expected-benchmark synthetic_metric_dsl_bootstrap \
  --run-id metric_dsl_bootstrap \
  --training-manifest-output results/train/metric_dsl_bootstrap.manifest.json

uv run python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl \
  --eval-data docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl \
  --expected-training-target direct_sql_control \
  --expected-evaluation-mode non_oracle_generation \
  --expected-benchmark metric_dsl_direct_sql \
  --run-id metric_dsl_direct_control \
  --training-manifest-output results/train/metric_dsl_direct_control.manifest.json
```

Those training manifests are not scorecards. Their job is simpler: prove which
prepared rows were used, which stage contract was enforced, and where the
resulting checkpoint lives before any eval manifest or comparison manifest is
trusted.

The evaluation runner lives in `eval.metric_dsl_eval`. It reads JSONL rows with
`generated_metric_dsl` or `predicted_dsl`, `reference_metric_dsl` or `gold_dsl`,
an inline `semantic_model`, and optional `reference_sql`/`database_path`.

```bash
uv run python -m eval.metric_dsl_eval \
  --input results/metric_dsl/<run-id>.predictions.jsonl \
  --output results/metric_dsl/<run-id>.jsonl \
  --manifest-output results/metric_dsl/<run-id>.manifest.json \
  --model-name <served-or-offline-model-name>
```

The output rows record parsed DSL, semantic intent scores, compiled SQL or
compile error, database-backed SQL execution scores when both `reference_sql`
and `database_path` are available, and semantic-model provenance. The manifest
aggregates:

- `metric_dsl_parse_rate`
- `metric_dsl_compile_rate`
- `compiled_sql_execution_attempt_rate`
- `compiled_sql_execution_evaluated_rows`
- `measure_f1`
- `dimension_f1`
- `filter_f1`
- `measure_preservation`
- `value_execution_accuracy`
- `strict_execution_accuracy`
- `semantic_model_sha256s`
- `semantic_model_sources`
- `semantic_model_oracle_derived_rows`

The parse gate requires the prediction to preserve at least one
`MEASURE(...)` token when the reference DSL uses a governed measure. A raw SQL
fragment such as `SUM(orders.amount) BY customer_country` can be reported as a
failed row, but it does not count as a parsed metric DSL and it is not compiled.
Rows whose semantic model is marked as oracle-derived make the manifest
diagnostic through `oracle_allowed=true`.

## Direct SQL Comparison Gate

A metric-DSL result by itself supports only a quality claim: the model can
produce a parseable intent representation, preserve governed measures, compile
through the semantic model, and execute the compiled SQL on database-backed rows.
It does not prove the DSL-first path is better than direct SQL.

To make that stronger claim, run a direct-SQL baseline on the same metric-heavy
rows and compare manifests:

```bash
uv run python -m eval.compare_metric_dsl_direct_sql \
  --metric-dsl-manifest results/metric_dsl/<run-id>.manifest.json \
  --direct-sql-manifest results/direct_sql/<run-id>.manifest.json \
  --output results/metric_dsl/<run-id>.compared.manifest.json
```

The direct manifest must use `benchmark=metric_dsl_direct_sql` and
`evaluation_mode=non_oracle_generation`. The comparer rejects oracle manifests,
oracle prompt markers in rows, row-count mismatches, row-identity mismatches,
unscored direct-SQL rows, and metric-DSL rows without database-backed compiled
SQL execution. The claim ledger then requires the compared manifest, the
referenced direct-SQL manifest, and a positive metric-DSL value delta before it
clears the `metric_dsl_beats_direct_sql` pending claim.

This is intentionally small. It is not a full semantic-layer compiler yet. Its job is
to create a runnable experiment surface for the next fine-tuning question:

> Is it better to train the local model to produce semantic intent first, then SQL?

## Offline Paired Runner

The repo now also has an offline Stage 4 comparison runner. It does not perform
checkpoint inference by itself. Instead, it takes the paired training manifests
for the Stage 4 metric-DSL run and the same-row direct-SQL control, validates
that they point at comparable prepared inputs, materializes explicit bootstrap
prediction rows when the checked-in artifacts still contain only labels, scores
both sides, and writes the comparison manifest:

```bash
uv run python -m eval.run_metric_dsl_comparison \
  --metric-training-manifest results/train/metric_dsl_bootstrap.manifest.json \
  --direct-training-manifest results/train/metric_dsl_direct_control.manifest.json \
  --output-dir results/metric_dsl_pairs \
  --run-id metric_dsl_bootstrap_pair \
  --model-name local-9b
```

This produces:

- `<run-id>.metric_dsl.jsonl`
- `<run-id>.metric_dsl.manifest.json`
- `<run-id>.direct_sql.jsonl`
- `<run-id>.direct_sql.manifest.json`
- `<run-id>.compared.manifest.json`

For the bootstrap contract, the runner projects:

- `gold_dsl -> predicted_dsl`
- `reference_sql -> generated_sql`

before scoring. That is deliberate. It gives the repo a measured compiler and
comparison sanity check before a real checkpoint exists, but it does not count
as model evidence.

The direct-SQL side is scored by `eval.direct_sql_eval`, which builds temporary
SQLite databases from the synthetic fixture pack when the rows are fixture-backed.
The metric-DSL side now uses the same fixture pack to attach database paths
before compiled SQL execution is scored. That keeps the Stage 4 pair runnable
before the repo has a generalized checkpoint-inference harness for the DSL path.

The current checked-in bootstrap artifact chain is:

- `docs/data_artifacts/metric_dsl_training_run.manifest.json`
- `docs/data_artifacts/metric_dsl_direct_sql_training_run.manifest.json`
- `docs/metric_dsl_comparison_preflight.json`
- `docs/result_manifests/metric_dsl_bootstrap_vs_direct_sql.json`

That comparison is intentionally mixed rather than flattering:

- value delta vs direct SQL: `+0.00`
- strict delta vs direct SQL: `-1.00`

So the bootstrap contract is now measured, but it does not support a
DSL-beats-SQL claim. The next stronger artifact must come from
`eval.run_local_metric_dsl_comparison`, where both sides are generated by actual
checkpoints instead of projected from labels.

## Local Checkpoint Loop

The repo now also has a local checkpoint runner for the Stage 4 pair. This is
the first path that takes actual adapters or checkpoints, generates outputs for
both sides, scores them, and writes the compared manifest:

```bash
uv run python -m eval.run_local_metric_dsl_comparison \
  --metric-training-manifest results/train/metric_dsl_bootstrap.manifest.json \
  --direct-training-manifest results/train/metric_dsl_direct_control.manifest.json \
  --output-dir results/metric_dsl_local \
  --run-id metric_dsl_local_pair \
  --model-name unsloth/Qwen3.5-9B \
  --metric-adapter-path outputs/metric_dsl/final \
  --direct-adapter-path outputs/direct_sql/final
```

Under the hood:

1. `eval.local_metric_dsl_benchmark` generates `generated_metric_dsl` rows for
   the DSL checkpoint and `generated_sql` rows for the direct-SQL checkpoint.
2. `eval.metric_dsl_eval` scores the DSL side.
3. `eval.direct_sql_eval` scores the direct-SQL side.
4. `eval.compare_metric_dsl_direct_sql` writes the same-row comparison manifest.

This keeps the experiment loop explicit: generation, scoring, and comparison are
separate artifacts, but they can now be run as one local command family.
