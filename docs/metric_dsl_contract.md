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

The evaluation runner lives in `eval.metric_dsl_eval`. It reads JSONL rows with
`generated_metric_dsl` or `predicted_dsl`, `reference_metric_dsl` or `gold_dsl`,
an inline `semantic_model`, and optional `reference_sql`/`database_path`.

```bash
python -m eval.metric_dsl_eval \
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
python -m eval.compare_metric_dsl_direct_sql \
  --metric-dsl-manifest results/metric_dsl/<run-id>.manifest.json \
  --direct-sql-manifest results/direct_sql/<run-id>.manifest.json \
  --output results/metric_dsl/<run-id>.compared.manifest.json
```

The direct manifest must use `benchmark=metric_dsl_direct_sql` and
`evaluation_mode=non_oracle_generation`. The comparer rejects oracle manifests,
oracle prompt markers in rows, row-count mismatches, row-identity mismatches,
unscored direct-SQL rows, and metric-DSL rows without database-backed compiled
SQL execution. A method claim requires the compared manifest, the referenced
direct-SQL manifest, and a positive metric-DSL value delta.

This is intentionally small. It is not a full semantic-layer compiler yet. Its job is
to create a runnable experiment surface for the next fine-tuning question:

> Is it better to train the local model to produce semantic intent first, then SQL?
