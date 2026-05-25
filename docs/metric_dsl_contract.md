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

This is intentionally small. It is not a full semantic-layer compiler yet. Its job is
to create a runnable experiment surface for the next fine-tuning question:

> Is it better to train the local model to produce semantic intent first, then SQL?
