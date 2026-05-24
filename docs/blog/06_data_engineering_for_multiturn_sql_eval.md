# Data Engineering Problems Behind Multi-Turn SQL Evals

The easy version of a text-to-SQL benchmark is a JSON file of questions, a
database, and a scoring script. The hard version is everything required to make
the score mean something.

Multi-turn SQL makes that harder because the benchmark is no longer just a
mapping from one question to one query. It is a mapping from conversation state,
database state, semantic state, and user intent to executable SQL.

This is where data engineering becomes central. The next quality gains in this
repo are likely to come from better metadata, fixtures, and eval design rather
than from another small training command.

The latest run makes that concrete. A plain semantic adapter with a minimal
executable prompt reached `0.640` value accuracy on the fixed 100-turn CoSQL
slice. Giving the same adapter gold SQL-derived schema labels and pruning the
prompt around those labels moved value accuracy to `0.850`. Training a new
100-step adapter on the same oracle-labelled, schema-pruned format moved it
again to `0.890` value accuracy, `0.820` strict accuracy, and `0.7500`
interaction match.

That is not a production result because the labels are extracted from gold SQL.
It is an oracle diagnostic. Most of the lost accuracy was not in surface SQL
syntax. It was in the unmodeled data-engineering layer between a conversation
and a query plan: which objects matter, which joins are valid, what the
projection should contain, and which previous-turn constraints are still active.

## 1. Schema Metadata Is Not a Semantic Model

Spider-style `tables.json` is useful. It provides table names, column names,
types, primary keys, and foreign keys. That is enough to derive a rough semantic
summary, and this repo now does that.

It is not enough to define business meaning.

Raw schema metadata rarely tells you:

- Whether a table is a fact, dimension, bridge, snapshot, or event log.
- What one row represents.
- Whether a numeric column is additive, semi-additive, or non-additive.
- Which timestamp defines metric time.
- Which joins preserve grain and which joins duplicate rows.
- Which filters are part of a governed metric definition.
- Which columns are identifiers rather than groupable dimensions.

Cube and dbt both model concepts beyond raw tables. Cube describes cubes,
measures, dimensions, joins, access control, caching, and APIs. dbt semantic
models define entities, dimensions, measures, and defaults such as aggregation
time dimensions. Those concepts exist because physical schemas are not enough
for consistent analytics.

For fine-tuning, this means a useful semantic artifact should be explicit and
versioned:

```yaml
semantic_model:
  name: orders
  physical_table: orders
  grain: one row per order
  entities:
    - name: order_id
      type: primary
    - name: customer_id
      type: foreign
  dimensions:
    - name: order_status
      expr: status
      type: categorical
    - name: ordered_at
      expr: created_at
      type: time
  measures:
    - name: order_count
      expr: order_id
      agg: count_distinct
    - name: gross_revenue
      expr: gross_amount
      agg: sum
  joins:
    - to: customers
      relationship: many_to_one
      on: orders.customer_id = customers.customer_id
```

The current derived prompt section is a start. The next version should be a real
artifact with linting, tests, and hashes.

## 2. Grain Is the Hidden Failure

Many SQL errors are grain errors disguised as join errors.

Example:

> "Show revenue by customer segment."

If `orders` joins to `order_items`, and revenue is stored on `orders`, joining
before aggregating can duplicate revenue by item count. If revenue is stored on
`order_items`, aggregating at the order grain can undercount or overcount
depending on filters. The SQL may execute. The result may look plausible. The
benchmark may not catch it unless the fixture data exposes the fanout.

A serious multi-turn SQL eval needs fixture cases for:

- One-to-many joins where the wrong side duplicates measures.
- Many-to-many bridge tables.
- Distinct counts across joins.
- Snapshot facts where "current" and "as of" mean different rows.
- Event tables where time filters apply to event time, not entity creation time.
- Semi-additive measures such as balances or MRR.

The model should not have to infer all of that from column names. The semantic
model should state it, and the evaluator should contain data that punishes the
wrong interpretation.

## 3. Follow-Up Questions Need Resolved Forms

Multi-turn SQL examples should store both the original user turn and a resolved
standalone form.

Conversation:

```text
User: Which schools have more than 500 students?
User: How many are public?
```

Resolved second turn:

```text
How many schools with more than 500 students are public?
```

That resolved question is not necessarily shown to the model at inference time.
It can still be used for:

- Training supervision.
- Error analysis.
- Retrieval of relevant schema objects.
- Comparing whether the generated SQL answered the intended turn.
- Building query skeletons from the resolved intent.

QURG points in this direction by using question rewriting and schema-linking
relations for context-dependent text-to-SQL. This repo should add
`resolved_question` to prepared benchmark items so errors can be separated into
"bad context resolution" and "bad SQL generation."

The oracle schema-pruned run also shows why this should be separate from final SQL
generation. Once the model is handed a good schema plan, history-related
failures drop sharply: history-resolution error turns fell from 18 under
`minimal_executable` to 5 under the trained oracle schema-pruned adapter. That
does not mean history is solved. It means the current error taxonomy is no
longer dominated by obvious schema mistakes under the oracle condition, so the
next audit can finally see whether a non-oracle planner carries the right
conversational state.

## 4. Schema Linking Should Be an Artifact

The current model sees schema text and learns implicitly. That is weak.

Each benchmark item should eventually have:

```json
{
  "relevant_tables": ["orders", "customers"],
  "relevant_columns": [
    "orders.customer_id",
    "orders.gross_amount",
    "customers.segment"
  ],
  "join_path": [
    "orders.customer_id = customers.customer_id"
  ],
  "measures": ["gross_revenue"],
  "dimensions": ["customer_segment"]
}
```

This artifact can be created from gold SQL, checked against the semantic model,
and used in three ways:

- As optional training context.
- As a retrieval target for prompt pruning.
- As an evaluation diagnostic when generated SQL uses the wrong objects.

RESDSQL's separation of schema linking from skeleton parsing is a useful design
hint. The project does not need to copy the architecture to borrow the data
idea: do not make the decoder learn every intermediate decision invisibly.

The latest failure comparison is the strongest evidence for this design:

| Run | Value accuracy | Schema link | Join path | Projection | History turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| `minimal_executable` | 0.640 | 7 | 7 | 11 | 18 |
| `oracle_schema_pruned_minimal` prompt | 0.850 | 0 | 1 | 0 | 9 |
| `oracle_schema_pruned_100` trained adapter | 0.890 | 0 | 0 | 0 | 5 |

The gold-labelled planner features erased schema-link, join-path, and projection
failures on this slice. That is exactly the kind of intermediate state a data
pipeline can create, validate, version, and eventually predict. The open problem
is producing the same fields without reference SQL.

The next version should split this artifact into two fields:

- `gold_plan`: extracted from gold SQL and used for training, diagnostics, and
  upper-bound prompt runs.
- `predicted_plan`: produced without gold SQL from question, history, schema,
  semantic metadata, and retrieval.

The evaluator should score both. A wrong `predicted_plan` is a planner failure.
A correct `predicted_plan` followed by wrong SQL is a generator failure. Without
that split, every miss looks like "bad SQL," which is too vague to improve.

## 5. Evaluation Databases Need Adversarial Coverage

Execution accuracy is only as good as the database state. If a wrong query
returns the same rows as the gold query, the evaluator calls it correct.

A better eval set should include synthetic or curated database variants that
force differences for:

- Null handling.
- Duplicate rows.
- Ties in ordering.
- Empty groups.
- Case sensitivity.
- Floating point comparison.
- Date boundaries and time zones.
- Fanout from one-to-many joins.
- Foreign-key gaps.
- Rows that violate convenient assumptions.

The test-suite accuracy work for text-to-SQL exists for exactly this reason:
semantic correctness is better approximated when a query is checked across
database states that expose behavioral differences.

For this repo, the practical next step is smaller than full test-suite
generation. Add targeted SQLite fixtures for the most common failure classes,
then run generated SQL against those fixtures as an auxiliary score.

The remaining failures after oracle schema-pruned training make the fixture
backlog more specific:

- Seven value-grounding failures need rows that distinguish similar names,
  aliases, codes, dates, and numeric thresholds.
- Three execution errors need repair examples that include the failed SQL,
  SQLite error text, and the corrected query.
- One grain/fanout failure needs duplicated child rows or bridge rows that make
  the wrong aggregation visibly wrong.

This is the point of adversarial eval data. A benchmark slice should not merely
ask common questions. It should contain database states where plausible wrong
queries produce measurably different answers.

## 6. Dialect Is Part of the Data

The current CoSQL dev path uses SQLite. Production analytics stacks often use
Snowflake, BigQuery, Databricks SQL, Postgres, Redshift, or Trino. Dialect
differences affect:

- Date truncation.
- String matching.
- Regex syntax.
- Boolean expressions.
- Limit and offset behavior.
- Identifier quoting.
- Array and JSON operations.
- Window function support.

Every benchmark item should carry dialect metadata. If the model is trained on
mixed dialects, the prompt needs to say which dialect to emit. If the evaluator
uses SQLite as a proxy, the report should say so clearly.

## 7. Value Grounding Needs Its Own Artifact

Once schema planning improves, literal grounding becomes visible. The latest
oracle schema-pruned adapter still misses values through casing, aliases,
abbreviations, Roman numerals, partial names, and date spellings. Those are data
problems before they are model problems.

Prepared eval rows should carry a value artifact alongside schema labels:

```json
{
  "entity_resolution": [
    {
      "mention": "Delta",
      "table": "airlines",
      "column": "airline_name",
      "canonical_value": "Delta Airlines"
    }
  ],
  "literal_normalization": [
    {
      "mention": "Baldwin 1",
      "canonical_value": "Baldwin I"
    }
  ]
}
```

This creates a clean distinction between two failures. If the value artifact is
wrong, retrieval or normalization failed. If the artifact is right but SQL uses
the wrong literal, generation failed. Without that split, value grounding becomes
another vague bucket.

## 8. Observability Needs an Error Taxonomy

An aggregate score is not enough to improve the model. The evaluator should
classify failures:

- Invalid SQL.
- SQL execution error.
- Wrong table.
- Wrong join path.
- Wrong aggregation.
- Wrong grouping grain.
- Missing previous-turn filter.
- Added stale previous-turn filter.
- Wrong metric definition.
- Wrong value literal.
- Wrong ordering or limit.
- Dialect mismatch.

Some of this can be inferred automatically by comparing gold SQL, predicted SQL,
schema objects, and execution errors. Some may require manual labeling on a
small audit set. Either way, it should become structured data. Once failures are
typed, fine-tuning data can target them.

The current taxonomy is already useful enough to steer training. Before labels
and pruning, the failure profile included schema links, join paths, projections,
history turns, value grounding, and execution errors. After oracle-labelled
schema-pruned training, the profile compressed to value grounding, execution
errors, and one grain/fanout issue. That compression is the desired shape of an
iteration loop, as long as it is reported as oracle-conditioned:

1. Measure the aggregate score.
2. Classify failures into data-engineering causes.
3. Add an artifact, fixture, prompt, or training example for the largest cause.
4. Re-run the same slice and verify that the target class shrank without
   creating a worse class elsewhere.

The taxonomy should also become more precise. The remaining execution errors
should separate generic SQLite failures from `alias_role_swap` or
`wrong_table_for_column`, because those are repairable before scoring. The
history label should split into `referent_resolution` and `literal_carryover`;
those need different training examples.

Strict and value accuracy should stay side by side. The new adapter reaches
`0.890` value accuracy and `0.820` strict accuracy. That gap is not automatically
bad, because strict scoring can punish harmless alias or formatting differences.
It is still useful as a risk signal: strict misses deserve review when they
change ordering, duplicate behavior, selected columns, or aggregation shape.

Pairwise regression tracking should be part of every report. The new adapter
fixed 28 baseline failures, regressed 3, and netted +25 versus
`minimal_executable`. That is more actionable than the aggregate score alone.

## 9. DSPy Should Optimize Programs, Not Only Wording

The current DSPy loop can propose prompt variants and rank them by execution
accuracy. That was useful, but the winning oracle schema-pruned prompt was the
concise static one. The next DSPy target should be the two-stage program:

1. Predict a compact plan from question, history, schema, semantic metadata, and
   optional value indexes.
2. Generate SQL from that plan.

DSPy should optimize prompts against planner label F1, value accuracy, and
failure-taxonomy deltas on a development split, then promote only prompts that
hold up on a separate slice. Otherwise prompt search can overfit the same
100-turn report it is trying to improve.

## 10. CI for Semantic Evals

The repository should eventually treat semantic metadata like code:

- Lint semantic model files.
- Check every entity key against actual data for uniqueness and nulls.
- Check join cardinality against sample data.
- Detect measure name collisions with physical columns.
- Validate that every metric has an aggregation and time behavior.
- Hash database snapshots used for evaluation.
- Fail CI if a benchmark result mixes old and new schema versions.

This may sound like overhead, but it is the same discipline analytics teams need
before letting humans rely on metrics. A model makes the need sharper because it
can generate many plausible wrong queries quickly.

## What This Repo Should Build Next

The next concrete data-engineering tasks are:

1. Build a non-oracle `predicted_plan` step for relevant tables, relevant
   columns, join paths, projection shape, and query skeleton.
2. Keep `gold_plan` from SQL extraction as a supervised target and diagnostic
   upper bound.
3. Add value and entity-resolution artifacts for canonical literals, aliases,
   display-to-storage mappings, and sampled distinct values.
4. Add alias-role diagnostics for wrong-table column references before scoring.
5. Use DSPy to optimize the planner prompt against label F1, value accuracy, and
   failure-taxonomy deltas, not just final prompt wording.
6. Generate `resolved_question` for every CoSQL turn and store it in prepared
   metadata.
7. Add `semantic_models/` files for CoSQL databases used in the benchmark slice,
   with explicit grain, entities, measures, dimensions, and join cardinality.
8. Create value-grounding, execution-repair, and fanout regression fixtures for
   representative databases.
9. Add CI checks that prevent mixing benchmark results across schema, database,
   semantic-model, prompt, and label-extractor versions.

That is the path from a promising local fine-tuning loop to a serious
multi-turn SQL evaluation system.

Sources:

- Cube semantic layer introduction: https://docs.cube.dev/docs/introduction
- Cube data modeling docs: https://docs.cube.dev/docs/data-modeling/cubes
- dbt semantic models: https://docs.getdbt.com/docs/build/semantic-models
- dbt entities: https://docs.getdbt.com/docs/build/entities
- dbt measures: https://docs.getdbt.com/docs/build/measures
- dbt dimensions: https://docs.getdbt.com/docs/build/dimensions
- QURG: https://arxiv.org/abs/2305.06655
- RESDSQL: https://arxiv.org/abs/2302.05965
- Test-suite accuracy: https://arxiv.org/abs/2010.02840
