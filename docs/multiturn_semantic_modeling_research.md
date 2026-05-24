# Multi-Turn SQL, Semantic Modeling, and Evaluation Research Notes

Date: 2026-05-24

## Goal

Improve the next fine-tuning phase by giving the model more than raw table
schemas. The target behavior is multi-turn SQL generation that:

- Resolves ellipsis and co-reference from previous turns.
- Links natural-language phrases to the right schema objects.
- Uses governed business semantics for metrics, dimensions, grain, and joins.
- Emits executable SQL over physical tables and columns.
- Can be evaluated with diagnostics that identify why a turn failed.

## Current Empirical Result

The repo now has a complete semantic-context experiment:

- Generated `data/processed/train_64_each_semantic.jsonl`.
- Trained `outputs/qwen35_9b_multiturn_sql_semantic_50steps/final`.
- Served the adapter through vLLM as `multiturn-sql-semantic-50`.
- Evaluated on the same 100 CoSQL dev assistant turns as the prior runs.

| Model | Execution accuracy | Dialog execution accuracy | Interaction match rate | Syntax accuracy |
| --- | ---: | ---: | ---: | ---: |
| Base `unsloth/Qwen3.5-9B` | 0.370 | 0.347 | 0.0625 | 1.000 |
| `multiturn-sql-50` | 0.420 | 0.382 | 0.09375 | 0.980 |
| `multiturn-sql-semantic-50` | 0.420 | 0.384 | 0.09375 | 1.000 |
| `multiturn-sql-100` | 0.530 | 0.515 | 0.15625 | 1.000 |

The first semantic run was mixed when evaluated without semantic context:
training loss improved slightly (`0.9246` vs `0.9418`) and syntax accuracy
returned to `1.000`, but execution accuracy stayed at `0.420`.

The first verified semantic gain came from fixing inference-time context and
adding DSPy-backed prompt search. Regenerating the eval slice with semantic
context and promoting the `semantic_grounding` prompt policy improved the
semantic adapter to `0.440` execution accuracy, `0.403` dialog execution
accuracy, and `0.125` interaction match rate.

The council review then found a measurement problem: local execution comparison
was too strict about output aliases. The evaluator now reports strict and
value-only execution accuracy. Re-scoring existing generations moves
semantic-50 to `0.630` value accuracy, tying the 100-step adapter, and
`minimal_executable` reaches `0.640` on the same 100-turn slice. This changes
the next question from "can semantic context move the aggregate score?" to "what
failure classes remain after alias-only false negatives are removed?"

The next pass answers that question with failure taxonomy reports. Every wrong
turn in the six rescored 100-turn runs is now labeled as schema linking, join
path, aggregation, grain/fanout, ordering/limit, value grounding, projection,
invalid SQL, or execution error. `minimal_executable` has the best pairwise
movement against the base model: 9 fixed turns, 4 regressions, and net +5. The
remaining semantic failures are concentrated in projection shape, schema links,
join paths, and value grounding.

The oracle schema-pruned diagnostic pass then extracted gold SQL labels for relevant
tables, relevant columns, join paths, query skeleton, projection shape, and
duplicate-row policy. Feeding those labels back into prompt pruning moves
`multiturn-sql-semantic-50[schema_pruned_minimal]` to `0.850` value accuracy,
`0.640` strict accuracy, and `0.59375` interaction match on the same 100 CoSQL
turns. It removes all primary schema-link and projection failures versus
`minimal_executable`, fixing 25 baseline failures while regressing 4. This is an
oracle-label result because the labels are derived from gold SQL; its main value
is proving an upper bound: schema-link and projection supervision are
high-leverage targets when a future system can predict them without reference
SQL.

Training on that oracle-labelled, schema-pruned format improves the result again. The
`multiturn-sql-schema-pruned-100[schema_pruned_minimal]` adapter reaches `0.890`
value accuracy, `0.820` strict accuracy, `1.000` syntax accuracy, and `0.7500`
interaction match. Pairwise against `minimal_executable`, it fixes 28 turns,
regresses 3, and nets +25. The remaining failures are concentrated in value
grounding, execution errors, and one grain/fanout issue, so the next research
question is no longer "can schema labels help?" It is "can the system recover
equivalent labels without gold SQL, then ground entity values reliably?"

The likely reason this helps only modestly is that the current semantic model is
derived mechanically from schema metadata. It exposes entities, dimensions,
measures, and joins, but it does not yet encode validated business grain, metric
definitions, fanout rules, or non-oracle prompt retrieval.

## Finding 1: Multi-Turn SQL Is State Tracking Over Executable Structure

CoSQL frames conversational text-to-SQL as dialogue state tracking where the
state is SQL rather than a task-specific slot-value map. Its official project
page describes 30k+ turns, 10k+ SQL queries, 3k dialogs, and 200 databases across
138 domains. It also emphasizes ambiguous questions, clarification, and
unanswerable turns.

SParC isolates the context-dependent parsing problem: coherent question
sequences over unseen databases where each turn may depend on previous turns.

Implementation consequence:

- Prepared CoSQL records stay multi-turn.
- Evaluation expands each assistant SQL turn while preserving prior gold turns.
- The system prompt tells the model to resolve follow-up questions from history.
- The report includes dialog-level metrics, not only per-turn metrics.

Sources:

- CoSQL project page: https://yale-lily.github.io/cosql
- CoSQL paper: https://arxiv.org/abs/1909.05378
- SParC paper: https://arxiv.org/abs/1906.02285

## Finding 2: Rewriting and Explicit State Should Be Data Artifacts

QURG argues that context-dependent text-to-SQL benefits from explicit modeling
of dependencies between the current question and previous questions. It uses
question rewriting plus schema-linking relations.

Rose-SQL is newer and training-free, but its Role-State framing is useful for
fine-tuning data: preserve a structural representation of query state and how it
changes through the conversation.

Implementation consequence:

- The current repo does not yet generate `resolved_question`.
- The next data-prep change should add a standalone resolved question per CoSQL
  turn.
- Error analysis should distinguish "bad context resolution" from "bad SQL after
  correct resolution."

Sources:

- QURG: https://arxiv.org/abs/2305.06655
- Rose-SQL: https://arxiv.org/abs/2605.03720

## Finding 3: Schema Linking Should Not Be Left Implicit

RAT-SQL, RASAT, RESDSQL, and DIN-SQL all point at the same bottleneck from
different angles: models struggle when schema relations and relevance are hidden
inside raw text.

- RAT-SQL encodes relations among schema items and question tokens.
- RASAT integrates relational structure into pretrained seq2seq modeling.
- RESDSQL decouples schema linking from SQL skeleton parsing.
- DIN-SQL decomposes prompting into schema linking, complexity classification,
  SQL generation, and self-correction.

Implementation consequence:

- The repo now injects compact semantic context, but that is only the first
  version of schema-linking support.
- Prepared examples should eventually include `relevant_tables`,
  `relevant_columns`, `join_path`, and `query_skeleton`.
- These artifacts can be extracted from gold SQL, validated against semantic
  models, used for prompt pruning, and reported in error analysis.

Sources:

- RAT-SQL: https://arxiv.org/abs/1911.04942
- RASAT: https://arxiv.org/abs/2205.06983
- RESDSQL: https://arxiv.org/abs/2302.05965
- DIN-SQL: https://arxiv.org/abs/2304.11015

## Finding 4: Semantic Layers Are the Right Abstraction, But Only If They Carry Real Semantics

Cube's docs frame the semantic layer as the shared context for humans and AI
agents. It centralizes metric definitions, joins, access rules, caching, and
APIs before queries reach the warehouse. Cube's data model uses cubes,
dimensions, measures, and joins.

dbt's Semantic Layer uses semantic models with entities, dimensions, measures,
metrics, and aggregation behavior. Its entity docs explicitly describe entities
as join keys across semantic models. Its measure docs show aggregation types and
non-additive dimensions, which are critical for analytics correctness.

Implementation consequence:

- The current `data.prepare` semantic context is Cube-inspired and useful for
  prompt structure.
- It should not be mistaken for a governed semantic layer.
- A stronger next version should store explicit semantic model files with grain,
  entity types, metric definitions, time behavior, join cardinality, and
  non-additive rules.

Sources:

- Cube introduction: https://docs.cube.dev/docs/introduction
- Cube cubes: https://docs.cube.dev/docs/data-modeling/cubes
- Cube joins: https://docs.cube.dev/docs/data-modeling/joins
- dbt semantic models: https://docs.getdbt.com/docs/build/semantic-models
- dbt entities: https://docs.getdbt.com/docs/build/entities
- dbt measures: https://docs.getdbt.com/docs/build/measures
- dbt dimensions: https://docs.getdbt.com/docs/build/dimensions

## Finding 5: Evaluation Needs Data Engineering, Not Just Scoring

Execution accuracy is necessary because SQL equivalence is not string
equivalence. It is still incomplete. A wrong query can pass if the database state
does not expose the difference. A correct query can fail if ordering, nulls,
duplicates, or dialect behavior differ from the gold query assumptions.

The test-suite accuracy work proposes evaluating against distilled database
test suites to better approximate semantic correctness. That idea matters here:
multi-turn SQL should be tested against database states that expose wrong joins,
wrong grain, stale filters, missing filters, and metric-definition errors.

Implementation consequence:

- Keep execution accuracy as the main quality loop.
- Add targeted fixture DBs for fanout, nulls, ties, empty groups, duplicate rows,
  and time-boundary cases.
- Add an error taxonomy so aggregate accuracy can be decomposed into actionable
  failure classes.

Source:

- Test-suite accuracy: https://arxiv.org/abs/2010.02840

## Incorporated Code Changes

Implemented in `data.prepare` and tests:

- `SYSTEM_PROMPT` now instructs the model to resolve follow-up questions from
  history and use semantic model hints.
- `data.prepare` can derive semantic model context from Spider/CoSQL
  `tables.json`.
- CoSQL examples include semantic context on the first user turn, so subsequent
  turns inherit it through chat history.
- Single-turn formatters accept `semantic_model_context`.
- Tests cover semantic model derivation, join extraction, measure candidates,
  and prompt injection.
- `eval.prompt_optimize` adds a DSPy-backed prompt search loop that proposes and
  scores prompt variants with the same execution evaluator used for endpoint
  benchmarks.
- `eval.result_compare` and `eval.rescore_results` add strict vs value-aware
  execution scoring and allow existing result files to be re-scored without
  regenerating model outputs.
- `eval.classify_errors` labels wrong turns with failure categories and emits
  per-run summaries under `plots/failure_taxonomy/`.
- `eval.compare_failures` compares classified runs against a baseline and
  reports fixed and regressed error classes under
  `plots/failure_taxonomy/comparison/`.
- `data.sql_labels` extracts SQL-derived schema-link and projection labels,
  including relevant tables, relevant columns, join path, query skeleton,
  selected-expression order, aggregations, grouping, ordering, limits, and
  duplicate-row policy.
- `data.prepare --include-sql-labels --prune-semantic-model` injects those
  labels as compact oracle planning hints, prunes semantic model context to the
  labelled tables for each turn, and marks records as
  `evaluation_mode=oracle_planner_diagnostic`.
- `eval.run_eval`, `eval.local_benchmark`, and `eval.prompt_optimize` reject
  oracle-conditioned prepared inputs by default unless `--allow-oracle-plan` is
  passed.
- Schema-aware label extraction parses compact schema text before filtering
  columns, preventing CoSQL double-quoted literal values from becoming
  `relevant_columns`.

Semantic model example:

```text
Semantic model:
- Cube orders (grain: one row per orders; primary key: order_id)
  Dimensions: order_id [number, primary_key], customer_id [number], amount [number]
  Measures: count, sum_amount=sum(amount), avg_amount=avg(amount)
  Joins: orders.customer_id -> customers.customer_id (many_to_one)
```

## Data Engineering Backlog

Completed diagnostic artifacts:

1. `relevant_tables` and `relevant_columns`: schema-linking labels extracted
   from gold SQL.
2. `join_path`: gold join path and relationship direction.
3. `query_skeleton`: structure labels for SELECT, JOIN, WHERE, GROUP BY,
   ORDER BY, and LIMIT decisions.
4. `projection_shape`: selected expressions, selected-column order,
   aggregation outputs, grouping, ordering, limits, and duplicate policy.
5. `error_taxonomy`: structured failure class for every wrong prediction.

The next phase should build these artifacts:

1. Non-oracle schema-link predictions from question, history, schema, and
   semantic model.
2. `semantic_models/`: versioned model files for benchmark databases.
3. `resolved_question`: standalone question per multi-turn example.
4. `db_snapshot`: hash or version marker for every evaluation DB.
8. `dialect`: explicit SQL dialect metadata for every benchmark item.

## Next Experiments

1. Build a non-oracle planner that predicts relevant tables, relevant columns,
   join path, query skeleton, projection shape, and duplicate policy from only
   question, history, schema, semantic metadata, and optional value indexes.
2. Keep gold SQL-derived labels as supervised targets and oracle upper bounds,
   not as production eval inputs.
3. Add projection-shape supervision: selected columns, selected column order,
   and duplicate-row requirements.
4. Add resolved-question supervision for CoSQL turns after schema-link labels
   are available, so context-resolution failures can be measured separately.
5. Add join-fanout and grain regression fixtures.
6. Add execution repair examples: flawed SQL, execution error, corrected SQL.
7. Train semantic-context data for 100 steps after the artifact and evaluator
   are stronger.
8. Use DSPy prompt search as the cheap first gate for semantic prompt changes,
   then promote only full-slice improvements.
