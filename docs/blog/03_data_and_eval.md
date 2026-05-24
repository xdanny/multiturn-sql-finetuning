# Data and Evaluation Are the Product

The model is not the hardest part of this project. The hard part is building a
data and evaluation loop that tells the truth.

A multi-turn SQL model can fail in ways that look superficially correct:

- It remembers the right table but drops the previous filter.
- It keeps the filter but changes the aggregation grain.
- It chooses a join path that duplicates rows.
- It uses the right metric name with the wrong numerator or denominator.
- It emits SQL that executes but answers a different resolved question.

Those failures are data-engineering failures as much as model failures. If the
training examples do not expose grain, joins, metrics, and conversation state,
the model has to infer too much from column names. If the evaluator only checks
that SQL parses, it rewards confident wrong answers.

## Training Record Shape

The prepared data is chat-native JSONL. Each row has a `messages` array, and SQL
appears in assistant turns:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Schema: ...\nQuestion: ..."},
    {"role": "assistant", "content": "SELECT ..."}
  ],
  "metadata": {
    "source": "...",
    "db_id": "..."
  }
}
```

For CoSQL, a single prepared row can contain multiple user and assistant turns.
That keeps the dialog structure visible during training. It also creates a clean
path for evaluation: expand every assistant SQL turn into a benchmark item while
preserving the earlier turns as context.

The target benchmark direction is BIRD-Interact-style multi-turn SQL. The
current data mix is a bridge toward that, not a claim that every dataset is
equivalent:

| Dataset | Job in this repo |
| --- | --- |
| BIRD-Interact | Target dynamic interaction benchmark and eventual larger-model comparison |
| BIRD mini-dev | BIRD-style execution harness and SQLite database handling |
| CoSQL | Current local proxy for multi-turn dialog state and CoSQL dev evaluation |
| SParC | Context-dependent SQL coverage, currently from a flattened HF mirror |
| Gretel synthetic SQL | Schema-rich single-turn reinforcement and SQL variety |

The CoSQL source is the official Yale archive. That matters because stale or
partial mirrors can silently change schema paths, split names, and evaluation
assumptions.

## Teacher-Forced Multi-Turn Evaluation

The benchmark used here is teacher-forced. For each evaluated SQL turn, previous
gold assistant SQL remains in the prompt history.

This measures a precise skill:

> Given clean conversation history and a database schema, can the model generate
> the next SQL query?

It does not measure fully autonomous agent behavior. The model is not forced to
live with its own previous mistakes. That harder evaluation should come later.
The teacher-forced version is still the right first gate because it isolates
context use from recovery behavior.

The fixed endpoint slice contains 100 assistant turns from CoSQL dev, spanning
32 dialogs. The evaluator reports:

- `accuracy`: mean normalized or execution score across turns.
- `dialog_execution_accuracy`: per-dialog mean score, then averaged.
- `interaction_match_rate`: fraction of dialogs where every evaluated turn
  scored correctly.
- `syntax_accuracy`: fraction of generations accepted by the SQL parser/scorer.
- `mean_latency_ms`: endpoint latency for the generated turn.

Interaction match rate is intentionally unforgiving. If a user is following a
thread of analysis, a single wrong turn can break the workflow.

## Execution Accuracy Is Necessary But Not Sufficient

When a matching SQLite database exists, the evaluator runs predicted SQL and
gold SQL and compares results. That is more useful than string matching because
many SQL strings can be semantically equivalent.

Execution accuracy still has traps:

- It can produce false positives when two wrong queries happen to return the
  same result on a small database.
- It can produce false negatives when queries are equivalent but differ in row
  ordering, duplicate handling, type coercion, or acceptable null behavior.
- It can hide metric-definition errors if the dataset does not contain cases
  that expose the difference.
- It can overfit to SQLite behavior when the production warehouse is Snowflake,
  BigQuery, Postgres, or Spark SQL.

This is why Spider moved toward test-suite accuracy and why the literature keeps
revisiting semantic evaluation. A serious multi-turn SQL eval needs more than
one static database state. It needs adversarial database instances or
regression fixtures that expose join fanout, null semantics, duplicate rows,
empty groups, time windows, and ordering assumptions.

## Semantic Context Injection

The current formatter can derive a compact semantic model section from
Spider-style `tables.json` metadata:

```text
Semantic model:
- Cube orders (grain: one row per orders; primary key: order_id)
  Dimensions: order_id [number, primary_key], customer_id [number], amount [number]
  Measures: count, sum_amount=sum(amount), avg_amount=avg(amount)
  Joins: orders.customer_id -> customers.customer_id (many_to_one)
```

For CoSQL rows, the semantic section is placed on the first user turn. Later
turns inherit it through chat history. The assistant still emits plain SQL
against physical tables and columns.

This is intentionally Cube-inspired, not Cube-compatible. It is a lightweight
training hint, not a replacement for a real semantic layer. The distinction is
important. A real semantic model would define:

- Entity keys and uniqueness assumptions.
- Grain for every fact-like object.
- Measures with aggregation functions and filters.
- Dimensions with time and categorical typing.
- Join relationships and row-preservation semantics.
- Access-control and policy constraints.
- Non-additive or semi-additive metric rules.

The first semantic adapter tied the plain 50-step adapter on per-turn execution
accuracy (`0.420`) and improved syntax accuracy (`1.000` vs `0.980`). That is a
reasonable first result: structured metadata helped formatting robustness but
did not yet provide enough signal to improve answer correctness.

## The Data Engineering Backlog

The next quality jump should come from better artifacts around the model, not
from treating the model as an isolated decoder.

The repo should grow these data artifacts:

- `resolved_question`: a standalone version of every follow-up question.
- `relevant_tables` and `relevant_columns`: schema-linking supervision.
- `gold_plan` and `predicted_plan`: separate answer-key supervision from the
  plan a production system can actually produce.
- `query_skeleton`: SQL structure without exact columns or values.
- `semantic_model`: a governed, versioned model per database, not only derived
  hints.
- `evaluation_mode`: a required field that distinguishes non-oracle generation
  from oracle planner diagnostics.
- `error_taxonomy`: execution error, wrong join, wrong grain, wrong metric,
  missing filter, wrong value, wrong ordering, and invalid SQL.
- `dialect`: explicit SQL dialect metadata for every benchmark item.
- `db_snapshot`: database version hash, because evaluation results are only
  meaningful against a specific data state.

Those are data-engineering concerns. They are also the difference between a
demo and a benchmark that can guide real model development.

Sources:

- CoSQL project page: https://yale-lily.github.io/cosql
- CoSQL paper: https://arxiv.org/abs/1909.05378
- Spider paper summary: https://semantic-parsing.github.io/publications/yu2018spider/
- Test-suite accuracy paper: https://arxiv.org/abs/2010.02840
- Cube semantic layer docs: https://docs.cube.dev/docs/introduction
- dbt Semantic Layer semantic models: https://docs.getdbt.com/docs/build/semantic-models
