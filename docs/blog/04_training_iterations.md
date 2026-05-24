# Training Iterations: What Changed, What Did Not

The training loop is intentionally small. This is not because small data is
ideal, but because small controlled runs make it easier to see whether the
plumbing is honest.

Every main iteration follows the same shape:

1. Prepare a bounded JSONL training file.
2. Fine-tune `unsloth/Qwen3.5-9B` with LoRA.
3. Serve the adapter through vLLM.
4. Run the same 100-turn CoSQL dev endpoint benchmark.
5. Compare per-turn, dialog-level, syntax, and latency metrics.

Keeping the slice fixed matters. If the data, serving path, and evaluator all
change at once, the result is impossible to interpret.

The big turn in the project is now clear: broad semantic context did less than
expected, but explicit planning labels changed the failure profile. The honest
path has two tracks: `0.640` value accuracy is the best non-oracle prompt result,
while `0.850` and `0.890` are oracle-conditioned diagnostics that use
gold SQL-derived planning hints.

## Baseline

The base vLLM endpoint for `unsloth/Qwen3.5-9B` scored:

| Metric | Value |
| --- | ---: |
| Execution accuracy | 0.370 |
| Dialog execution accuracy | 0.347 |
| Interaction match rate | 0.0625 |
| Syntax accuracy | 1.000 |
| Mean latency | 292.79 ms |

That is the reference point. Any adapter has to beat this on execution, not just
produce cleaner-looking SQL.

## Iteration 1: 50 Steps on 192 Mixed Examples

The first non-smoke adapter used 64 examples from each configured training
source: CoSQL, SParC, and Gretel synthetic SQL.

```bash
.venv/bin/python -m data.prepare \
  --config configs/qwen35_9b_5090.yaml \
  --section train \
  --limit 64 \
  --output data/processed/train_64_each.jsonl

CC=/home/dan/.local/bin/cc .venv/bin/python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_64_each.jsonl \
  --max-steps 50 \
  --output-dir outputs/qwen35_9b_multiturn_sql_50steps \
  --report-to none
```

Observed training result:

- 192 records.
- 50 optimizer steps.
- Runtime: 194.5 seconds.
- Train loss: 0.9418.

Endpoint result:

| Metric | Base | 50-step adapter |
| --- | ---: | ---: |
| Execution accuracy | 0.370 | 0.420 |
| Dialog execution accuracy | 0.347 | 0.382 |
| Interaction match rate | 0.0625 | 0.09375 |
| Syntax accuracy | 1.000 | 0.980 |

This was the first useful signal: a small LoRA did improve execution outcomes,
but it also introduced a small syntax regression.

## Iteration 2: 100 Steps on the Same 192 Examples

The second adapter kept the data fixed and doubled the optimizer budget.

Observed training result:

- 192 records.
- 100 optimizer steps.
- Runtime: 370.0 seconds.
- Train loss: 0.7215.

Endpoint result:

| Metric | 50-step adapter | 100-step adapter |
| --- | ---: | ---: |
| Execution accuracy | 0.420 | 0.530 |
| Dialog execution accuracy | 0.382 | 0.515 |
| Interaction match rate | 0.09375 | 0.15625 |
| Syntax accuracy | 0.980 | 1.000 |

At this stage, this was the strongest adapter run. The important part was not
the absolute score; the slice was too small for that. The useful finding was
that more training on the same curated mix helped more than merely adding rows
at the same step count.

## Iteration 3: 384 Examples, Still 50 Steps

The third adapter doubled the per-source limit to 128 examples while holding the
step count at 50.

Observed training result:

- 384 records.
- 50 optimizer steps.
- Runtime: 190.4 seconds.
- Train loss: 0.9701.

Endpoint result:

| Metric | 50-step, 192 rows | 50-step, 384 rows |
| --- | ---: | ---: |
| Execution accuracy | 0.420 | 0.420 |
| Dialog execution accuracy | 0.382 | 0.389 |
| Interaction match rate | 0.09375 | 0.12500 |
| Syntax accuracy | 0.980 | 0.990 |

This did not move per-turn execution accuracy, but it improved dialog syntax and
interaction match rate. The likely interpretation is that broader data gave the
model more stable formatting and dialog coverage, but not enough optimizer
budget or targeted supervision to improve the specific turn-level decisions.

## Iteration 4: Semantic Context, 192 Examples, 50 Steps

The semantic iteration regenerated the 64-each training file with derived
semantic model context when Spider/CoSQL `tables.json` metadata was available.
CoSQL rows received both raw schema and a compact semantic section with cubes,
grain, dimensions, numeric measure candidates, and join hints.

```bash
.venv/bin/python -m data.prepare \
  --config configs/qwen35_9b_5090.yaml \
  --section train \
  --limit 64 \
  --output data/processed/train_64_each_semantic.jsonl

CC=/home/dan/.local/bin/cc .venv/bin/python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_64_each_semantic.jsonl \
  --max-steps 50 \
  --output-dir outputs/qwen35_9b_multiturn_sql_semantic_50steps \
  --report-to none
```

Observed training result:

- 192 records.
- 64 records with semantic model context.
- 50 optimizer steps.
- Runtime: 314.6 seconds.
- Train loss: 0.9246.

Endpoint result:

| Metric | Plain 50-step | Semantic 50-step |
| --- | ---: | ---: |
| Execution accuracy | 0.420 | 0.420 |
| Dialog execution accuracy | 0.382 | 0.384 |
| Interaction match rate | 0.09375 | 0.09375 |
| Syntax accuracy | 0.980 | 1.000 |
| Mean latency | 360.07 ms | 2285.79 ms |

This is not the win I wanted, but it is a useful result. The semantic prompts
were longer and slower. They improved training loss slightly and restored syntax
accuracy to 1.000, but they did not improve execution accuracy on the fixed
slice.

That does not invalidate semantic modeling. It says the current derived semantic
model is too shallow to change the model's decisions. Real semantic modeling
work needs validated grain, business metric definitions, join cardinality, and
fanout protection. Those details cannot be recovered reliably from table and
column names alone.

## Iteration 5: Gold Schema Labels as an Inference Diagnostic

The next jump did not come from another adapter. It came from changing what the
model was allowed to see at inference time.

For each eval turn, `data.prepare --include-sql-labels --prune-semantic-model`
extracts labels from the gold SQL: relevant tables, relevant columns, join
paths, query skeleton, projection shape, aggregation outputs, grouping, ordering,
limits, and duplicate-row policy. Those labels are then used to prune the schema
and semantic context before generation.

This is an oracle diagnostic, not a deployable system. It answers a narrower
question: if the model had the right schema-link plan, would SQL generation
still be the bottleneck?

The answer was mostly no.

| Run | Value accuracy | Strict accuracy | Interaction match | Schema link errors | Join path errors | Projection errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-semantic-50[minimal_executable]` | 0.640 | 0.420 | 0.3125 | 7 | 7 | 11 |
| `multiturn-sql-semantic-50[schema_pruned_minimal]` | 0.850 | 0.640 | 0.59375 | 0 | 1 | 0 |

Against `minimal_executable`, the oracle schema-pruned prompt fixed 25 turns,
regressed 4, and netted +21. The fixed set included schema-link, join-path, and
projection failures. That is strong evidence for an upper-bound claim: when the
intermediate plan is supplied from the answer key, final SQL generation becomes
much easier. It is not evidence that the model can infer the plan by itself.

DSPy was useful here as a search harness, but not because a verbose generated
prompt beat the hand-written one. The best DSPy candidates reached `0.820` to
`0.830` value accuracy, while the concise `schema_pruned_minimal` prompt reached
`0.850`. The lesson is that DSPy should optimize the planner contract and
failure-specific prompts, not just add more prose around final SQL generation.

## Iteration 6: Training on Oracle Schema-Pruned Labelled Prompts

The sixth run turned that diagnostic into training data. It regenerated the
training mix with oracle-labelled and schema-pruned prompts, using 128 examples
each from CoSQL, SParC, and Gretel:

```bash
.venv/bin/python -m data.prepare \
  --config configs/qwen35_9b_5090.yaml \
  --section train \
  --limit 128 \
  --output data/processed/train_128_each_semantic_pruned_labels.jsonl \
  --include-sql-labels \
  --prune-semantic-model \
  --strict

CC=/home/dan/.local/bin/cc .venv/bin/python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_128_each_semantic_pruned_labels.jsonl \
  --max-steps 100 \
  --output-dir outputs/qwen35_9b_multiturn_sql_schema_pruned_100steps \
  --report-to none
```

Observed training result:

- 384 records.
- 100 optimizer steps.
- Runtime: 707.5 seconds.
- Final train loss: 0.6783.
- Step-100 logged loss: 0.2569.

Oracle-conditioned endpoint result on the same 100-turn CoSQL slice:

| Run | Value accuracy | Strict accuracy | Syntax accuracy | Interaction match | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-semantic-50[minimal_executable]` | 0.640 | 0.420 | 1.000 | 0.3125 | n/a |
| `multiturn-sql-semantic-50[schema_pruned_minimal]` | 0.850 | 0.640 | 1.000 | 0.59375 | 548.36 ms |
| `multiturn-sql-schema-pruned-100[schema_pruned_minimal]` | 0.890 | 0.820 | 1.000 | 0.7500 | 560.00 ms |

The trained oracle-labelled adapter fixed 28 turns versus `minimal_executable`
and regressed 3, for a net +25. It also improved on the oracle prompt-only run:
value accuracy moved from `0.850` to `0.890`, strict accuracy from `0.640` to
`0.820`, and interaction match from `0.59375` to `0.7500`.

The remaining 11 wrong turns are now a different problem:

| Failure class | Count |
| --- | ---: |
| Value grounding | 7 |
| Execution error | 3 |
| Grain/fanout | 1 |

That residual distribution matters. The earlier system was dominated by schema
selection and projection shape. After oracle schema-pruned labelled training,
those failure classes disappear on this slice because the plan is supplied. The
next work should not be "more of the same prompt." It should focus on predicting
the plan without gold SQL, then on value grounding, execution repair, and
grain-aware fixture design.

The caveat remains important: both Iteration 5 and Iteration 6 rely on labels
extracted from gold SQL. The result is an upper-bound experiment for the
schema-linking layer. The production path is to predict or retrieve those labels
from the current question, dialog history, schema, and semantic model before SQL
generation.

## Current Iteration Table

| Iteration | Adapter | Train data | Steps | Train loss | Reported metric |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | none | 0 | 0 | n/a | 0.370 |
| 1 | `multiturn-sql-50` | 192 | 50 | 0.9418 | 0.420 |
| 2 | `multiturn-sql-100` | 192 | 100 | 0.7215 | 0.530 |
| 3 | `multiturn-sql-128each-50` | 384 | 50 | 0.9701 | 0.420 |
| 4 | `multiturn-sql-semantic-50` | 192 | 50 | 0.9246 | 0.420 |
| 5 | `multiturn-sql-semantic-50` + `semantic_grounding` prompt | 192 | 50 | 0.9246 | 0.440 |
| 6 | `multiturn-sql-semantic-50` + oracle schema-pruned prompt | 192 | 50 | 0.9246 | 0.850 value-aware oracle diagnostic |
| 7 | `multiturn-sql-schema-pruned-100` + oracle schema-pruned prompt | 384 | 100 | 0.6783 | 0.890 value-aware oracle diagnostic |

## What To Try Next

The next experiments should remove the oracle part of the current best run:

- Train a non-oracle schema-link planner that predicts relevant tables,
  relevant columns, join paths, projection shape, and query skeleton before SQL
  generation.
- Use DSPy against the planner prompt, not only the final SQL prompt, optimizing
  for value accuracy and failure-class reduction.
- Generate resolved standalone questions for CoSQL follow-up turns, then compare
  failures caused by context resolution with failures caused by SQL synthesis.
- Add value-grounding supervision for literals, filters, and entity references;
  that is now the largest remaining error class.
- Add value normalization artifacts for aliases, casing, abbreviations, Roman
  numerals, dates, and display-to-storage mappings.
- Add execution repair examples with failed SQL, SQLite error text, and corrected
  SQL, because execution errors survived even after schema-link pruning.
- Add alias-role validation so swapped table aliases are classified and repaired
  before they collapse into generic execution errors.
- Build grain and fanout stress fixtures instead of trusting the current SQLite
  dev data to expose every wrong join.

The current evidence says the model benefits from LoRA training, and the first
semantic improvement appears only when semantic context is present at inference
and the prompt explicitly asks the model to map the question to semantic cubes,
metrics, dimensions, filters, and join hints before writing SQL. That moved the
semantic 50-step adapter from `0.420` to `0.440` under the older strict metric.
Value-aware scoring then exposed a larger truth: schema-link labels and prompt
pruning are far more important than broad semantic context alone.

More rows alone did not solve it. Shallow semantic hints alone did not solve it.
Gold schema-link labels nearly eliminated the earlier dominant failure classes
under the oracle condition, and training on that format improved strictness and
dialog consistency. The data needs to encode the decisions that make multi-turn
SQL hard, then the system needs a non-oracle way to recover those decisions at
inference time.

Sources:

- QURG: https://arxiv.org/abs/2305.06655
- RAT-SQL: https://arxiv.org/abs/1911.04942
- RESDSQL: https://arxiv.org/abs/2302.05965
- DIN-SQL: https://arxiv.org/abs/2304.11015
- CoSQL: https://arxiv.org/abs/1909.05378
