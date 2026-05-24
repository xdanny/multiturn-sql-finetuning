# RTX 5090 Fine-Tuning and Benchmark Report

Date: 2026-05-24

## Executive Summary

This repo now has a verified local loop for multi-turn SQL fine-tuning:

- Prepare chat-format SQL data.
- Fine-tune `unsloth/Qwen3.5-9B` with LoRA on WSL2 + RTX 5090.
- Serve adapters through vLLM.
- Evaluate a fixed CoSQL dev slice through an OpenAI-compatible endpoint.
- Plot per-turn, dialog-level, syntax, and latency metrics.

The strongest non-oracle adapter remains `multiturn-sql-100`, trained for 100
steps on 192 mixed examples. It improves endpoint execution accuracy from
`0.370` to `0.530` on the same 100-turn CoSQL dev slice. Under value-aware
rescoring, the best non-oracle prompt condition reaches `0.640`.

The strongest overall number in this report is different in kind: the
schema-pruned 100-step adapter reaches `0.890` value accuracy only when the eval
prompt contains planning hints derived from gold/reference SQL. That is an
oracle diagnostic upper bound for the planning layer, not a production
text-to-SQL result.

The new semantic-context 50-step adapter is complete. Without semantic eval
prompts it ties the plain 50-step adapter on strict per-turn execution accuracy
(`0.420`) and improves syntax accuracy (`1.000` vs `0.980`). After regenerating
the eval slice with semantic context and running DSPy-backed prompt search, the
promoted `semantic_grounding` policy improves strict semantic accuracy to
`0.440`.

The first council-driven iteration repaired execution comparison. The original
metric required output column labels to match, which created false negatives for
alias-only differences such as `count(*)` vs `count(*) AS total`. The evaluator
now reports strict label-aware accuracy and value-only accuracy. Under
value-aware re-scoring, semantic-50 reaches `0.630`, tying `multiturn-sql-100`,
and the `minimal_executable` prompt variant reaches `0.640`.

## Research Notes

- Qwen/Qwen3.5-9B is the target 9B base model. The Hugging Face model card lists
  Apache-2.0 licensing and serving compatibility: https://huggingface.co/Qwen/Qwen3.5-9B
- CoSQL is the main multi-turn source. The official Yale page describes it as a
  conversational text-to-SQL corpus with 30k+ turns, 10k+ SQL queries, 3k
  dialogs, and 200 databases: https://yale-lily.github.io/cosql
- CoSQL frames the task as SQL-grounded dialogue state tracking over unseen
  databases: https://arxiv.org/abs/1909.05378
- SParC motivates context-dependent semantic parsing over coherent question
  sequences: https://arxiv.org/abs/1906.02285
- QURG, RAT-SQL, RESDSQL, and DIN-SQL all point toward explicit context
  resolution, schema linking, and decomposition rather than raw sequence
  generation alone: https://arxiv.org/abs/2305.06655,
  https://arxiv.org/abs/1911.04942, https://arxiv.org/abs/2302.05965, and
  https://arxiv.org/abs/2304.11015
- Cube and dbt semantic-layer docs motivate governed entities, measures,
  dimensions, joins, and aggregation behavior for AI-facing analytics:
  https://docs.cube.dev/docs/introduction and
  https://docs.getdbt.com/docs/build/semantic-models
- Test-suite accuracy research shows why one database state can be insufficient
  for semantic correctness: https://arxiv.org/abs/2010.02840

## Verified Local Environment

- WSL2 sees an RTX 5090 with about 31.8 GB VRAM.
- Training works with `torch 2.10.0+cu128` and Unsloth in `.venv`.
- vLLM serving works with `vllm 0.21.0`, `torch 2.11.0+cu130`, and CUDA 13.0
  runtime wheels in `.venv-vllm`.
- A user-local C compiler is available at `/home/dan/.local/bin/cc`.
- System `nvcc` is not installed. This is not a blocker for the current vLLM
  endpoint path because FlashInfer sampler JIT is disabled.

Verified command:

```bash
.venv/bin/python scripts/verify_blackwell.py --phase training
```

## Data Preparation

Training data is prepared as TRL-compatible chat JSONL. CoSQL records remain
multi-turn; evaluation expands each assistant SQL turn into a benchmark item
while preserving previous gold turns in prompt history.

Key prepared files:

| File | Purpose |
| --- | --- |
| `data/processed/train_64_each.jsonl` | 192-row mixed training set |
| `data/processed/train_64_each_semantic.jsonl` | 192-row mixed training set with semantic context where available |
| `data/processed/eval_100_each.jsonl` | Fixed CoSQL dev evaluation input |
| `data/processed/eval_100_each_semantic.jsonl` | Fixed CoSQL dev evaluation input with semantic context |

The endpoint benchmark consumes the first 100 assistant turns from CoSQL dev,
spanning 32 dialogs.

## Semantic Model Context

`data.prepare` now derives a Cube-inspired semantic summary from Spider/CoSQL
`tables.json` metadata:

- Cubes from physical tables.
- Grain and primary-key notes from table primary keys.
- Dimensions from table columns and inferred semantic types.
- Candidate measures from non-key numeric columns.
- Join hints from foreign-key pairs.

Example injected section:

```text
Semantic model:
- Cube orders (grain: one row per orders; primary key: order_id)
  Dimensions: order_id [number, primary_key], customer_id [number], amount [number]
  Measures: count, sum_amount=sum(amount), avg_amount=avg(amount)
  Joins: orders.customer_id -> customers.customer_id (many_to_one)
```

This is input context only. The model still emits SQL over physical tables and
columns.

Semantic training command:

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

Observed semantic training result:

- 192 total records.
- 64 rows include semantic context.
- Runtime: 314.6 seconds.
- Train loss: 0.9246.
- Adapter: `outputs/qwen35_9b_multiturn_sql_semantic_50steps/final`.

## vLLM Serving Setup

Working single-adapter serving command:

```bash
CC=/home/dan/.local/bin/cc \
VLLM_USE_FLASHINFER_SAMPLER=0 \
PATH="$PWD/.venv-vllm/bin:$PATH" \
vllm serve unsloth/Qwen3.5-9B \
  --enable-lora \
  --lora-modules multiturn-sql-semantic-50=outputs/qwen35_9b_multiturn_sql_semantic_50steps/final \
  --max-lora-rank 64 \
  --max-loras 1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-num-seqs 64
```

Important details:

- `CC=/home/dan/.local/bin/cc` gives Triton a compiler.
- `VLLM_USE_FLASHINFER_SAMPLER=0` avoids a JIT path that wants system `nvcc`.
- `--max-num-seqs 64` avoids cache-budget failures from higher default
  concurrency.
- Qwen endpoint calls pass `chat_template_kwargs.enable_thinking=false` so the
  model returns SQL rather than reasoning prose plus SQL.

Multi-LoRA serving was attempted but hit cache/CUDA graph memory pressure in the
current WSL2 environment. Serving one adapter at a time is the reliable path for
now.

## Endpoint Evaluation Results

All rows below use the same 100 CoSQL dev assistant turns, spanning 32 dialogs.

`plots/vllm_semantic_iterations_100turns/summary.csv`:

| Model | Execution accuracy | Syntax accuracy | Mean latency ms | Samples |
| --- | ---: | ---: | ---: | ---: |
| `multiturn-sql-100` | 0.530 | 1.000 | 392.88 | 100 |
| `multiturn-sql-128each-50` | 0.420 | 0.990 | 390.05 | 100 |
| `multiturn-sql-50` | 0.420 | 0.980 | 360.07 | 100 |
| `multiturn-sql-semantic-50` | 0.420 | 1.000 | 2285.79 | 100 |
| `unsloth/Qwen3.5-9B` | 0.370 | 1.000 | 292.79 | 100 |

`plots/vllm_semantic_iterations_100turns/dialog_summary.csv`:

| Model | Dialog execution accuracy | Dialog syntax accuracy | Interaction match rate | Dialogs | Turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-100` | 0.515 | 1.000 | 0.15625 | 32 | 100 |
| `multiturn-sql-128each-50` | 0.389 | 0.995 | 0.12500 | 32 | 100 |
| `multiturn-sql-50` | 0.382 | 0.977 | 0.09375 | 32 | 100 |
| `multiturn-sql-semantic-50` | 0.384 | 1.000 | 0.09375 | 32 | 100 |
| `unsloth/Qwen3.5-9B` | 0.347 | 1.000 | 0.06250 | 32 | 100 |

Iteration summary:

| Iteration | Adapter | Training data | Steps | Runtime | Train loss | Endpoint execution accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | none | 0 | 0 | n/a | n/a | 0.370 |
| 1 | `multiturn-sql-50` | 192 | 50 | 194.5s | 0.9418 | 0.420 |
| 2 | `multiturn-sql-100` | 192 | 100 | 370.0s | 0.7215 | 0.530 |
| 3 | `multiturn-sql-128each-50` | 384 | 50 | 190.4s | 0.9701 | 0.420 |
| 4 | `multiturn-sql-semantic-50` | 192 | 50 | 314.6s | 0.9246 | 0.420 |

## Corrected Execution Scoring

The local evaluator now distinguishes:

- `strict_execution_score`: result values and output column labels must match.
- `value_execution_score`: result values must match while harmless aliases may
  differ. Duplicate rows are preserved, and ordering is preserved for reference
  SQL with `ORDER BY` or `LIMIT`.

Existing generations were re-scored without regenerating model output:

```bash
.venv/bin/python -m eval.rescore_results \
  --input results/prompt_search_semantic50_semantic_grounding_100/semantic_grounding.jsonl \
  --output results/rescored/semantic_grounding.jsonl
```

Corrected summary:

| Model | Prompt variant | Value accuracy | Strict accuracy | Changed rows |
| --- | --- | ---: | ---: | ---: |
| `multiturn-sql-semantic-50` | `minimal_executable` | 0.640 | 0.420 | 22 |
| `multiturn-sql-100` | none | 0.630 | 0.530 | 10 |
| `multiturn-sql-semantic-50` | none | 0.630 | 0.420 | 21 |
| `multiturn-sql-semantic-50` | `semantic_grounding` | 0.630 | 0.440 | 19 |
| `multiturn-sql-50` | none | 0.610 | 0.420 | 19 |
| `unsloth/Qwen3.5-9B` | none | 0.590 | 0.370 | 22 |

Corrected plot outputs:

- `plots/rescored_vllm_semantic_prompt_iteration_100turns/summary.csv`
- `plots/rescored_vllm_semantic_prompt_iteration_100turns/dialog_summary.csv`

## Failure Taxonomy and Prompt Regression Report

The value-aware result files were classified into primary failure categories:
`schema_link`, `join_path`, `aggregation`, `grain_fanout`, `history_resolution`,
`ordering_limit`, `value_grounding`, `projection`, `invalid_sql`, and
`execution_error`. Classifier output is written under `results/classified/`, and
per-run summaries are written under `plots/failure_taxonomy/`.

The classifier repairs CoSQL operator spacing such as `> =`, preserves
case-sensitive string literals for SQLite value comparisons, distinguishes
projection order from alias-only differences, and flags `DISTINCT` mismatches as
grain/fanout errors. After this pass, every wrong turn in the six 100-turn runs
has an actionable primary label; there are zero `other` failures.

Comparison summary from `plots/failure_taxonomy/comparison/model_error_summary.csv`:

| Run | Value accuracy | Strict accuracy | Correct | Schema link | Join path | Value grounding | Projection | History turns | Invalid SQL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-semantic-50[minimal_executable]` | 0.640 | 0.420 | 64 | 7 | 7 | 6 | 11 | 18 | 0 |
| `multiturn-sql-100` | 0.630 | 0.530 | 63 | 7 | 4 | 8 | 9 | 17 | 0 |
| `multiturn-sql-semantic-50[semantic_grounding]` | 0.630 | 0.440 | 63 | 8 | 5 | 6 | 11 | 19 | 0 |
| `multiturn-sql-semantic-50` | 0.630 | 0.420 | 63 | 8 | 4 | 7 | 11 | 18 | 0 |
| `multiturn-sql-50` | 0.610 | 0.420 | 61 | 11 | 3 | 6 | 9 | 19 | 2 |
| `unsloth/Qwen3.5-9B` | 0.590 | 0.370 | 59 | 11 | 6 | 11 | 5 | 25 | 0 |

Pairwise comparison against `unsloth/Qwen3.5-9B`, from
`plots/failure_taxonomy/comparison/pairwise_vs_baseline.csv`:

| Candidate | Fixed | Regressed | Net fixed | Main fixed labels | Main regressed labels |
| --- | ---: | ---: | ---: | --- | --- |
| `multiturn-sql-semantic-50[minimal_executable]` | 9 | 4 | +5 | value grounding 4, schema link 3, aggregation 2 | projection 2, schema link 2 |
| `multiturn-sql-100` | 9 | 5 | +4 | schema link 3, value grounding 3 | projection 3 |
| `multiturn-sql-semantic-50` | 9 | 5 | +4 | schema link 3, value grounding 3, aggregation 2 | schema link 2, projection 2 |
| `multiturn-sql-semantic-50[semantic_grounding]` | 8 | 4 | +4 | value grounding 4, schema link 2, aggregation 2 | projection 2, schema link 2 |
| `multiturn-sql-50` | 8 | 6 | +2 | value grounding 3, schema link 2, aggregation 2 | schema link 3, projection 2 |

Diagnostic conclusion: `minimal_executable` is the best current semantic prompt
policy by value accuracy, interaction match rate, and net fixed turns. The
remaining failures show that the next semantic-modeling iteration should not be
another broad prompt rewrite. It should add schema-link labels and prompt
pruning first, then preserve output-shape instructions so fixes do not regress
projection order or required selected columns.

## DSPy-Backed Prompt Search

The semantic adapter was originally evaluated on `eval_100_each.jsonl`, which
does not include semantic model context. Regenerating the same 100-turn eval
slice with semantic context produced `data/processed/eval_100_each_semantic.jsonl`.

Base Qwen with semantic prompts stayed at `0.370`, so semantic context alone did
not help the base model. The semantic adapter with semantic prompts improved to
`0.440`.

Prompt optimization then searched static prompt policies plus two DSPy-proposed
policies on a 30-turn slice:

```bash
.venv/bin/python -m eval.prompt_optimize \
  --benchmark prepared \
  --endpoint http://127.0.0.1:8000/v1 \
  --model-name multiturn-sql-semantic-50 \
  --input data/processed/eval_100_each_semantic.jsonl \
  --limit 30 \
  --max-tokens 192 \
  --database-root data/raw/cosql_dataset/database \
  --output-dir results/prompt_search_semantic50_limit30 \
  --dspy-proposals 2
```

The best 30-turn variants were `semantic_grounding`, `minimal_executable`, and
`dspy_1` at `0.400`; the baseline variant scored `0.333`.

Full 100-turn promotion:

| Model | Prompt variant | Execution accuracy | Dialog execution accuracy | Interaction match rate |
| --- | --- | ---: | ---: | ---: |
| `multiturn-sql-semantic-50` | none | 0.420 | 0.384 | 0.09375 |
| `multiturn-sql-semantic-50` | `semantic_grounding` | 0.440 | 0.403 | 0.12500 |
| `multiturn-sql-semantic-50` | `minimal_executable` | 0.420 | 0.382 | 0.12500 |

Promoted config:

- `configs/prompt_variants/semantic_grounding.json`

Clean comparison output:

- `plots/vllm_semantic_prompt_iteration_100turns/summary.csv`
- `plots/vllm_semantic_prompt_iteration_100turns/dialog_summary.csv`

## Oracle Schema-Pruned Gold-Label Iteration

`data.prepare --include-sql-labels --prune-semantic-model` now extracts gold
SQL-derived planning labels and injects compact hints into each prepared turn:
relevant tables, relevant columns, join path, query skeleton, projection order,
aggregation outputs, group/order/limit shape, and duplicate-row policy. CoSQL
schema text is parsed into an allowlist before label extraction, so double-quoted
literal values such as names or roles are not stored as relevant columns.

Generated oracle-diagnostic artifacts:

- `data/processed/eval_100_each_semantic_pruned_labels.jsonl`
- `data/processed/train_64_each_semantic_pruned_labels.jsonl`
- `results/prompt_search_schema_pruned_projection_schemafix_100/summary.csv`
- `plots/failure_taxonomy/schema_pruned_projection_schemafix_comparison/`

Corrected 100-turn oracle prompt search:

| Model | Prompt variant | Value accuracy | Strict accuracy | Syntax accuracy | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: |
| `multiturn-sql-semantic-50` | `schema_pruned_minimal` | 0.850 | 0.640 | 1.000 | 548.36 ms |
| `multiturn-sql-semantic-50` | `schema_pruned_projection` | 0.830 | 0.620 | 1.000 | 542.86 ms |
| `multiturn-sql-semantic-50` | `dspy_2` | 0.830 | 0.620 | 1.000 | 566.77 ms |
| `multiturn-sql-semantic-50` | `dspy_1` | 0.820 | 0.620 | 1.000 | 564.67 ms |

Failure comparison against `minimal_executable`:

| Run | Value accuracy | Strict accuracy | Interaction match | Schema link | Join path | Projection | History turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-semantic-50[schema_pruned_minimal]` | 0.850 | 0.640 | 0.59375 | 0 | 1 | 0 | 9 |
| `multiturn-sql-semantic-50[minimal_executable]` | 0.640 | 0.420 | 0.3125 | 7 | 7 | 11 | 18 |

Pairwise movement: 25 fixed turns, 4 regressions, and net +21 versus
`minimal_executable`. The fixed failures include 10 projection errors, 6
schema-link errors, and 5 join-path errors. The remaining candidate failures are
mostly value grounding, grain/fanout, and execution errors.

This satisfies the diagnostic goal, but it is not a deployable serving path:
the labels are extracted from gold SQL. These rows are now treated as
`evaluation_mode=oracle_planner_diagnostic` and require `--allow-oracle-plan`
for endpoint or prompt-search evaluation. The next production-quality iteration
must predict or retrieve those labels from the question, history, schema, and
semantic model before SQL generation.

## Oracle Schema-Pruned Labelled Training Iteration

The next run trained directly on oracle-labelled, schema-pruned prompts. The
training set used 128 rows each from CoSQL, SParC, and Gretel:

- `data/processed/train_128_each_semantic_pruned_labels.jsonl`
- `outputs/qwen35_9b_multiturn_sql_schema_pruned_100steps/final`

Training summary:

- 384 records.
- 100 optimizer steps.
- Runtime: 707.5 seconds.
- Final train loss: 0.6783.
- Step-100 logged loss: 0.2569.

Oracle-conditioned endpoint result:

| Model | Prompt variant | Value accuracy | Strict accuracy | Syntax accuracy | Interaction match | Mean latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-schema-pruned-100` | `schema_pruned_minimal` | 0.890 | 0.820 | 1.000 | 0.75000 | 560.00 ms |
| `multiturn-sql-semantic-50` | `schema_pruned_minimal` | 0.850 | 0.640 | 1.000 | 0.59375 | 548.36 ms |
| `multiturn-sql-semantic-50` | `minimal_executable` | 0.640 | 0.420 | 1.000 | 0.31250 | n/a |

Failure comparison:

| Run | Value accuracy | Correct | Schema link | Join path | Projection | Value grounding | Execution error | Grain/fanout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-schema-pruned-100[schema_pruned_minimal]` | 0.890 | 89 | 0 | 0 | 0 | 7 | 3 | 1 |
| `multiturn-sql-semantic-50[schema_pruned_minimal]` | 0.850 | 85 | 0 | 1 | 0 | 8 | 3 | 3 |
| `multiturn-sql-semantic-50[minimal_executable]` | 0.640 | 64 | 7 | 7 | 11 | 6 | 2 | 1 |

Against `minimal_executable`, the trained oracle schema-pruned adapter fixed 28
turns, regressed 3, and netted +25. This confirms the main lesson from the
diagnostic run: explicit planning labels and prompt pruning are high-leverage
intermediate artifacts. It does not prove that the system can produce those
labels at inference time. The remaining bottleneck under the oracle condition
has shifted to value canonicalization, execution repair, and grain/fanout
fixtures.

The caveat is unchanged. The current best result still depends on gold
SQL-derived labels. The next target is a non-oracle planner that predicts the
same fields before SQL generation, then measures planner label F1 and final SQL
value accuracy separately.

## Interpretation

Among non-oracle runs, the strongest signal remains optimizer budget on the
balanced 192-row mix. The 100-step adapter improved both per-turn and dialog
execution accuracy.

The 384-row 50-step run did not improve per-turn execution, but it improved
interaction match rate versus the plain 50-step run.

The semantic 50-step run improved syntax reliability. Under the original strict
metric, semantic prompts plus `semantic_grounding` improved execution accuracy
from `0.420` to `0.440`. Under value-aware execution, semantic-50 is already
competitive with the 100-step adapter. Oracle schema-link pruning moves the same
adapter to `0.850`, and training on the oracle-labelled pruned format moves the
best oracle-conditioned run to `0.890`. The next semantic iteration should
replace oracle labels with a non-oracle schema-link planner and then repeat the
same failure-taxonomy comparison.

The next quality work should focus on a targeted data-engineering loop:

- Predict relevant tables, relevant columns, join paths, and projection shape
  without reading gold SQL.
- Add value/entity normalization for aliases, casing, abbreviations, Roman
  numerals, dates, and display-to-storage mappings.
- Add alias-role diagnostics for wrong-table column references before scoring.
- Use predicted labels to prune semantic prompt context before generation.
- Preserve projection shape explicitly: requested selected columns, selected
  column order, and whether duplicate rows are meaningful.
- Add resolved standalone questions for follow-up turns after schema-link labels
  exist, so context resolution errors can be separated from table-selection
  errors.
- Store real semantic models with grain, entities, measures, dimensions, and
  join cardinality.
- Build fanout and grain regression fixtures.
- Keep DSPy prompt search in the loop, but optimize planner prompts and
  failure-specific prompts on a held-out slice before full retraining.

## Blog and Research Docs

The blog series has been rewritten to reflect the current findings:

- `docs/blog/01_problem_and_result.md`
- `docs/blog/02_wsl_5090_setup.md`
- `docs/blog/03_data_and_eval.md`
- `docs/blog/04_training_iterations.md`
- `docs/blog/05_vllm_blackwell_deep_dive.md`
- `docs/blog/06_data_engineering_for_multiturn_sql_eval.md`

The deeper research note is:

- `docs/multiturn_semantic_modeling_research.md`

## GitHub Repo Feature Note

The GitHub connector/repo feature does not replace the local checkout or local
credentials. It can inspect GitHub metadata and repository files when installed
and authorized, but it does not bridge the WSL SSH agent, 1Password CLI, or
local `git clone` authentication. For private repo cloning, pushing, or SSH
checks, the reliable path is still local Git plus the user's configured SSH or
HTTPS credentials.

## Remaining Work Before Claiming Quality

- Evaluate on a larger CoSQL dev sample.
- Compare against the official CoSQL evaluation harness if leaderboard-style
  metrics are required.
- Replace gold SQL-derived schema-link labels with a non-oracle planner and
  score planner label F1 separately from final SQL execution.
- Add semantic-model fixtures beyond mechanically derived `tables.json` hints.
- Add value/entity normalization artifacts and alias-role diagnostics for the
  current value-grounding and execution-error bottlenecks.
- Add test-suite-style database variants for wrong-join and wrong-grain cases.
- Install a system CUDA Toolkit only if future source builds or FlashInfer JIT
  paths become necessary.
