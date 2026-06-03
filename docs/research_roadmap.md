# Research Roadmap

This is the canonical roadmap for the repo.

The end goal is a measured multi-turn SQL claim:

> On held-out multi-turn analytical SQL tasks, the best local Qwen fine-tuned
> method beats raw Qwen and is compared against a hosted SOTA model under the
> same row-matched protocol.

The roadmap is data-first. We should not spend more effort on elaborate
planning abstractions until direct fine-tuning, clean splits, hosted baselines,
and failure analysis are boring and reproducible.

## Current State

The repo has useful proxy evidence, not a final benchmark claim.

- Raw Qwen 9B on the fixed 100-turn CoSQL proxy slice: `0.590` value accuracy,
  `0.370` strict accuracy.
- Direct SQL LoRA, 100 steps: `0.630` value accuracy, `0.530` strict accuracy,
  `1.000` syntax accuracy.
- Semantic-context LoRA, 50 steps: `0.640` value accuracy, `0.420` strict
  accuracy, `1.000` syntax accuracy.
- Answer-key-pruned diagnostic LoRA, 100 steps: `0.890` value accuracy,
  `0.820` strict accuracy, `1.000` syntax accuracy. This is diagnostic only
  because reference-derived context narrowed the task.

The biggest lesson so far is not that a hand-written planning layer is ready.
It is that the model often fails to select the right schema objects, values, or
query shape. The next work should teach and evaluate those skills through data,
clean prompt-visible context, and benchmarks.

## Principles

- Multi-turn is different from single-turn SQL because later questions depend
  on dialog state, changed constraints, prior entities, and generated-history
  errors.
- The answer key can be used for scoring, failure analysis, and training-split
  supervision. It must not be shown to the model during evaluation.
- Every local-vs-local and local-vs-hosted claim must use the same row IDs,
  same database files, same scorer, and comparable manifests.
- Direct SQL is the control. Any semantic, DSL, value, or recovery method must
  beat the direct-SQL version of the same model on the same rows.
- Inconclusive or negative results are research evidence. Keep them visible and
  label them as such.

## Checkpoint 0: Clean The Repo

Status: in progress.

Remove the old active process-heavy workflow from code and docs. Keep the useful
evaluators, scorers, data generators, and result manifests. Keep small artifacts
only when they define a fixture, data input, or measured run.

Done when:

- Active code has no process-heavy cleanup-era entry points.
- README explains what the repo does and how to reproduce the core run.
- Old parallel docs are deleted or replaced by this roadmap.
- Remaining checked-in artifacts are small and explainable.

## Checkpoint 1: Split Integrity

Create durable split manifests before scaling training.

Required roles:

- `train`: rows allowed for supervised fine-tuning.
- `validation`: rows allowed for method design and failure analysis.
- `proxy_dev_seen`: historical CoSQL dev 100 continuity slice.
- `clean_local_holdout`: rows reserved for local benchmark claims.
- `external_target`: BIRD, BIRD-Interact, LiveSQLBench, Spider 2.0, or similar
  transfer benchmarks.

Done when:

- Each split manifest records source, row IDs, database IDs, dialog IDs, turn
  counts, hash, and intended role.
- The 100-turn CoSQL proxy is explicitly marked as seen proxy evidence, not a
  clean final benchmark.
- Evaluation commands refuse accidental row mismatches in comparisons.

## Checkpoint 2: Direct SQL Baselines At Scale

Rebuild the boring baseline before adding more methods.

Compare:

- raw Qwen;
- direct SQL LoRA trained on the larger available training split;
- direct SQL LoRA with different step counts or mixtures only if the data role
  is unchanged.

Report:

- value accuracy;
- strict accuracy;
- syntax accuracy;
- interaction match;
- generated-history rollout accuracy;
- latency;
- run cost when hosted models are used.

Done when:

- Raw Qwen and best direct-SQL LoRA have row-matched manifests on validation and
  clean holdout.
- The best local direct-SQL adapter is selected without looking at holdout
  results.

## Checkpoint 3: Hosted SOTA Comparator

Use OpenRouter or another hosted endpoint as the SOTA comparator, not as a
label-smuggling mechanism.

Compare:

- raw Qwen baseline;
- best local Qwen LoRA;
- hosted SOTA model such as Claude Sonnet through the same prompt-visible
  inputs.

Done when:

- Hosted and local outputs are scored on the same row IDs.
- Hosted manifests include endpoint/model, latency, and cost fields.
- The repo can say whether local fine-tuning closes, matches, or fails to close
  the hosted gap.

## Checkpoint 4: Failure Taxonomy

Let failures decide which method deserves more work.

Classify misses into:

- dialog state: follow-up question misunderstood;
- schema selection: wrong table, column, join, aggregation, grouping, or
  duplicate policy;
- value grounding: display value does not map to storage value;
- metric semantics: governed measure, dimension, filter, or grain is wrong;
- execution recovery: invalid SQL, empty result, or later turn degraded by
  earlier generated SQL.

Done when:

- Failure reports exist for raw Qwen, best direct-SQL LoRA, and hosted SOTA on
  the same validation rows.
- Method proposals are tied to the largest remaining failure bucket.

## Checkpoint 5: Data Pre-Training For Decomposition

Replace planner-first code with data that teaches decomposition implicitly and
measures whether it helps SQL outcomes.

The model does not need to expose a separate planner product. We can train on
examples whose assistant answer is still SQL, while the prompt and data mixture
teach useful decomposition:

- conversation state summaries;
- schema object selection from prompt-visible schema;
- value candidates from database-derived indexes;
- metric definitions from governed semantic artifacts;
- repair examples that show failed generated SQL and observed rows.

Done when:

- Training rows are generated from `train` data or synthetic fixtures with clear
  provenance.
- Validation prompts never include future turns, expected rows, or evaluation
  reference SQL.
- A decomposed-data adapter beats the direct-SQL adapter on the same validation
  rows before it is tried on holdout.

## Checkpoint 6: Semantic And Value Context

Scale only context that can exist at inference time:

- database schema introspection;
- semantic model metadata;
- value indexes from database contents;
- aliases and column-role constraints;
- retrieval summaries derived from visible user text.

Done when:

- Context generation has manifests and leakage boundaries.
- Context coverage is measured before SQL accuracy is claimed.
- The context-enhanced adapter or prompt beats direct SQL on row-matched
  validation.

## Checkpoint 7: Metric DSL Only If It Parses

Metric DSL is still a research arm, not a primary API.

Scale it only if:

- generated DSL parses;
- generated DSL compiles to executable SQL;
- governed `MEASURE(...)` intent survives;
- compiled SQL beats direct SQL on metric-heavy rows.

If it parses but loses, record that as evidence against scaling the arm.

## Checkpoint 8: Generated-History Recovery

Teacher-forced history is not enough. Multi-turn recovery must evaluate later
turns after earlier turns used generated SQL.

Done when:

- rollout eval runs over multi-dialog slices;
- recovery rows expose generated SQL and observed results, not future answers;
- a recovery adapter beats the same direct-SQL adapter under generated-history
  rollout.

## Checkpoint 9: Publishable Claim

A publishable claim needs:

- clean local holdout result;
- raw Qwen baseline;
- best local fine-tuned model;
- hosted SOTA comparator;
- failure taxonomy;
- negative evidence for methods that did not help;
- reproduction commands and manifests.

Until then, write the project as an empirical notebook: what we tried, how it
was measured, what improved, what failed, and what the next clean test is.
