# Methodology

This project asks whether a local Qwen 3.5 9B model can become competitive with
state-of-the-art general models on data-analysis work that requires database
reasoning, not just one-shot SQL completion. The sharper research question is
whether a small specialized model can learn behavior and semantic concepts that
matter specifically for multi-turn analytical SQL.

The current evidence is a CoSQL proxy loop. It is not yet a BIRD-Interact score
and not yet a hosted-model comparison.

## Experiment Taxonomy

| Term | Meaning in this repo | What it can prove |
| --- | --- | --- |
| `non_oracle_generation` | SQL generation from the user question, prior conversation, schema/semantic context, and non-oracle prompt instructions. | A deployable path improved on the fixed proxy slice. |
| `predicted_planner` | A planner predicts tables, columns, joins, shape, and value hints without reading reference SQL; the SQL generator consumes that plan. | A production-style planner-to-SQL system can be evaluated end to end. |
| `oracle_planner_diagnostic` | Planning hints or schema pruning are derived from the reference SQL. | A ceiling test for how much schema linking and planning matter. |
| `gold_plan` | Answer-key labels extracted from reference SQL. | Planner scoring target only. |
| `predicted_plan` | Non-oracle planner output. | Prompt input only when `evaluation_mode=predicted_planner`. |

Gold SQL-derived labels may be used for scoring in every mode. They may enter
the model prompt only in `oracle_planner_diagnostic`. Training now rejects
oracle diagnostic rows by default; use `--allow-oracle-diagnostic-data` only for
explicit teacher-forced diagnostic experiments.

## Dataset Composition

| Source | Turn shape | Role | Current caveat |
| --- | --- | --- | --- |
| CoSQL | Multi-turn dialogs over Spider-style SQLite databases. | Primary local proxy for stateful SQL evaluation and LoRA iteration. | Evaluation history is teacher-forced by prior gold SQL, so this does not test recovery after an earlier model mistake. |
| SParC | Context-dependent SQL questions. | Supplemental signal for stateful SQL behavior. | The accessible `jellyChiru/SParC` mirror is flattened, so it is not treated as a full dialog substitute. |
| Synthetic schema-rich SQL | Single-turn SQL with explicit schema context. | Schema variety and SQL-format reinforcement. | It does not prove conversational ability. |
| BIRD mini-dev | Single-turn BIRD-style SQLite rows. | Execution harness and BIRD-style schema debugging. | It is not an interactive benchmark. |
| BIRD-Interact | Interactive data-analysis target. | Future local-vs-hosted comparison. | Not run yet in this repo. |

Prepared JSONL records now carry `assistant_turn_count`, `turn_format`, and
`history_policy`. A multi-turn CoSQL dialog should show
`history_policy=gold_sql_teacher_forced`, because the expanded prompt for turn N
contains prior reference SQL from the dialog, not the model's own earlier output.
`data.prepare --manifest-output <path>` writes a companion JSON manifest with
source counts, evaluation modes, turn formats, history policies, assistant-turn
totals, and configured dataset weights.

Generated-history rollout evaluation is separate from this teacher-forced path.
`eval.rollout_eval` runs dialog turns sequentially and writes
`history_policy=model_generated_sql_rollout` output rows, where each later turn
sees the model's generated SQL from earlier turns. This is the required surface
for behavior/recovery claims. A rollout result alone is not enough; it must be
compared against the same model and input under teacher-forced history before
claiming recovery behavior improved.

## Dataset Decomposition

The fixed proxy result is intentionally decomposed before making broader claims:

1. CoSQL-only proxy slice: fast local iteration over 100 evaluated turns across
   32 dialogs.
2. Non-oracle SQL generation: base model and LoRA adapters without reference
   SQL-derived prompt hints.
3. Oracle diagnostics: the same slice with gold SQL-derived planning hints to
   estimate how much a correct intermediate plan helps.
4. Planner scoring: predicted plans are compared with `gold_plan` before SQL is
   generated.
5. Future BIRD-Interact run: same claim ledger, manifests, and mode separation
   applied to the real target benchmark.

This split is there to avoid a common failure mode: mixing datasets, prompts,
models, and evaluators until a number improves but no one knows why.

## Fine-Tuning Strategies

The repo currently distinguishes these training strategies:

| Strategy | Data | Claim status |
| --- | --- | --- |
| Plain non-oracle LoRA | CoSQL/SParC/synthetic-style chat rows without gold planning hints in the prompt. | Production-style proxy. |
| Semantic-context LoRA | Non-oracle rows with schema/semantic model context. | Production-style proxy if no gold pruning is used. |
| Oracle-labelled LoRA | Rows with gold SQL-derived planning hints or semantic pruning by gold tables. | Diagnostic only; useful for testing whether the SQL generator can consume a correct plan. |
| Predicted-planner-to-SQL LoRA | Rows or prompts where the plan is produced without reference SQL. | Target production path; it supports an improvement claim only after same-model direct-SQL comparison metrics show a positive value-accuracy delta. |

The next strategy table should be more ambitious than these early runs:

| Strategy | Question it answers |
| --- | --- |
| Direct SQL SFT | Can small-model SQL behavior be improved with ordinary supervised fine-tuning? |
| Planner/DSL first, SQL second | Is it easier to learn a typed intermediate representation than raw SQL directly? |
| Semantic-layer tuning | Does a governed model of entities, dimensions, measures, grain, and joins reduce errors that raw schema text cannot? |
| `MEASURE()`-preserving metric DSL | Should the model preserve governed metrics until a compiler expands them to SQL? |
| Behavior/recovery tuning | Can the model learn when to clarify, inspect values, repair SQL, and recover after its own previous mistakes under generated-history rollout? |

The first `MEASURE()` experiment surface is implemented in `data.metric_dsl` and
documented in `docs/metric_dsl_contract.md`. It scores semantic intent before SQL
execution so metric preservation can be compared against direct SQL generation.

The `weight` field in dataset configs is metadata for experiment design today;
current preparation caps each configured source with `--limit` and does not yet
perform weighted sampling. A larger training run should replace per-source caps
with an explicit mixture manifest before claiming dataset-scale conclusions.
Until then, every prepared artifact used for a claim should include the
composition manifest produced by `data.prepare --manifest-output`.

## Benchmark Methodology

Every publishable number should have:

- a prepared-input hash;
- an output hash;
- the model and adapter path;
- the evaluation mode;
- whether oracle inputs were allowed;
- the database root;
- strict and value-only execution accuracy;
- row and dialog counts;
- the exact command or enough command metadata to rerun it.

`eval.run_eval` writes result manifests by default. Historical proxy manifests
are tracked in `docs/result_manifests/cosql_dev_100_proxy.json`.

Strict execution accuracy checks returned values and output labels. Value-only
accuracy ignores harmless alias differences but still cares about row values,
duplicates, and order-sensitive outputs. Older single `accuracy` fields should
be treated as strict-era results unless a manifest or rescoring artifact says
otherwise.

## Planner Methodology

The planner is not another name for a prompt. It is the intermediate state that
must be correct before SQL generation becomes reliable:

- relevant tables;
- relevant columns;
- join path;
- query skeleton;
- projection shape;
- aggregation and grouping;
- duplicate-row policy;
- value/entity matches.

The first implemented planner is a lexical baseline. It is intentionally weak
and inspectable. Its purpose is to create a scoring surface before building a
stronger planner, not to claim the planner problem is solved.

Predicted-planner SQL execution is compared against direct SQL, not judged in
isolation. A raw `predicted_planner` manifest proves only that the planner path
ran. The repo requires a comparison artifact from `eval.compare_predicted_planner`
and the referenced direct-SQL manifest in the ledger input before clearing the
planner-to-SQL improvement claim.

## Current Claim Boundary

Supported:

- local 9B improves on a fixed CoSQL proxy slice;
- value-aware scoring exposes real gains hidden by alias-sensitive strict
  scoring;
- oracle planning hints produce a much higher ceiling, so schema linking and
  projection planning are high-leverage;
- the first non-oracle planner baseline is measurable.

Not supported yet:

- local 9B competes with hosted state-of-the-art systems;
- local 9B is competitive on BIRD-Interact;
- the predicted planner improves SQL execution;
- current dataset mixing is optimal.
