# Evidence Contract

This repo is testing whether a local Qwen 3.5 9B model can become useful on
BIRD-Interact-style multi-turn SQL. The current reproducible evidence is still a
CoSQL proxy slice, not a BIRD-Interact score and not a hosted-model comparison.
The broader methodology, dataset roles, training boundaries, and benchmark
rules are defined in `docs/methodology.md`.

The machine-readable benchmark protocol registry is
`configs/benchmark_protocols.yaml`. Run manifests and comparison artifacts
should name protocol ids from that registry before they can support benchmark or
transfer claims.

## Evaluation Modes

| Mode | Inference inputs | Allowed public claim |
| --- | --- | --- |
| `non_oracle_generation` | Question, conversation history, schema, semantic context, and non-oracle prompt variants | A deployable path improved on the fixed proxy slice. |
| Structured query brief | Non-oracle SQL generation with a compact visible brief before the SQL. | A structured-brief method claim only after same-row comparison beats direct SQL. |
| Historical `predicted_planner` | The same inputs plus a plan predicted without reference SQL | Negative planner-to-SQL evidence and leakage diagnostics only. |
| `metric_dsl` | A generated semantic metric intent, semantic model, and optional database-backed reference SQL | Metric-DSL quality, not direct-SQL superiority by itself. |
| `oracle_planner_diagnostic` | Gold SQL-derived planning hints or schema pruning from those hints | A ceiling test for how much planning/schema linking matters. |

Gold SQL-derived labels may be used as scorer targets in every mode. They may
enter the prompt only in `oracle_planner_diagnostic`.

## Current Claim Boundaries

| Claim | Status | Evidence | Mode | Allowed in blog |
| --- | --- | --- | --- | --- |
| Base local Qwen 3.5 9B reaches `0.370` strict and `0.590` value accuracy on the fixed CoSQL proxy turns. | `supported_proxy` | historical run manifests under `results/` or regenerated run manifests | `non_oracle_generation` | Yes, as a proxy result. |
| The 100-step LoRA reaches `0.530` strict and `0.630` value accuracy on the same proxy turns. | `supported_proxy` | historical run manifests under `results/` or regenerated run manifests | `non_oracle_generation` | Yes, as a proxy result. |
| The best non-oracle prompt/result currently reaches `0.640` value accuracy. | `supported_proxy` | historical run manifests under `results/` or regenerated run manifests | `non_oracle_generation` | Yes, if labeled value-only. |
| A prompt-only oracle diagnostic reaches `0.850` value accuracy. | `diagnostic_upper_bound` | diagnostic run manifest | `oracle_planner_diagnostic` | Yes, only as a ceiling test. |
| Gold SQL-derived planning hints can push the best diagnostic run to `0.890` value accuracy. | `diagnostic_upper_bound` | diagnostic run manifest | `oracle_planner_diagnostic` | Yes, only as a ceiling test. |
| The lexical planner baseline has macro score `0.571`, table F1 `0.599`, column F1 `0.117`, and skeleton F1 `0.648`. | `supported_planner_quality` | `docs/planner_baseline_cosql_dev_100_summary.json` | planner scoring | Yes, as planner quality, not SQL accuracy. |
| The non-oracle value index covers `0.783` of resolved SQL values and `0.755` of user-visible mention aliases on the fixed proxy labels. | `supported_value_retrieval_coverage` | `docs/data_artifacts/value_index_cosql_dev_100.manifest.json` | value retrieval coverage | Yes, as retrieval coverage only. |
| Semantic value retrieval inputs are ready for clean-holdout endpoint comparison. | `supported_preflight` | `docs/training_runs/semantic_value_clean_holdout_preflight_20260602.json` | `non_oracle_generation` | Yes, as endpoint-pair readiness only. |
| Semantic value retrieval improves SQL outcomes. | `unsupported_regression` | `docs/training_runs/semantic_value_clean_holdout_full_20260602.json` | `non_oracle_generation` | No. The full clean-holdout endpoint pair regressed versus direct SQL. |
| A structured query-brief adapter improves SQL execution. | Pending | `eval.run_structured_brief_comparison` is implemented | `non_oracle_generation` plus structured brief output | No, until same-row direct-SQL comparison metrics show a positive value-accuracy delta. |
| A non-oracle predicted planner improves SQL execution. | `unsupported_regression` | `docs/training_runs/planner_sft_1000_sql_limit24_negative_20260531.json` | historical `predicted_planner` | No. Keep as negative evidence, not an active claim. |
| A metric-DSL result exists with parse, compile, execution, and measure-preservation metrics. | `supported_metric_dsl_quality` | metric-DSL result manifest | `metric_dsl` | Yes, as metric-DSL quality only. |
| Metric-DSL generation beats direct SQL on metric-heavy rows. | Pending | `eval.compare_metric_dsl_direct_sql` is implemented | `metric_dsl` | No, until the compared manifest has a positive value delta and references the direct-SQL manifest. |
| A generated-history rollout result exists for the fixed CoSQL proxy. | Pending | `eval.rollout_eval` is implemented | `non_oracle_generation` | No, until a rollout result manifest exists. |
| Generated-history rollout beats teacher-forced history for the same model/input. | Pending | none | not run | No, diagnostic only until side-by-side comparison metrics exist. |
| Behavior/recovery tuning beats direct SQL under generated-history rollout. | Pending | none | not run | No, until recovery and direct-SQL control adapters are compared on the same rollout rows. |
| A same-protocol hosted baseline exists. | Pending | `eval.compare_hosted_baseline` is implemented | not run | No, this only proves the comparison protocol exists. |
| Local 9B beats the hosted baseline on the same rows. | Pending | none | not run | No, until the local manifest references a hosted baseline and shows a positive value delta. |
| Local 9B competes on real BIRD-Interact/Multi-BIRD. | Pending | none | not run | No. |

## Inconclusive Evidence And Writeups

Inconclusive conclusions are evidence, not throwaway notes. If a run is
well-formed but does not cleanly support a win, regression, or promotion, keep a
small durable record that names the rows, control, scorer, artifacts, and the
reason the result is inconclusive. Future writeups should cite that record when
explaining why a method was paused, rerouted, or redesigned.

Before a writeup uses an inconclusive result, verify three things:

- the run used the intended split roles and row identities;
- the inconclusive status follows from a manifest, comparison artifact,
  readiness report, or documented scorer audit rather than a chat-only
  impression;
- the writeup states both what was learned and what remains unproven.

Do not flatten inconclusive evidence into either a failure narrative or a
promising-result narrative. For example, a readiness run can prove that endpoint
inputs are compatible while also proving that method promotion is not justified.

## Data Artifact Evidence

The value-grounding artifact is the first versioned intermediate-state artifact
behind the semantic/value/entity work:

- `docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl`
- `docs/data_artifacts/value_grounding_labels_cosql_dev_100_summary.json`
- `docs/data_artifacts/value_grounding_labels_cosql_dev_100.manifest.json`
- `docs/data_artifacts/value_index_cosql_dev_100.jsonl`
- `docs/data_artifacts/value_index_cosql_dev_100_summary.json`
- `docs/data_artifacts/value_index_cosql_dev_100.manifest.json`

It is generated from prepared CoSQL turns and gold/reference SQL, so it is
allowed as a training label, scoring target, and coverage diagnostic. It is not
allowed as production prompt context. A production-style value/entity claim
still needs a non-oracle retriever or model-produced artifact to recover the same
bindings from question text, conversation history, schema, and versioned
value/entity artifacts.

Current label coverage on the fixed proxy slice: 106 SQL value references across
17 databases, 21 references that require conversation carryover, and 3
references where exact user-text matching cannot recover the stored SQL literal.

The non-oracle value index is generated from SQLite database contents, not from
reference SQL. It currently has 12,661 entries across 20 fixed-slice CoSQL
databases. Against the gold labels as a coverage evaluation only, it indexes
78.3% of resolved stored values and 75.5% of user-visible mention aliases. That
remaining gap is the artifact-backed reason to add alias/entity expansion before
claiming semantic value grounding improved SQL.

## Reproducible Proxy Commands

Create the direct-SQL control prepared artifacts:

```bash
uv run --active --no-sync python -m scripts.direct_sql_full_control \
  --config configs/direct_sql_full_non_oracle.yaml \
  --stage prepare \
  --run
```

Future endpoint runs through `eval.run_eval` write a manifest next to the JSONL
output by default. Keep run-specific manifest snapshots under `results/`.

Compare a structured-brief SQL run against direct SQL before claiming the
brief-first adapter improved execution:

```bash
uv run --active --no-sync python -m eval.run_structured_brief_comparison \
  --structured-manifest results/structured_brief/<run-id>.manifest.json \
  --direct-manifest results/direct_sql/<run-id>.manifest.json \
  --output results/structured_brief/<run-id>.compared.manifest.json
```

The comparison command requires non-oracle `prepared` manifests, the same model,
`non_oracle_generation` output rows on both sides, execution scores, and matching
row identities. A structured-brief SQL claim requires the compared manifest to
show that `structured_brief_value_delta_vs_direct_sql` is positive versus the
same-row direct-SQL control.

Run a semantic value-retrieval SQL pair against direct SQL before claiming the
value index improved execution:

```bash
uv run --active --no-sync python -m data.semantic_value_retrieval_inputs \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --value-index docs/data_artifacts/value_index_cosql_dev_100.jsonl \
  --value-index-manifest docs/data_artifacts/value_index_cosql_dev_100.manifest.json \
  --output data/processed/eval_cosql_dev_100_semantic_value_retrieval.jsonl \
  --summary-output docs/data_artifacts/semantic_value_retrieval_inputs_summary.json \
  --manifest-output docs/data_artifacts/semantic_value_retrieval_inputs.manifest.json \
  --max-matches-per-turn 4 \
  --retrieval-scope current_turn \
  --min-alias-chars 3
```

This input builder matches database-derived value-index aliases against
current-turn user-authored text by default, caps retrieval volume, and prunes
short ambiguous aliases while retaining numeric aliases. It does not use
reference SQL, gold planner labels, expected rows, assistant SQL, or future user
turns for retrieval matching. Use `--retrieval-scope history` only as a
diagnostic policy and label the run accordingly.

```bash
uv run --active --no-sync python -m eval.run_semantic_value_retrieval_comparison \
  --direct-input data/processed/eval_cosql_dev_100.jsonl \
  --semantic-input data/processed/eval_cosql_dev_100_semantic_value_retrieval.jsonl \
  --value-index-manifest docs/data_artifacts/value_index_cosql_dev_100.manifest.json \
  --output-dir results/semantic_value_retrieval \
  --run-id semantic_value_retrieval_cosql_dev_100 \
  --model-name <served-model> \
  --endpoint http://localhost:8000/v1 \
  --database-root data/raw/cosql_dataset/database \
  --limit 100 \
  --preflight-output results/semantic_value_retrieval/<run-id>.preflight.json
```

The paired runner validates that both prepared inputs expand to the same
non-oracle row identities, that the semantic input differs from the direct-SQL
control, and that the value-index manifest is database-derived. It then writes
direct, semantic, and compared manifests. The lower-level
`eval.compare_semantic_value_retrieval` still enforces the final comparison
contract: same model, matching row identities, non-oracle output rows, execution
scores on both sides, and the value-index manifest SHA. A semantic
value-retrieval SQL claim requires a positive value-accuracy delta versus direct
SQL. Coverage alone remains a retrieval artifact, not a SQL win.

Write semantic value-retrieval preflight outputs under `results/`. A preflight is
only input-compatibility evidence. It does not support a SQL-improvement claim
without endpoint result manifests.

Historical planner-risk summaries are bounded to `risk summary only; no SQL
execution claim`. They can explain why the old planner route was paused, but
they are not active Checkpoint 5 evidence. For historical non-lexical planners,
JSONL predictions were keyed by expanded turn id and evaluated through
`eval.planner_eval`. The planner loader preserved raw unknown fields long enough
to reject oracle provenance markers before writing a predicted prepared artifact.

Evaluate a `MEASURE()`-preserving metric DSL:

```bash
python -m eval.metric_dsl_eval \
  --input results/metric_dsl/<run-id>.predictions.jsonl \
  --output results/metric_dsl/<run-id>.jsonl \
  --manifest-output results/metric_dsl/<run-id>.manifest.json \
  --model-name <served-or-offline-model-name>
```

Compare it with direct SQL on the same metric-heavy rows before claiming the
DSL-first path is better:

```bash
python -m eval.compare_metric_dsl_direct_sql \
  --metric-dsl-manifest results/metric_dsl/<run-id>.manifest.json \
  --direct-sql-manifest results/direct_sql/<run-id>.manifest.json \
  --output results/metric_dsl/<run-id>.compared.manifest.json
```

The direct manifest must use `benchmark=metric_dsl_direct_sql` and
`evaluation_mode=non_oracle_generation`. The models may differ, but row
identities, hashes, non-oracle status, direct execution scores, and the comparer
provenance marker must match before a metric-DSL comparison can support a method
claim.

Run generated-history rollout without teacher-forcing prior gold SQL:

```bash
uv run --active --no-sync python -m eval.rollout_eval \
  --model-name <served-model-name> \
  --endpoint http://127.0.0.1:8000/v1 \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --database-root data/raw/cosql_dataset/database \
  --output results/rollout/<run-id>.jsonl
```

Rollout rows use `history_policy=model_generated_sql_rollout`. A rollout result
can support a proxy rollout claim, but not a behavior/recovery method win.
Teacher-forced comparison metrics are a diagnostic comparison that shows whether
clean history was hiding generated-history failure.

Create those comparison metrics with:

```bash
uv run --active --no-sync python -m eval.compare_rollout_history \
  --rollout-manifest results/rollout/<run-id>.manifest.json \
  --teacher-forced-manifest results/teacher_forced/<run-id>.manifest.json \
  --output results/rollout/<run-id>.compared.manifest.json
```

For behavior/recovery smoke inputs, `eval.run_behavior_recovery_comparison`
produces the rollout, teacher-forced diagnostic, and comparison manifest
together.

The comparison command requires the same model, same input hash, and non-oracle
manifests before a rollout-vs-teacher-forced diagnostic can be reported. A
separate behavior/recovery claim still requires recovery-adapter generations
compared with the direct-SQL control adapter under generated-history rollout.

## Blog Rule

A blog sentence can make a benchmark claim only if it names one of:

- `non_oracle_generation` for deployable proxy results;
- structured query brief for same-row brief-first SQL comparison claims;
- historical `predicted_planner` for negative planner-to-SQL evidence;
- `metric_dsl` for metric-intent quality and direct-SQL comparison claims;
- `oracle_planner_diagnostic` for ceiling tests;
- `pending` for work that has not been run.

If the sentence cannot be labeled with one of those modes, it is probably too
vague to publish.
