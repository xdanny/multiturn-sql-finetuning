# Evidence Contract

This repo is testing whether a local Qwen 3.5 9B model can become useful on
BIRD-Interact-style multi-turn SQL. The current reproducible evidence is still a
CoSQL proxy slice, not a BIRD-Interact score and not a hosted-model comparison.
The broader methodology, dataset roles, training boundaries, and benchmark
rules are defined in `docs/methodology.md`.

## Evaluation Modes

| Mode | Inference inputs | Allowed public claim |
| --- | --- | --- |
| `non_oracle_generation` | Question, conversation history, schema, semantic context, and non-oracle prompt variants | A deployable path improved on the fixed proxy slice. |
| `predicted_planner` | The same inputs plus a plan predicted without reference SQL | A production-style planner-to-SQL path can be evaluated. |
| `metric_dsl` | A generated semantic metric intent, semantic model, and optional database-backed reference SQL | Metric-DSL quality, not direct-SQL superiority by itself. |
| `oracle_planner_diagnostic` | Gold SQL-derived planning hints or schema pruning from those hints | A ceiling test for how much planning/schema linking matters. |

Gold SQL-derived labels may be used as scorer targets in every mode. They may
enter the prompt only in `oracle_planner_diagnostic`.

## Claim Ledger

The claim ledger turns this contract into a checked artifact. It reads result
manifests, verifies referenced hashes, scans input/output rows for oracle
planning leakage, joins classified failure counts, and writes one row per
claimable artifact plus explicit pending rows for missing evidence:

```bash
python -m eval.claim_ledger
```

Tracked outputs:

- `docs/claim_ledgers/cosql_dev_100.jsonl`
- `docs/claim_ledgers/cosql_dev_100_summary.csv`

The current ledger is intentionally conservative:

- non-oracle CoSQL results are `supported_proxy`;
- oracle-planner rows are `diagnostic_upper_bound`;
- planner summaries are `supported_planner_quality`, not SQL accuracy;
- generated-history rollout existence, rollout-vs-teacher-forced improvement,
  predicted-planner SQL execution, metric-DSL evaluation, metric-DSL-vs-direct-SQL
  improvement, hosted baseline existence, local-vs-hosted outperformance, and
  BIRD-Interact comparison remain `pending` until same-protocol result manifests
  and positive comparison deltas exist.

Any hash mismatch, missing manifest field, non-oracle oracle marker, or
predicted-planner manifest whose output rows are not also marked
`predicted_planner` is downgraded to `pending` with a blocker. A `metric_dsl`
manifest is also downgraded unless it reports positive parse, compile, execution,
and measure-preservation metrics.

## Claim Ledger Rows

| Claim | Status | Artifact | Mode | Allowed in blog |
| --- | --- | --- | --- | --- |
| Base local Qwen 3.5 9B reaches `0.370` strict and `0.590` value accuracy on the fixed CoSQL proxy turns. | `supported_proxy` | `docs/result_manifests/cosql_dev_100_proxy.json` | `non_oracle_generation` | Yes, as a proxy result. |
| The 100-step LoRA reaches `0.530` strict and `0.630` value accuracy on the same proxy turns. | `supported_proxy` | `docs/result_manifests/cosql_dev_100_proxy.json` | `non_oracle_generation` | Yes, as a proxy result. |
| The best non-oracle prompt/result currently reaches `0.640` value accuracy. | `supported_proxy` | `docs/result_manifests/cosql_dev_100_proxy.json` | `non_oracle_generation` | Yes, if labeled value-only. |
| A prompt-only oracle diagnostic reaches `0.850` value accuracy. | `diagnostic_upper_bound` | `docs/result_manifests/cosql_dev_100_proxy.json` | `oracle_planner_diagnostic` | Yes, only as a ceiling test. |
| Gold SQL-derived planning hints can push the best diagnostic run to `0.890` value accuracy. | `diagnostic_upper_bound` | `docs/result_manifests/cosql_dev_100_proxy.json` | `oracle_planner_diagnostic` | Yes, only as a ceiling test. |
| The lexical planner baseline has macro score `0.571`, table F1 `0.599`, column F1 `0.117`, and skeleton F1 `0.648`. | `supported_planner_quality` | `docs/planner_baseline_cosql_dev_100_summary.json` | planner scoring | Yes, as planner quality, not SQL accuracy. |
| A non-oracle predicted planner improves SQL execution. | Pending | `data/processed/eval_cosql_dev_predicted_planner_100.jsonl` can now be generated | `predicted_planner` | No, until same-model direct-SQL comparison metrics show a positive value-accuracy delta. |
| A metric-DSL result exists with parse, compile, execution, and measure-preservation metrics. | Pending | `eval.metric_dsl_eval` is implemented | `metric_dsl` | No, until a valid metric-DSL manifest exists. |
| Metric-DSL generation beats direct SQL on metric-heavy rows. | Pending | `eval.compare_metric_dsl_direct_sql` is implemented | `metric_dsl` | No, until the compared manifest has a positive value delta and references the direct-SQL manifest. |
| A generated-history rollout result exists for the fixed CoSQL proxy. | Pending | `eval.rollout_eval` is implemented | `non_oracle_generation` | No, until a rollout result manifest exists. |
| Generated-history rollout beats teacher-forced history for the same model/input. | Pending | none | not run | No, until side-by-side comparison metrics exist. |
| A same-protocol hosted baseline exists. | Pending | `eval.compare_hosted_baseline` is implemented | not run | No, this only proves the comparison protocol exists. |
| Local 9B beats the hosted baseline on the same rows. | Pending | none | not run | No, until the local manifest references a hosted baseline and shows a positive value delta. |
| Local 9B competes on real BIRD-Interact/Multi-BIRD. | Pending | none | not run | No. |

The Stage 6 input contract is now also checked in:

- `docs/data_artifacts/hosted_baseline_rows.jsonl`
- `docs/data_artifacts/hosted_baseline_summary.json`
- `docs/data_artifacts/hosted_baseline.manifest.json`

Those rows freeze the non-oracle prepared proxy slice that a hosted baseline
must use before any local-vs-hosted claim is even comparable.

The Stage 6 BIRD-Interact transfer input contract is also checked in:

- `docs/data_artifacts/bird_interact_transfer_rows.jsonl`
- `docs/data_artifacts/bird_interact_transfer_summary.json`
- `docs/data_artifacts/bird_interact_transfer.manifest.json`

Those artifacts do not claim a real BIRD-Interact result yet. They freeze the
benchmark surface and naming contract that a future local and hosted run must
declare before the claim ledger can clear `bird_interact_local_vs_hosted`.

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
still needs a non-oracle retriever or planner to recover the same bindings from
question text, conversation history, schema, and versioned value/entity
artifacts.

Current label coverage on the fixed proxy slice: 106 SQL value references across
17 databases, 21 references that require conversation carryover, and 3
references where exact user-text matching cannot recover the stored SQL literal.

The non-oracle value index is generated from SQLite database contents, not from
reference SQL. It currently has 12,661 entries across 20 fixed-slice CoSQL
databases. Against the gold labels as a coverage evaluation only, it indexes
78.3% of resolved stored values and 75.5% of user-visible mention aliases. That
remaining gap is the artifact-backed reason to add alias/entity expansion before
claiming semantic value grounding improved SQL.

The Stage 3 synthetic finetuning path now mirrors that claim boundary:

- `docs/data_artifacts/semantic_layer_training_rows.jsonl`
- `docs/data_artifacts/semantic_layer_training_rows_summary.json`
- `docs/data_artifacts/semantic_layer_training_rows.manifest.json`
- `docs/data_artifacts/semantic_layer_direct_sql_training_rows.jsonl`
- `docs/data_artifacts/semantic_layer_direct_sql_training_rows_summary.json`
- `docs/data_artifacts/semantic_layer_direct_sql_training_rows.manifest.json`

Those rows are derived from the curated synthetic fixtures, but the prompt path
is explicitly non-oracle. The semantic-layer prompt can see schema, dialog
history, and governed semantic model context. It cannot see scorer-only fields
such as `expected_rows`, `reference_sql`, or gold DSL strings. The direct
control uses the same rows and SQL targets without the semantic model so the
comparison stays interpretable.

The Stage 3 proxy package makes the same boundary concrete on the fixed CoSQL
slice:

- `docs/data_artifacts/semantic_proxy_train.jsonl`
- `docs/data_artifacts/semantic_proxy_eval.jsonl`
- `docs/data_artifacts/semantic_proxy_summary.json`
- `docs/data_artifacts/semantic_proxy.manifest.json`
- `docs/data_artifacts/semantic_proxy_direct_sql_train.jsonl`
- `docs/data_artifacts/semantic_proxy_direct_sql_eval.jsonl`
- `docs/data_artifacts/semantic_proxy_direct_sql_summary.json`
- `docs/data_artifacts/semantic_proxy_direct_sql.manifest.json`

Those artifacts reference the non-oracle value-label and value-index summaries,
reuse the prepared CoSQL eval contract, and keep the same dialog-turn identity
across the semantic and direct-control arms. The semantic training pack also
filters mixed source files down to the rows that actually contain semantic
context, so Stage 3 is not benchmarked against a half-semantic training split.

The Stage 6 hosted comparison wrapper now validates the local candidate manifest
before comparing result manifests:

```bash
uv run python -m eval.run_hosted_baseline_comparison \
  --local-training-manifest outputs/<local-run>/training.manifest.json \
  --local-result-manifest results/<local-run>.manifest.json \
  --hosted-result-manifest results/<hosted-run>.manifest.json \
  --output results/<local-run>.vs_hosted.manifest.json
```

That wrapper is intentionally narrow. It does not run the hosted model. It
checks that the local candidate came from the prepared non-oracle path first,
then hands off to `eval.compare_hosted_baseline` for the same-row comparison.

The BIRD-Interact transfer wrapper is similarly narrow:

```bash
uv run python -m eval.run_bird_interact_comparison \
  --local-result-manifest results/<local-bird-run>.manifest.json \
  --hosted-result-manifest results/<hosted-bird-run>.manifest.json \
  --output results/<local-bird-run>.vs_hosted.manifest.json
```

It does not run either model. It requires both result manifests to already use
a `bird_interact` benchmark and `non_oracle_generation`, then hands off to the
same local-vs-hosted comparer so the Stage 6 transfer path stays machine-checkable.

## Reproducible Proxy Commands

Create a CoSQL-only prepared artifact:

```bash
python -m data.prepare \
  --config configs/cosql_dev_planner.yaml \
  --section eval \
  --limit 100 \
  --output data/processed/eval_cosql_dev_100.jsonl \
  --manifest-output data/processed/eval_cosql_dev_100.manifest.json
```

Generate non-oracle planner predictions with an OpenAI-compatible endpoint:

```bash
python -m eval.planner_predict \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --limit 100 \
  --model-name <planner-model> \
  --endpoint http://localhost:8000/v1 \
  --output results/planner_predictions/<run-id>.jsonl
```

Screen planner prompt or DSPy-program variants before full SQL generation:

```bash
python -m eval.planner_optimize \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --limit 100 \
  --model-name <planner-model> \
  --endpoint http://localhost:8000/v1 \
  --output-dir results/planner_prompt_search/<run-id> \
  --dspy-proposals 2
```

The planner optimizer writes `summary.csv` and per-variant JSONL files. It ranks
parseable planner output before field-level F1, and malformed JSON receives zero
planner credit. These are planner-quality artifacts only; they do not clear the
`predicted_planner_sql_execution` claim until a predicted-planner SQL manifest
beats a row-matched direct-SQL manifest.

Create the first 100-turn predicted-planner artifact from those predictions:

```bash
python -m eval.planner_eval \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --limit 100 \
  --planner-source json_planner_predictions \
  --planner-predictions results/planner_predictions/<run-id>.jsonl \
  --predicted-prepared-output data/processed/eval_cosql_dev_predicted_planner_100.jsonl \
  --output results/planner_eval_cosql_dev_100.jsonl \
  --summary-output results/planner_eval_cosql_dev_100_summary.json
```

For local planner-supervision checkpoints, the repo now also has:

```bash
uv run python -m eval.run_local_planner_eval \
  --training-manifest results/train/planner_supervision.manifest.json \
  --output-dir results/planner_local \
  --run-id planner_local_probe \
  --model-name unsloth/Qwen3.5-9B \
  --adapter-path outputs/planner_supervision/final \
  --predicted-prepared-output data/processed/eval_cosql_dev_predicted_planner_100.jsonl
```

That path generates planner JSON locally, scores it with `planner_eval`, and
can write the `predicted_planner` prepared artifact that Stage 2 consumes.

Future endpoint runs through `eval.run_eval` write a manifest next to the JSONL
output by default. Historical manifest snapshots for the currently cited proxy
numbers live in `docs/result_manifests/cosql_dev_100_proxy.json`.

Compare a predicted-planner SQL run against direct SQL before claiming the
planner improved execution:

```bash
python -m eval.compare_predicted_planner \
  --predicted-manifest results/predicted_planner/<run-id>.manifest.json \
  --direct-manifest results/direct_sql/<run-id>.manifest.json \
  --output results/predicted_planner/<run-id>.compared.manifest.json
```

The comparison command requires non-oracle `prepared` manifests, the same model,
`predicted_planner` output rows, direct `non_oracle_generation` output rows, and
matching row identities. The claim ledger clears
`predicted_planner_sql_execution` only when the predicted-planner value accuracy
beats the direct-SQL value accuracy and the referenced direct-SQL manifest is
present in the same ledger input.

Use the paired runner for the actual endpoint experiment:

```bash
python -m eval.run_predicted_planner_comparison \
  --direct-input data/processed/eval_cosql_dev_100.jsonl \
  --predicted-input data/processed/eval_cosql_dev_predicted_planner_100.jsonl \
  --output-dir results/predicted_planner \
  --run-id lexical_planner_cosql_dev_100 \
  --model-name <served-model> \
  --endpoint http://localhost:8000/v1 \
  --database-root data/raw/cosql_dataset/database \
  --limit 100 \
  --preflight-output docs/predicted_planner_comparison_preflight.json
```

`docs/predicted_planner_comparison_preflight.json` records that the current
direct and predicted prepared inputs have matching row identities for the fixed
100-turn proxy. This is only a readiness artifact. It cannot clear the claim
without the endpoint result manifests and comparison metrics.

For local checkpoint experiments on the same paired inputs, use:

```bash
uv run python -m eval.run_local_predicted_planner_comparison \
  --direct-training-manifest results/train/direct_sql_control.manifest.json \
  --predicted-training-manifest results/train/predicted_planner.manifest.json \
  --output-dir results/predicted_planner_local \
  --run-id lexical_planner_local \
  --model-name unsloth/Qwen3.5-9B \
  --direct-adapter-path outputs/direct_sql/final \
  --predicted-adapter-path outputs/predicted_planner/final \
  --database-root data/raw/cosql_dataset/database
```

That path reuses `eval.local_benchmark`, writes local result manifests for both
sides, and then writes the same predicted-planner comparison manifest format.

Summarize whether that endpoint pair is worth running before spending model
time:

```bash
python -m eval.planner_readiness \
  --planner-input results/planner_eval_cosql_dev_100.jsonl \
  --preflight-input docs/predicted_planner_comparison_preflight.json \
  --output docs/planner_readiness_cosql_dev_100.json
```

The tracked readiness report is bounded to `readiness only; no SQL execution
claim`. It records that the row pair is ready, but the lexical planner still has
high column-linking and projection-shape risk: `0.790` zero-column-F1 turns,
`0.820` selected-count mismatches, and `1.000` empty projection-expression
turns. Its recommendation is `improve_planner_before_claim`, not `run_endpoint_pair`.

For non-lexical planners, write JSONL predictions keyed by expanded turn id and
run `eval.planner_eval --planner-source json_planner_predictions
--planner-predictions <path>`. The planner loader preserves raw unknown fields
long enough to reject oracle provenance markers before writing a predicted
prepared artifact.

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
provenance marker must match before the ledger can clear
`metric_dsl_beats_direct_sql`.

Run generated-history rollout without teacher-forcing prior gold SQL:

```bash
python -m eval.rollout_eval \
  --model-name <served-model-name> \
  --endpoint http://127.0.0.1:8000/v1 \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --database-root data/raw/cosql_dataset/database \
  --output results/rollout/<run-id>.jsonl
```

Rollout rows use `history_policy=model_generated_sql_rollout`. A rollout result
can support a proxy rollout claim, but not a behavior/recovery improvement claim
until the ledger also has same-model teacher-forced comparison metrics.

Create those comparison metrics with:

```bash
python -m eval.compare_rollout_history \
  --rollout-manifest results/rollout/<run-id>.manifest.json \
  --teacher-forced-manifest results/teacher_forced/<run-id>.manifest.json \
  --output results/rollout/<run-id>.compared.manifest.json
```

The comparison command requires the same model, same input hash, non-oracle
manifests, and a positive rollout value delta before the claim ledger can clear
`rollout_beats_teacher_forced_history`.

The repo also carries a smaller synthetic recovery gate for Stage 5:

```bash
uv run python -m eval.run_local_behavior_recovery_comparison \
  --behavior-recovery-training-manifest outputs/behavior_recovery/training.manifest.json \
  --direct-training-manifest outputs/behavior_recovery_direct_sql/training.manifest.json \
  --output-dir results/behavior_recovery \
  --run-id <run-id> \
  --model-name <local-model-name>
```

That path compares `benchmark=behavior_recovery` against
`benchmark=behavior_recovery_direct_sql` on the same synthetic recovery row and
writes deltas for value accuracy, strict accuracy, and `recovery_success_rate`.
It is still a synthetic gate, not a rollout claim.

## Blog Rule

A blog sentence can make a benchmark claim only if it names one of:

- `non_oracle_generation` for deployable proxy results;
- `predicted_planner` for non-oracle planner-to-SQL results;
- `metric_dsl` for metric-intent quality and direct-SQL comparison claims;
- `oracle_planner_diagnostic` for ceiling tests;
- `pending` for work that has not been run.

If the sentence cannot be labeled with one of those modes, it is probably too
vague to publish.
