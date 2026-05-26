# Multi-Turn SQL Fine-Tuning

Fine-tune and evaluate a small specialized local model for multi-turn analytical
SQL. The research question is whether a 9B-class local model can learn the
behavior and semantic concepts needed to outperform much larger state-of-the-art
general models on multi-turn data-analysis tasks.

The project target is not "make CoSQL go up" in isolation. CoSQL is the first
small, reproducible multi-turn proxy slice. The longer benchmark direction is a
BIRD-Interact-style comparison with the same interaction protocol, SQL execution
checks, cost accounting, and larger-model baselines.

See `docs/research_goal.md` for the explicit research program, including the
fine-tuning methods this repo should compare: direct SQL SFT, planner/DSL first
then SQL, semantic-layer tuning, `MEASURE()`-preserving metric DSLs, and
behavior/recovery tuning.
See `docs/finetuning_ladder.md` for the structured training ladder that turns
those ideas into named finetuning steps, evaluation gates, and claim boundaries.
The machine-checkable cross-stage map for that ladder is
`data.finetuning_program_registry`, which writes
`docs/data_artifacts/finetuning_program_registry.json`.
The short human-readable companion is
`docs/data_artifacts/finetuning_stage_scorecard.md`, which answers the same
questions for every stage in plain language: what it learns, what it is
compared against, and what would count as a real win.
For SQL-generating method-vs-control experiments, the shared local comparison
contract now lives in `eval.local_sql_pair`. New methods in that family should
add a small `SqlPairSpec` plus a method-specific comparer, not a bespoke local
runner from scratch.
For mixed-output method-vs-control experiments, where the method side and the
control side use different scorers, the shared orchestration now lives in
`eval.local_generation_pair`. Stage 4 metric DSL uses that path.
Paired training-manifest validation now lives in `eval.training_manifest_pair`.
New comparison runners should reuse that helper instead of re-encoding stage,
benchmark, and evaluation-mode checks inline.
Paired row-identity and non-empty input checks now live in
`eval.pair_input_contract`, so same-row comparison rules are shared instead of
being reimplemented inside each runner.
For paired local `run_local_benchmark` experiments where both sides still emit
SQL through the same benchmark harness, the shared orchestration now lives in
`eval.local_benchmark_pair`. Predicted-planner SQL and semantic-proxy SQL use
that path instead of carrying their own paired local benchmark wiring.

The first metric-DSL experiment surface is documented in
`docs/metric_dsl_contract.md`, implemented in `data.metric_dsl`, and evaluated
offline through `eval.metric_dsl_eval`.
Its first finetuning-data bootstrap path is implemented in
`data.metric_dsl_dataset`, with a same-row direct-SQL control in
`data.metric_dsl_direct_sql_dataset`; both write small Stage 4 training
artifacts under `docs/data_artifacts/`. `train.finetune` can now also emit a
training manifest for those stages, so eval and comparison artifacts can be
traced back to exact prepared inputs and expected stage metadata. The offline
paired evaluation path is `eval.run_metric_dsl_comparison`, and the local
checkpoint experiment loop is `eval.run_local_metric_dsl_comparison`.
Stage 3 now has the same shape: `data.semantic_layer_dataset` writes the
semantic-aware SQL rows, `data.semantic_layer_direct_sql_dataset` writes the
same-row control without semantic-model prompt context, and
`eval.run_local_semantic_layer_comparison` generates, scores, and compares the
paired local checkpoints.
It now also has a prepared CoSQL proxy package:
`data.semantic_proxy_dataset` writes the semantic train/eval artifacts,
`data.semantic_proxy_direct_sql_dataset` writes the direct control, and
`eval.run_local_semantic_proxy_comparison` compares local checkpoints on the
same prepared dialog turns.
The predicted-planner path now also has a local checkpoint comparison loop in
`eval.run_local_predicted_planner_comparison`.
Planner supervision has a local checkpoint evaluation path in
`eval.run_local_planner_eval`.
Generated-history rollout evaluation is documented in
`docs/rollout_eval_contract.md` and implemented in `eval.rollout_eval`.

> Oracle diagnostic: the `0.890` schema-pruned result uses gold SQL-derived
> planning hints in the eval prompt. It is an upper bound for the
> schema-linking/planning layer, not a production evaluation. Production-style
> evals omit `--allow-oracle-plan`, and training rejects those rows unless
> `--allow-oracle-diagnostic-data` is passed explicitly.

## Current State

This repo is now organized around verified, runnable gates:

- BIRD-Interact/BIRD-style evaluation is the target direction; CoSQL is the
  current local proxy while that harness is built.
- Data preparation writes TRL-compatible `messages` JSONL.
- Data preparation now injects Cube-inspired semantic model hints from Spider/CoSQL `tables.json` when available.
- Training consumes prepared JSONL and supports bounded smoke tests with `--max-steps`.
- Evaluation compares base and fine-tuned models through either a local Transformers runner or an OpenAI-compatible endpoint.
- Endpoint evaluation writes a manifest that records the input hash, output hash, model, mode, command, and metrics behind each reported number.
- The claim ledger in `docs/claim_ledgers/` verifies those manifests, marks
  non-oracle CoSQL results as proxy-only, marks oracle rows as diagnostics, and
  keeps predicted-planner SQL, metric-DSL-vs-direct-SQL, hosted baselines,
  local-vs-hosted wins, and BIRD-Interact claims pending until matching artifacts
  and positive comparison deltas exist.
- Stage 6 hosted and BIRD-Interact wrappers now also require each result
  manifest's `input_sha256` to match the checked-in frozen contract artifact,
  so a comparison cannot quietly drift onto a different input slice.
- Generated-history rollout evaluation is now wired so behavior/recovery can be
  tested without teacher-forcing prior gold SQL into later turns.
- Tests cover dataset formatting, training-data validation, SQL scoring, result loading, and plotting.
- vLLM serving is verified in a separate `.venv-vllm` environment on WSL2 + RTX 5090.
- The best oracle-conditioned endpoint run is the schema-pruned 100-step LoRA adapter at `0.890` value accuracy, `0.820` strict accuracy, and `1.000` syntax accuracy on the fixed 100-turn CoSQL dev slice. That run is a diagnostic upper bound because the planning hints are derived from gold/reference SQL.
- The first semantic-context 50-step adapter is complete. It ties the plain 50-step adapter on execution accuracy (`0.420`) without semantic eval prompts.
- Evaluating that semantic adapter with semantic prompts and the promoted `semantic_grounding` prompt policy improves strict execution accuracy to `0.440`.
- The evaluator now also reports value-only execution accuracy, which ignores harmless alias differences while preserving duplicate rows and order-sensitive outputs. Under this repaired metric, semantic-50 reaches `0.630`, tying the 100-step adapter, and `minimal_executable` reaches `0.640`.

Known constraints:

- CoSQL is sourced from the official Yale/Google Drive archive, not from the stale `alpineai/cosql` Hugging Face ID.
- The accessible `jellyChiru/SParC` Hugging Face mirror is flattened to question/query rows, not full dialogs.
- BIRD mini-dev uses splits such as `mini_dev_sqlite`, not `validation`.
- System `nvcc` is still absent in WSL. PyTorch CUDA and vLLM wheels work, but source builds and FlashInfer sampler JIT need a full toolkit.
- vLLM serving is intentionally separate from the training setup. `.venv` is for Unsloth training; `.venv-vllm` is for serving.
- Qwen thinking mode must be disabled during endpoint evaluation with `chat_template_kwargs.enable_thinking=false`; otherwise generations can include reasoning prose.
- Semantic model context increases prompt length. The current semantic endpoint run shows this cost directly, so future semantic prompts need retrieval and pruning.
- DSPy-backed prompt search is available through `eval.prompt_optimize`; it can propose and score prompt variants against execution accuracy.
- A non-oracle `predicted_planner` path is now wired: lexical planner output can be written back into prepared JSONL and injected into the SQL-generation prompt without reference SQL.
- A planner-readiness report now checks the paired input before endpoint time and
  records why the current lexical planner should be improved before treating the
  predicted-planner path as a likely SQL win.
- A `MEASURE()`-preserving metric-DSL evaluator is now wired for offline
  JSONL predictions; it scores semantic intent, compiles through a semantic
  model, optionally executes compiled SQL, and writes result manifests.
- A first value-grounding label artifact is generated from the fixed CoSQL proxy
  slice. It extracts gold SQL literal predicates into auditable rows for
  current-turn values, history-carried values, and display-to-storage
  normalization failures. This is a supervision/evaluation artifact, not a
  production prompt hint.
- A first non-oracle value index is generated from the fixed-slice SQLite
  database contents. Its coverage report separates stored-value coverage from
  user-mention alias coverage, so value/entity work has a measurable next target.
- A first synthetic schema-rich fixture pack is generated under
  `docs/data_artifacts/synthetic_method_fixtures.jsonl`. It turns value
  normalization, entity resolution, grain/fanout, `MEASURE()` preservation, and
  recovery into curated rows with expected outputs before spending endpoint time.
- Semantic-layer Stage 3 now has checked-in paired training artifacts under
  `docs/data_artifacts/semantic_layer_training_rows.jsonl` and
  `docs/data_artifacts/semantic_layer_direct_sql_training_rows.jsonl`. Those
  rows are intentionally non-oracle at prompt time: the semantic path can see
  governed semantic context, but neither side can see scorer-only labels or
  future-only outputs.
- The fixed CoSQL proxy now also has checked-in Stage 3 package artifacts under
  `docs/data_artifacts/semantic_proxy*.jsonl`. Those artifacts point back to the
  non-oracle value-label and value-index summaries and keep semantic/direct row
  identity aligned at the prepared-record level.
- Local execution scoring reports both strict label-aware accuracy and value-only accuracy. Treat older single `accuracy` numbers as strict-era results unless they come from `results/rescored/`.
- Failure analysis now classifies every wrong rescored turn into actionable labels and compares adapters or prompt variants against a baseline under `plots/failure_taxonomy/`.
- Schema-link label generation and semantic prompt pruning are available through `data.prepare --include-sql-labels --prune-semantic-model`. These flags now mark produced rows as `evaluation_mode=oracle_planner_diagnostic`. On the fixed 100-turn CoSQL slice, the best oracle prompt-only pruned-label run reaches `0.850` value accuracy, and training on that oracle-labelled format reaches `0.890`.
- The end-to-end methodology, dataset roles, training strategy boundaries, and
  benchmark claim rules are documented in `docs/methodology.md`.
- The public post now has one attached codebase, a published HTML lab, and generated evidence
  assets; see `docs/blog/README.md`.

## Blog-Attached Lab

The blog post should point readers to the attached codebase and a published HTML
lab at `/labs/local-multiturn-sql-finetuning/`. The same lab can be rerun from
the source code with Marimo:

```bash
marimo edit notebooks/labs/local_multiturn_sql_lab.py
```

A portable Jupyter export remains available for readers who prefer notebooks:

```bash
jupyter lab notebooks/labs/local_multiturn_sql_lab.ipynb
```

The lab compares five fine-tuning targets: direct SQL, planner-first SQL,
semantic-layer state, `MEASURE()`-preserving DSL, and behavior/recovery tuning.
It also separates dataset roles for BIRD-Interact, BIRD mini-dev, CoSQL, SParC,
synthetic schema-rich SQL, and the tiny SQLite lab so the repo does not treat
every SQL row as interchangeable training data.
It auto-selects CUDA, MPS, or XPU when PyTorch can see an accelerator and falls
back to CPU. It should stay a portable lab, not a serving or dependency
installation guide. The public site consumes generated evidence such as
`docs/blog/generated/shareable-lab.md`,
`docs/blog/generated/dataset-role-matrix.md`,
`docs/blog/generated/lab-method-scores.md`,
`docs/blog/generated/data-artifact-contract.md`,
`docs/blog/generated/value-grounding-labels.md`,
`docs/blog/generated/synthetic-method-fixtures.md`,
`docs/blog/generated/target-evidence-matrix.md`, and
`docs/blog/generated/lab-failure-trace.md`.

Deeper evidence maintenance stays separate from the public reader path. Expensive
GPU training, vLLM serving, and local setup notes stay in scripts and docs outside
the public lab flow.

## Leakage Policy

This repo separates three different claims that are easy to blur:

| Mode | Inputs available at inference | What it can prove |
| --- | --- | --- |
| `non_oracle_generation` | Question, conversation history, schema, semantic context, and any non-oracle retrieval artifacts | A deployable text-to-SQL path can work under those inputs. |
| `oracle_planner_diagnostic` | The same inputs plus planning hints extracted from reference SQL, or semantic context pruned by those hints | An upper bound: SQL generation becomes easier when schema linking, join choice, projection shape, and duplicate policy are already solved. |
| `predicted_planner` | Planner output predicted from question, history, schema, and optional value indexes | A production-style planner-to-SQL proxy: the system creates its own plan before generating SQL. |

Any row prepared with `--include-sql-labels` or `--prune-semantic-model` is
teacher-forced by gold SQL. The code writes `uses_oracle_planning_hints`,
`semantic_context_pruned_by_oracle_labels`, `planning_label_source`, and
`evaluation_mode` fields into JSONL records so downstream training and eval
outputs carry that caveat with them. Training on those rows is still useful, but
it should be reported as learning to consume an oracle planning contract, not as
solving multi-turn SQL end to end. `train.finetune` now fails on oracle
diagnostic rows by default; pass `--allow-oracle-diagnostic-data` only when the
run name, result manifest, and writeup all label the run as a diagnostic.

The next academically valid comparison is:

1. no planner hints,
2. oracle planner hints,
3. predicted planner hints generated without reference SQL.

Only the third row supports a production claim.

## Planner Evaluation

The repo now has a planner-evaluation path before SQL generation. It treats
gold SQL-derived labels as the answer key and scores a predicted plan against
that answer key. The predicted plan must come from visible prompt inputs, not
from the reference SQL.

Prepared JSONL now carries that split explicitly. Dialog-level records include
`gold_plans`, and expanded evaluation turns expose `gold_plan` plus optional
`predicted_plan`. `gold_plan` is allowed as a scorer target; it is not allowed
as prompt context unless the run is marked `oracle_planner_diagnostic`.

The first baseline is intentionally weak and inspectable: a lexical schema
planner that reads the user-visible schema and question text, predicts relevant
tables/columns and coarse query shape, and writes per-turn planner scores:

```bash
python -m data.prepare \
  --config configs/cosql_dev_planner.yaml \
  --section eval \
  --limit 100 \
  --output data/processed/eval_cosql_dev_100.jsonl \
  --manifest-output data/processed/eval_cosql_dev_100.manifest.json

python -m eval.planner_eval \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --limit 100 \
  --predicted-prepared-output data/processed/eval_cosql_dev_predicted_planner_100.jsonl \
  --output results/planner_eval_cosql_dev_100.jsonl \
  --summary-output results/planner_eval_cosql_dev_100_summary.json
```

The repo can now generate non-oracle planner JSON with the same OpenAI-compatible
endpoint path used by SQL evaluation:

```bash
python -m eval.planner_predict \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --limit 100 \
  --model-name <planner-model> \
  --endpoint http://localhost:8000/v1 \
  --output results/planner_predictions/<run-id>.jsonl
```

Before promoting a planner prompt or DSPy program to full SQL generation, score
planner variants directly:

```bash
python -m eval.planner_optimize \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --limit 100 \
  --model-name <planner-model> \
  --endpoint http://localhost:8000/v1 \
  --output-dir results/planner_prompt_search/<run-id> \
  --dspy-proposals 2
```

This writes one JSONL file per planner variant plus `summary.csv`, ranked by
parse rate first and planner F1 after that. Malformed JSON receives no planner
credit. The scores use SQL-derived planner labels, so this is a
planner-quality screen, not a SQL execution claim.

Then score those predictions and write the matching `predicted_planner` prepared
artifact:

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

JSON planner predictions are normalized into the same plan contract, but raw
unknown fields are still scanned before prompt injection so oracle provenance
such as `gold_reference_sql` or `derived from reference sql` cannot be hidden by
normalization.

This produces:

- `gold_plan`: normalized labels extracted from reference SQL, used only for
  scoring;
- `predicted_plan`: the non-oracle planner output;
- `planner_scores`: table F1, column F1, join F1, skeleton F1, aggregation F1,
  group-by F1, selected-count match, duplicate-policy match, and a macro planner
  score.

Like SQL evaluation, planner evaluation rejects prompt records that already
contain oracle planning hints unless `--allow-oracle-plan` is passed. That keeps
the next project concrete: improve planner F1 first, then measure whether SQL
generation improves from predicted plans.

The generated `data/processed/eval_cosql_dev_predicted_planner_100.jsonl`
contains the first 100 CoSQL turns across 32 dialogs with `evaluation_mode` set
to `predicted_planner`. It is ready for endpoint SQL evaluation, but it is not
itself an execution result.

After running endpoint SQL evaluation on both the direct prepared input and the
predicted-planner prepared input, compare the manifests before claiming the
planner path helped:

```bash
python -m eval.compare_predicted_planner \
  --predicted-manifest results/predicted_planner/multiturn_sql_100_cosql_dev_predicted.manifest.json \
  --direct-manifest results/direct_sql/multiturn_sql_100_cosql_dev_direct.manifest.json \
  --output results/predicted_planner/multiturn_sql_100_cosql_dev_predicted.compared.manifest.json
```

The comparison command refuses oracle diagnostics, non-prepared manifests,
model mismatches, wrong output modes, and row-identity mismatches. The claim
ledger only clears the predicted-planner SQL execution claim when the compared
predicted-planner run beats direct SQL on value accuracy and the referenced
direct-SQL manifest is included in the ledger input.

The safer way to produce that pair is the paired runner, which preflights row
identity before spending endpoint time, runs the direct-SQL control and
predicted-planner path with the same model/scorer/database root, then writes the
comparison manifest:

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

For endpoint-free validation, run only the preflight:

```bash
python -m eval.run_predicted_planner_comparison \
  --direct-input data/processed/eval_cosql_dev_100.jsonl \
  --predicted-input data/processed/eval_cosql_dev_predicted_planner_100.jsonl \
  --output-dir results/predicted_planner \
  --run-id lexical_planner_cosql_dev_100 \
  --model-name <served-model> \
  --preflight-output docs/predicted_planner_comparison_preflight.json \
  --preflight-only
```

The tracked preflight currently shows that the direct and predicted inputs align
for 100 turns across 32 dialogs. It is a readiness artifact only; it does not
support a SQL execution claim.

Before running that endpoint pair, summarize whether the planner is ready enough
to make the endpoint spend useful:

```bash
python -m eval.planner_readiness \
  --planner-input results/planner_eval_cosql_dev_100.jsonl \
  --preflight-input docs/predicted_planner_comparison_preflight.json \
  --output docs/planner_readiness_cosql_dev_100.json
```

The current tracked report is deliberately conservative. It says the paired
direct/predicted inputs are row-matched, but the lexical planner still has
`0.790` zero-column-F1 turns, `0.820` selected-count mismatches, and `1.000`
empty projection-expression turns on the 100-turn CoSQL proxy. The recommendation
is `improve_planner_before_claim`. That is not a SQL execution score; it is the
reason to improve column linking and projection shape before making the endpoint
comparison the next public claim.

## Value Grounding Artifacts

The first concrete data-engineering artifact turns reference SQL predicates into
labels for value grounding:

```bash
python -m data.value_artifacts \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --output docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl \
  --summary docs/data_artifacts/value_grounding_labels_cosql_dev_100_summary.json \
  --manifest docs/data_artifacts/value_grounding_labels_cosql_dev_100.manifest.json
```

The current label artifact contains 106 SQL value references across 17 databases
on the fixed CoSQL proxy slice. It separates values stated in the current user
turn, values recovered from previous user turns, values carried by prior SQL,
and stored literals that are missing from user text. Those missing-text rows are
the practical seed for entity-resolution labels and alias expansion.

The labels are derived from gold/reference SQL, so they are valid as
supervision, scoring targets, and coverage diagnostics. They are not valid as
production inference context unless a non-oracle retriever or planner predicts
the same bindings from the question, history, schema, and allowed value/entity
artifacts.

Build the corresponding non-oracle value index from database contents:

```bash
python -m data.value_index \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --database-root data/raw/cosql_dataset/database \
  --output docs/data_artifacts/value_index_cosql_dev_100.jsonl \
  --summary docs/data_artifacts/value_index_cosql_dev_100_summary.json \
  --manifest docs/data_artifacts/value_index_cosql_dev_100.manifest.json \
  --labels docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl
```

The current index contains 12,661 database-derived entries across 20 CoSQL
databases. Against the gold labels, it covers 78.3% of resolved stored values but
only 75.5% of user-visible mention aliases. That gap is the next concrete
semantic-layer problem: add alias/entity expansion, then score value retrieval
before SQL generation.

## Metric DSL Evaluation

Metric-DSL evaluation is the first runnable gate for testing whether a model
should produce semantic intent before SQL. Input rows contain a predicted DSL, a
reference DSL, a semantic model, and optional reference SQL/database path:

```json
{
  "id": "metric-1",
  "generated_metric_dsl": "MEASURE(revenue) BY customer_country",
  "reference_metric_dsl": "MEASURE(revenue) BY customer_country",
  "semantic_model": {"base_table": "orders", "measures": {}, "dimensions": {}},
  "reference_sql": "SELECT ...",
  "database_path": "data/raw/metric_fixtures/store.sqlite"
}
```

Run the offline evaluator:

```bash
python -m eval.metric_dsl_eval \
  --input results/metric_dsl/<run-id>.predictions.jsonl \
  --output results/metric_dsl/<run-id>.jsonl \
  --manifest-output results/metric_dsl/<run-id>.manifest.json \
  --model-name <served-or-offline-model-name>
```

The manifest reports parse rate, compile rate, measure preservation, measure F1,
dimension F1, filter F1, database-backed compiled-SQL value accuracy, and the
semantic-model hashes used by the run. Execution accuracy is computed only over
rows with both `reference_sql` and `database_path`. A row that expands
`SUM(orders.amount)` instead of emitting `MEASURE(revenue)` fails the metric-DSL
parse gate, so it cannot be hidden by a compiled SQL score.

Before claiming DSL-first generation is better than direct SQL, run a direct-SQL
baseline on the same metric-heavy rows and compare manifests:

```bash
python -m eval.compare_metric_dsl_direct_sql \
  --metric-dsl-manifest results/metric_dsl/<run-id>.manifest.json \
  --direct-sql-manifest results/direct_sql/<run-id>.manifest.json \
  --output results/metric_dsl/<run-id>.compared.manifest.json
```

The direct baseline must use `benchmark=metric_dsl_direct_sql` and
`evaluation_mode=non_oracle_generation`. The metric-DSL and direct-SQL models may
differ, but the output rows must have matching identities and no oracle markers.
The claim ledger clears `metric_dsl_beats_direct_sql` only when the compared
metric-DSL manifest references the direct manifest, preserves `MEASURE(...)`,
covers every row with database-backed execution, and has a positive value delta.

## Generated-History Rollout

Teacher-forced CoSQL evaluation answers a narrow question: can the model produce
the current SQL when prior turns are clean reference SQL? Rollout evaluation asks
the harder multi-turn question: what happens when the model has to continue from
its own earlier SQL?

```bash
python -m eval.rollout_eval \
  --model-name multiturn-sql-100 \
  --endpoint http://127.0.0.1:8000/v1 \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --database-root data/raw/cosql_dataset/database \
  --output results/rollout/multiturn_sql_100_cosql_dev_100_rollout.jsonl
```

The runner writes `history_policy=model_generated_sql_rollout` rows and a result
manifest. The claim ledger keeps the behavior/recovery improvement claim pending
until a rollout result is compared against the same model and input under
teacher-forced history.

```bash
python -m eval.compare_rollout_history \
  --rollout-manifest results/rollout/multiturn_sql_100_cosql_dev_100_rollout.manifest.json \
  --teacher-forced-manifest results/teacher_forced/multiturn_sql_100_cosql_dev_100.manifest.json \
  --output results/rollout/multiturn_sql_100_cosql_dev_100_rollout.compared.manifest.json
```

The comparison command refuses mismatched models, mismatched input hashes, oracle
diagnostics, and non-rollout manifests.

For local checkpoint work, the repo now also has a same-checkpoint rollout path:

```bash
uv run python -m eval.run_local_rollout_comparison \
  --training-manifest outputs/behavior_recovery_proxy/training.manifest.json \
  --output-dir results/rollout \
  --run-id <run-id> \
  --model-name <local-model-name> \
  --adapter-path <adapter-dir> \
  --database-root data/raw/cosql_dataset/database
```

That runner executes both:

- teacher-forced `benchmark=prepared`
- generated-history `benchmark=prepared_rollout`

for the same checkpoint and prepared input, then writes the compared rollout
manifest through `eval.compare_rollout_history`. This is the real Stage 5
prepared-dialog gate; the synthetic recovery pair stays as the smaller diagnostic
gate before spending GPU time on multi-turn rollout.

The matching prepared training/eval package for that local rollout gate is now
checked in:

- `docs/data_artifacts/behavior_recovery_proxy_train.jsonl`
- `docs/data_artifacts/behavior_recovery_proxy_eval.jsonl`
- `docs/data_artifacts/behavior_recovery_proxy.manifest.json`

Those rows stay on the same non-oracle semantic prepared slice as the current
CoSQL proxy, but they are labeled as `training_target=behavior_recovery` so the
run manifest and rollout comparison path can treat Stage 5 as a distinct
finetuning rung instead of a generic prepared run.

The first Stage 6 benchmark artifact is also checked in:

- `docs/data_artifacts/hosted_baseline_rows.jsonl`
- `docs/data_artifacts/hosted_baseline_summary.json`
- `docs/data_artifacts/hosted_baseline.manifest.json`

This freezes the same non-oracle prepared slice that a hosted baseline must use.
The comparison wrapper is:

```bash
uv run python -m eval.run_hosted_baseline_comparison \
  --local-training-manifest outputs/<local-run>/training.manifest.json \
  --local-result-manifest results/<local-run>.manifest.json \
  --hosted-result-manifest results/<hosted-run>.manifest.json \
  --output results/<local-run>.vs_hosted.manifest.json
```

That wrapper refuses to compare a local candidate unless its training manifest
declares the prepared non-oracle protocol first, then hands off to
`eval.compare_hosted_baseline`.

The matching BIRD-Interact transfer contract is also checked in:

- `docs/data_artifacts/bird_interact_transfer_rows.jsonl`
- `docs/data_artifacts/bird_interact_transfer_summary.json`
- `docs/data_artifacts/bird_interact_transfer.manifest.json`

And the transfer comparison wrapper is:

```bash
uv run python -m eval.run_bird_interact_comparison \
  --local-result-manifest results/<local-bird-run>.manifest.json \
  --hosted-result-manifest results/<hosted-bird-run>.manifest.json \
  --output results/<local-bird-run>.vs_hosted.manifest.json
```

That wrapper is intentionally strict: both result manifests must already declare
a `bird_interact` benchmark and `non_oracle_generation` mode before the
comparison is allowed.

There is now a separate synthetic Stage 5 recovery pair for faster finetuning
iteration before endpoint rollout:

- `docs/data_artifacts/behavior_recovery_training_rows.jsonl`
- `docs/data_artifacts/behavior_recovery_direct_sql_training_rows.jsonl`
- `uv run python -m eval.run_local_behavior_recovery_comparison`

That pair stays non-oracle. It keeps the visible schema, conversation, prior
SQL, and observed empty rows, but it does not expose repair labels such as
`requires_repair_action` in the prompt. The shared scorer records both SQL
accuracy and `recovery_success_rate`, so recovery is not flattened into generic
execution accuracy.

The optional `data.prepare --manifest-output` file records dataset composition:
source counts, evaluation modes, turn formats, history policies, assistant-turn
totals, and configured dataset weights. Use it when reporting a training or eval
artifact so dataset mixing is not hidden in prose.

Current fixed-slice lexical planner baseline:

| Slice | Planner source | Rows | Dialogs | Oracle prompt rows | Macro planner score | Table F1 | Column F1 | Skeleton F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CoSQL dev first 100 turns | `lexical_schema_baseline` | 100 | 32 | 0 | 0.571 | 0.599 | 0.117 | 0.648 |

The tracked summary is `docs/planner_baseline_cosql_dev_100_summary.json`.
This is intentionally not a SQL execution result. It measures whether the
non-oracle planner can recover the answer-key plan fields before SQL generation.
The claim ledger and historical result manifests are tracked under
`docs/evidence_contract.md` and `docs/result_manifests/`.

## Stack

| Layer | Tool | Status |
| --- | --- | --- |
| Base model | `Qwen/Qwen3.5-9B` / `unsloth/Qwen3.5-9B` | HF metadata reachable |
| Training | Unsloth + TRL `SFTTrainer` | wired for JSONL SFT |
| Serving | vLLM OpenAI-compatible server | verified in `.venv-vllm` |
| Evaluation | deterministic SQL parser/result metrics | implemented |
| Planner eval | non-oracle planner prediction vs gold SQL-derived labels | implemented |
| Plotting | accuracy-vs-latency summary | implemented |
| Compute | WSL2 + RTX 5090 | PyTorch CUDA verified |

## Datasets

| Dataset | Role | Current handling |
| --- | --- | --- |
| `jellyChiru/SParC` | SQL question/query training and validation | accessible, flattened |
| `gretelai/synthetic_text_to_sql` | schema-rich SQL reinforcement | accessible |
| `birdsql/bird_mini_dev` | SQLite benchmark rows | accessible via `mini_dev_sqlite` |
| CoSQL | multi-turn dialog training and eval | official local archive under `data/raw/cosql_dataset` |

## Quick Start

```bash
source .venv/bin/activate

# Download official CoSQL if data/raw/cosql_dataset is absent.
gdown 1Y3ydpFiQQ3FC0bzdfy3groV95O_f1nXF -O data/raw/cosql_dataset.zip

# Verify the local training stack. Use --phase serving when vLLM/nvcc matter.
python scripts/verify_blackwell.py --phase training

# Prepare training data from accessible configured datasets.
python -m data.prepare --config configs/qwen35_9b_5090.yaml --limit 100

# Prepare evaluation data from CoSQL dev, SParC validation, and BIRD mini-dev.
python -m data.prepare \
  --config configs/qwen35_9b_5090.yaml \
  --section eval \
  --limit 25 \
  --output data/processed/eval.jsonl

# Validate prepared JSONL without loading GPU/model libraries.
python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train.jsonl \
  --validate-data-only

# Run a bounded fine-tuning smoke test on the 5090.
python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train.jsonl \
  --max-steps 1 \
  --report-to none
```

## Benchmark Flow

Run a local base-model benchmark directly through Transformers:

```bash
python -m eval.local_benchmark \
  --model-name unsloth/Qwen3.5-9B \
  --benchmark prepared \
  --input data/processed/eval.jsonl \
  --database-root data/raw/cosql_dataset/database \
  --limit 25 \
  --output results/qwen35_9b_base_cosql.jsonl
```

Run the same benchmark with the fine-tuned LoRA adapter:

```bash
python -m eval.local_benchmark \
  --model-name unsloth/Qwen3.5-9B \
  --adapter-path outputs/qwen35_9b_multiturn_sql_50steps/final \
  --benchmark prepared \
  --input data/processed/eval.jsonl \
  --database-root data/raw/cosql_dataset/database \
  --limit 25 \
  --output results/qwen35_9b_lora_cosql.jsonl
```

Plot a selected comparison. Use `--include` to avoid mixing old smoke runs into the same summary:

```bash
python -m eval.plot_pareto \
  --results-dir results \
  --output-dir plots/dev_100turns_lora50 \
  --include 'qwen35_9b_base_cosql_dev_100turns.jsonl' \
  --include 'qwen35_9b_lora50_cosql_dev_100turns.jsonl'
```

Optional vLLM serving is available through the checked-in script. It uses `.venv-vllm` when
that environment exists and disables the FlashInfer sampler JIT by default:

```bash
MODEL=unsloth/Qwen3.5-9B \
LORA_PATH=outputs/qwen35_9b_multiturn_sql_50steps/final \
bash serve/serve_vllm.sh
```

For the current reliable vLLM path, serve one adapter at a time. Multi-adapter
serving was attempted, but the current WSL2 + RTX 5090 setup hit cache/CUDA graph
memory pressure with several LoRAs mounted together:

```bash
CC=/home/dan/.local/bin/cc \
VLLM_USE_FLASHINFER_SAMPLER=0 \
PATH="$PWD/.venv-vllm/bin:$PATH" \
vllm serve unsloth/Qwen3.5-9B \
  --enable-lora \
  --lora-modules multiturn-sql-100=outputs/qwen35_9b_multiturn_sql_100steps/final \
  --max-lora-rank 64 \
  --max-loras 1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-num-seqs 64
```

Endpoint evaluation:

```bash
python -m eval.run_eval \
  --benchmark prepared \
  --endpoint http://127.0.0.1:8000/v1 \
  --model-name multiturn-sql-100 \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --limit 100 \
  --max-tokens 192 \
  --database-root data/raw/cosql_dataset/database \
  --output results/vllm_qwen35_9b_lora100_cosql_dev_100turns.jsonl
```

## Current Results

`plots/vllm_semantic_iterations_100turns/summary.csv`:

| Model | Execution accuracy | Syntax accuracy | Mean latency ms | Samples |
| --- | ---: | ---: | ---: | ---: |
| `multiturn-sql-100` | 0.530 | 1.000 | 392.88 | 100 |
| `multiturn-sql-128each-50` | 0.420 | 0.990 | 390.05 | 100 |
| `multiturn-sql-50` | 0.420 | 0.980 | 360.07 | 100 |
| `multiturn-sql-semantic-50` | 0.420 | 1.000 | 2285.79 | 100 |
| `unsloth/Qwen3.5-9B` | 0.370 | 1.000 | 292.79 | 100 |

Latest semantic prompt iteration, from `plots/vllm_semantic_prompt_iteration_100turns/summary.csv`:

| Model | Prompt variant | Execution accuracy | Syntax accuracy | Mean latency ms | Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| `multiturn-sql-100` | none | 0.530 | 1.000 | 392.88 | 100 |
| `multiturn-sql-semantic-50` | `semantic_grounding` | 0.440 | 1.000 | 542.19 | 100 |
| `multiturn-sql-semantic-50` | none | 0.420 | 1.000 | 2285.79 | 100 |
| `multiturn-sql-50` | none | 0.420 | 0.980 | 360.07 | 100 |
| `unsloth/Qwen3.5-9B` | none | 0.370 | 1.000 | 292.79 | 100 |

Re-scored value-aware comparison, from `plots/rescored_vllm_semantic_prompt_iteration_100turns/summary.csv`:

| Model | Prompt variant | Value accuracy | Strict accuracy | Syntax accuracy | Samples |
| --- | --- | ---: | ---: | ---: | ---: |
| `multiturn-sql-semantic-50` | `minimal_executable` | 0.640 | 0.420 | 1.000 | 100 |
| `multiturn-sql-100` | none | 0.630 | 0.530 | 1.000 | 100 |
| `multiturn-sql-semantic-50` | none | 0.630 | 0.420 | 1.000 | 100 |
| `multiturn-sql-semantic-50` | `semantic_grounding` | 0.630 | 0.440 | 1.000 | 100 |
| `multiturn-sql-50` | none | 0.610 | 0.420 | 0.980 | 100 |
| `unsloth/Qwen3.5-9B` | none | 0.590 | 0.370 | 1.000 | 100 |

Failure taxonomy comparison, from `plots/failure_taxonomy/comparison/model_error_summary.csv`:

| Run | Value accuracy | Correct | Schema link | Join path | Value grounding | Projection | History turns | Invalid SQL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-semantic-50[minimal_executable]` | 0.640 | 64 | 7 | 7 | 6 | 11 | 18 | 0 |
| `multiturn-sql-100` | 0.630 | 63 | 7 | 4 | 8 | 9 | 17 | 0 |
| `multiturn-sql-semantic-50[semantic_grounding]` | 0.630 | 63 | 8 | 5 | 6 | 11 | 19 | 0 |
| `multiturn-sql-semantic-50` | 0.630 | 63 | 8 | 4 | 7 | 11 | 18 | 0 |
| `multiturn-sql-50` | 0.610 | 61 | 11 | 3 | 6 | 9 | 19 | 2 |
| `unsloth/Qwen3.5-9B` | 0.590 | 59 | 11 | 6 | 11 | 5 | 25 | 0 |

Pairwise comparison against `unsloth/Qwen3.5-9B` shows `minimal_executable` has
the best net movement: 9 fixed turns, 4 regressed turns, and a net +5. Its fixes
come from schema linking, value grounding, and aggregation; its regressions are
projection and schema-link misses.

Oracle schema-pruned label iteration, from
`plots/failure_taxonomy/schema_pruned_projection_schemafix_comparison/model_error_summary.csv`:

| Run | Value accuracy | Strict accuracy | Interaction match | Schema link | Join path | Projection | History turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-semantic-50[schema_pruned_minimal]` | 0.850 | 0.640 | 0.59375 | 0 | 1 | 0 | 9 |
| `multiturn-sql-semantic-50[minimal_executable]` | 0.640 | 0.420 | 0.3125 | 7 | 7 | 11 | 18 |

Against `minimal_executable`, `schema_pruned_minimal` fixes 25 turns, regresses
4, and nets +21. It removes all primary schema-link and projection failures on
the same 100 CoSQL turns. This is an oracle-label diagnostic result because the
labels are extracted from gold SQL.

Oracle schema-pruned labelled training, from
`plots/failure_taxonomy/schema_pruned_trained100_comparison/model_error_summary.csv`:

| Run | Value accuracy | Strict accuracy | Interaction match | Schema link | Join path | Projection | History turns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-schema-pruned-100[schema_pruned_minimal]` | 0.890 | 0.820 | 0.75000 | 0 | 0 | 0 | 5 |
| `multiturn-sql-semantic-50[schema_pruned_minimal]` | 0.850 | 0.640 | 0.59375 | 0 | 1 | 0 | 9 |
| `multiturn-sql-semantic-50[minimal_executable]` | 0.640 | 0.420 | 0.31250 | 7 | 7 | 11 | 18 |

The new trained adapter fixes 28 `minimal_executable` failures, regresses 3,
and nets +25. The remaining failures are concentrated in value grounding
(`7`), execution errors (`3`), and grain/fanout (`1`), which makes non-oracle
planning, value canonicalization, alias-role validation, and adversarial
fixtures the next bottlenecks.

`plots/vllm_semantic_iterations_100turns/dialog_summary.csv`:

| Model | Dialog execution accuracy | Dialog syntax accuracy | Interaction match rate | Dialogs | Turns |
| --- | ---: | ---: | ---: | ---: | ---: |
| `multiturn-sql-100` | 0.515 | 1.000 | 0.15625 | 32 | 100 |
| `multiturn-sql-128each-50` | 0.389 | 0.995 | 0.12500 | 32 | 100 |
| `multiturn-sql-50` | 0.382 | 0.977 | 0.09375 | 32 | 100 |
| `multiturn-sql-semantic-50` | 0.384 | 1.000 | 0.09375 | 32 | 100 |
| `unsloth/Qwen3.5-9B` | 0.347 | 1.000 | 0.06250 | 32 | 100 |

Interpretation:

- The schema-pruned 100-step adapter is the best oracle-conditioned endpoint
  result, but its eval path still uses gold SQL-derived planning labels.
- The semantic 50-step adapter improves syntax reliability and reaches `0.440` with semantic eval prompts plus `semantic_grounding`.
- After value-aware re-scoring, semantic-50 is competitive with the 100-step adapter and `minimal_executable` is the best prompt variant on the current 100-turn slice.
- Oracle schema-link labels plus prompt pruning are now the strongest diagnostic
  path, moving the semantic adapter to `0.850` value accuracy and the labelled
  100-step adapter to `0.890` without reducing syntax accuracy.
- Semantic context should become a governed, pruned artifact rather than an exhaustive prompt dump.

## Verification

```bash
ruff check .
pytest -q
uv pip check --python .venv/bin/python
python -m compileall -q data/prepare.py eval train scripts tests
```

Re-score existing results without regenerating model output:

```bash
python -m eval.rescore_results \
  --input results/prompt_search_semantic50_semantic_grounding_100/semantic_grounding.jsonl \
  --output results/rescored/semantic_grounding.jsonl
```

Classify failures and compare prompt or adapter regressions:

```bash
python -m eval.classify_errors \
  --input results/rescored/minimal_executable.jsonl \
  --output results/classified/minimal_executable.jsonl \
  --summary plots/failure_taxonomy/minimal_executable.csv

python -m eval.compare_failures \
  --input results/classified/vllm_qwen35_9b_base_cosql_dev_100turns.jsonl \
  --input results/classified/vllm_qwen35_9b_lora100_cosql_dev_100turns.jsonl \
  --input results/classified/minimal_executable.jsonl \
  --output-dir plots/failure_taxonomy/comparison \
  --baseline 'unsloth/Qwen3.5-9B'
```

Generate the oracle schema-pruned labelled eval slice and run the promoted
prompt:

```bash
python -m data.prepare \
  --config configs/qwen35_9b_5090.yaml \
  --section eval \
  --limit 100 \
  --output data/processed/eval_100_each_semantic_pruned_labels.jsonl \
  --include-sql-labels \
  --prune-semantic-model \
  --strict

# Oracle diagnostic only: eval prompt contains labels extracted from reference SQL.
python -m eval.prompt_optimize \
  --benchmark prepared \
  --endpoint http://127.0.0.1:8000/v1 \
  --model-name multiturn-sql-semantic-50 \
  --input data/processed/eval_100_each_semantic_pruned_labels.jsonl \
  --limit 100 \
  --max-tokens 192 \
  --database-root data/raw/cosql_dataset/database \
  --output-dir results/prompt_search_schema_pruned_projection_schemafix_100 \
  --variants configs/prompt_variants/schema_pruned_projection.json \
  --allow-oracle-plan \
  --dspy-proposals 2
```

Prompt optimization smoke:

```bash
python -m eval.prompt_optimize \
  --benchmark prepared \
  --endpoint http://127.0.0.1:8000/v1 \
  --model-name multiturn-sql-semantic-50 \
  --input data/processed/eval_100_each_semantic.jsonl \
  --limit 30 \
  --database-root data/raw/cosql_dataset/database \
  --output-dir results/prompt_search_semantic50_limit30 \
  --dspy-proposals 2
```

The current WSL environment verifies PyTorch CUDA on RTX 5090, but
`scripts/verify_blackwell.py --phase serving` reports missing system `nvcc`; the `.venv-vllm`
wheel install is nevertheless sufficient for the verified endpoint runs above when FlashInfer sampler JIT is disabled.

Latest local smoke evidence:

- 5-step Unsloth LoRA training completed on RTX 5090 and wrote `outputs/qwen35_9b_multiturn_sql/final`.
- A 50-step LoRA run on 192 mixed SQL-chat examples completed on the RTX 5090 and wrote `outputs/qwen35_9b_multiturn_sql_50steps/final`.
- On a 100-turn CoSQL-dev local benchmark spanning 32 dialogs, base Qwen scored 0.360 per-turn execution accuracy and the 50-step adapter scored 0.420.
- Dialog-level aggregation over the same 100 turns reports base dialog execution accuracy 0.337 and interaction-match rate 0.0625; the 50-step adapter reports dialog execution accuracy 0.382 and interaction-match rate 0.09375.
- The filtered plot command above was verified and wrote `plots/dev_100turns_lora50/summary.csv` plus `plots/dev_100turns_lora50/dialog_summary.csv`.
- vLLM endpoint iteration testing improved from base `0.370` to `0.530` with the 100-step adapter on the same 100-turn CoSQL-dev slice.
- Semantic-context 50-step endpoint testing completed at `0.420` execution accuracy and `1.000` syntax accuracy on the same slice.
- Semantic prompt iteration completed at `0.440` execution accuracy with `semantic_grounding` on the same slice.
- Value-aware re-scoring completed under `results/rescored/`, showing semantic-50 at `0.630` and `minimal_executable` at `0.640`.
- Failure classification completed under `results/classified/`; comparison reports are in `plots/failure_taxonomy/comparison/`.
- Oracle schema-pruned DSPy label evaluation completed under `results/prompt_search_schema_pruned_projection_schemafix_100/`, with comparison reports in `plots/failure_taxonomy/schema_pruned_projection_schemafix_comparison/`.

See `docs/5090_benchmark_report.md` for the current research notes, commands, observed results, and remaining quality work.
See `docs/multiturn_semantic_modeling_research.md` for the deeper research pass and the semantic-modeling fine-tuning changes.
See `docs/blog/` for the rewritten blog series, including `06_data_engineering_for_multiturn_sql_eval.md`.
