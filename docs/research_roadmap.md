# Research Roadmap

This is the canonical long-term roadmap for the multi-turn SQL fine-tuning
program. It replaces the old gate-heavy day-to-day direction with smaller,
row-matched method comparisons.

Last audited: 2026-06-02 after the Checkpoint 3 endpoint evaluation,
generated-history rollout, clean-holdout failure analysis, Checkpoint 5 full
structured query-brief training and clean-holdout comparison, and Checkpoint 6
pruned semantic value-retrieval endpoint comparison.

Checkpoint status legend:

- `[x]` complete for the current repo state.
- `[~]` in progress with useful artifacts or code present, but the checkpoint's
  evidence requirement is not fully satisfied.
- `[ ]` pending; no sufficient current evidence exists yet.

Inconclusive checkpoint evidence should stay visible. When a result is
well-formed but does not cleanly support promotion, regression, or a benchmark
claim, record the artifact, row scope, blocker, and remaining uncertainty. These
records are inputs to future writeups: they explain why the roadmap paused,
rerouted, or redesigned a method without overstating either success or failure.

Current checkpoint progress:

- `[x]` Checkpoint 0: Freeze The Current State.
- `[x]` Checkpoint 1: Simplify The Research Loop.
- `[x]` Checkpoint 2: Establish Honest Dataset Roles.
- `[x]` Checkpoint 3: Rebuild Baselines At Real Scale.
- `[x]` Checkpoint 4: Let Failure Analysis Choose Methods.
- `[x]` Checkpoint 5: Structured Query Brief SFT.
- `[x]` Checkpoint 6: Semantic Layer And Value Grounding.
- `[~]` Checkpoint 7: Metric DSL.
- `[~]` Checkpoint 8: Generated-History Recovery.
- `[ ]` Checkpoint 9: Hosted And Target Benchmark Transfer.

Audit the current checkpoint statuses without running GPU training or endpoints:

```bash
uv run --active --no-sync python -m eval.roadmap_status --format markdown
```

The publishable benchmark claim is simple:

> A local fine-tuned method must beat the right direct-SQL control on clean
> held-out multi-turn SQL tasks, then compare against hosted baselines under the
> same protocol.

The current repo is not at a hosted benchmark claim yet. It now has a full
direct-SQL local control baseline for the configured CoSQL proxy and clean
holdout, a Checkpoint 5 structured query-brief clean-holdout win, and one
narrow clean-holdout semantic value-retrieval win under the current comparer.
Other structured method arms still need same-row clean-holdout wins. Many
method artifacts remain intentionally small: 2 metric-DSL rows, 1
behavior-recovery row, and 5 synthetic fixtures.

Supported non-oracle claims are still scoped. The full direct-SQL LoRA improves
over its base-model control on the configured local rows: proxy value/strict
accuracy moves from `0.223` to `0.355`, clean-holdout value/strict accuracy
moves from `0.218` to `0.349`, and generated-history rollout value/strict
accuracy moves from `0.049` to `0.090`. The `0.890` schema-pruned result remains
an oracle diagnostic only because reference SQL-derived planning hints enter
that older run.

## Research Anchors

These benchmarks and papers define the direction, but this repo should not
borrow their claims until it runs matching protocols:

- BIRD: https://arxiv.org/abs/2305.03111
- BIRD-Interact: https://arxiv.org/abs/2510.05318 and
  https://bird-interact.github.io/
- LiveSQLBench: https://livesqlbench.ai/
- Spider 2.0: https://spider2-sql.github.io/
- CoSQL: https://yale-lily.github.io/cosql
- SParC: https://arxiv.org/abs/1906.02285
- ReViSQL clean-data/RLVR direction: https://arxiv.org/abs/2603.20004

## Checkpoint 0: Freeze The Current State

Status: `[x]` complete for current `main`. The inventory is
`docs/current_research_inventory.md`.

The first checkpoint is historical cleanup, not new modeling.

- Mark CoSQL dev 100 as `proxy_dev_seen`: an inspected, reproducible proxy
  slice, not a clean benchmark.
- Inventory current datasets, prepared rows, runs, adapters, results, and claims.
- Remove checked-in gate ledgers, historical result-manifest snapshots, and
  generated blog evidence from the active source tree. Keep run-specific outputs
  under `results/` unless they are only old gate reports.
- Remove the old ledger/readiness workflow from the main research loop. It
  should not decide what researchers do each day.
- Record that the raw local CoSQL data is much larger than the current scored
  slice: 2,159 train dialogs / 7,343 train turns and 293 dev dialogs /
  1,007 dev turns.
- Record that `data/processed/train.jsonl` is absent or empty in the checked-in
  tree and should not be treated as evidence of full-scale training.

The output of this checkpoint is `docs/current_research_inventory.md`, which
names the actual rows, manifests, adapters, outputs, and claims currently
available.

## Checkpoint 1: Simplify The Research Loop

Status: `[x]` complete for the current roadmap interface. The old gate workflow
and generated ledgers are removed, the active guardrails are documented, and
the compact experiment registry is `configs/experiments.yaml`.

Replace the gate-first workflow with four required guardrails:

- split integrity;
- prompt leakage prevention;
- row identity matching;
- run manifests.

Everything else should become optional reporting until the empirical base is
larger. Keep contract tests for leakage and scoring correctness. Stop expanding
tests that only prove docs/config freshness.

The compact experiment registry is `configs/experiments.yaml`. Each run should
name:

- hypothesis;
- train split;
- validation split;
- test split;
- model and adapter path;
- method;
- scorer;
- output path.

This registry should answer "what did we test and why?" before any generated
status table or runbook checklist tries to rank the method.

## Checkpoint 2: Establish Honest Dataset Roles

Status: `[x]` complete for the current split interface. Frozen split manifests
live under `data/splits/`; external target manifests are explicit pending
records until those rows are available.

The explicit split manifests are:

- `data/splits/cosql_train_v1.json`;
- `data/splits/cosql_dev_100_proxy_seen_v1.json`;
- `data/splits/cosql_dev_clean_holdout_v1.json`;
- `data/splits/synthetic_method_fixtures_v1.json`;
- `data/splits/sparc_context_transfer_pending_v1.json`;
- `data/splits/bird_mini_dev_pending_v1.json`;
- `data/splits/bird_interact_lite_pending_v1.json`.

Treat existing CoSQL dev 100 as `proxy_dev_seen`. Create at least one clean
local holdout slice that is not used for prompt search, method selection,
training, manual diagnostics, or evidence iteration.

Dataset roles should be explicit:

- `train`: allowed for supervised training and label extraction.
- `validation`: allowed for method development and failure analysis.
- `proxy_dev_seen`: allowed for continuity with historical evidence, but not for
  clean benchmark claims.
- `clean_local_holdout`: reserved for final local method comparisons.
- `external_target`: BIRD-Interact, LiveSQLBench, Spider 2.0, or similar
  transfer targets run only after the local loop is stable.

For Spider 2.0 and BIRD-style released evaluation data, do not train on gold SQL
that is intended for evaluation or prompt design. If gold SQL is available only
for scoring, keep it scorer-side.

## Checkpoint 3: Rebuild Baselines At Real Scale

Status: `[x]` complete for the configured local direct-SQL control. Split-based
non-oracle inputs exist, the full direct-SQL LoRA adapter has been trained
locally, base and LoRA endpoint evaluations have run on the proxy and clean
holdout rows, and generated-history rollouts exist for both model roles.

Run direct SQL base and direct SQL LoRA on full available non-oracle training
data, not 64-row or 128-row samples. Direct SQL is the stable control for every
structured method.

Report at least:

- per-turn value accuracy;
- strict accuracy;
- interaction match;
- generated-history rollout accuracy;
- latency;
- cost.

The direct-SQL baseline must be boring, durable, and row-matched. Every planner,
semantic-layer, metric-DSL, and recovery comparison should use the same row IDs,
scorer, oracle policy, and manifest shape.

Current Checkpoint 3 artifacts:

- `data.prepare_split` materializes prepared JSONL from frozen split manifests
  without oracle planning hints or oracle-pruned semantic context.
- `configs/direct_sql_full_non_oracle.yaml` names the direct-SQL full-control
  adapter and prepared train/proxy/holdout paths.
- `scripts.direct_sql_full_control` prints or runs the ordered Checkpoint 3
  workflow from that config, including split prep, data validation, full LoRA
  training, row-matched endpoint evals, generated-history rollouts,
  clean-holdout failure analysis, and the final artifact audit.
- `eval.run_eval` result manifests preserve split provenance, latency, token
  usage, and estimated generation cost when the endpoint returns usage fields.
- `eval.checkpoint3_artifact_audit` verifies the required prepared inputs,
  base/LoRA proxy and clean-holdout result manifests, generated-history rollout
  manifests, non-oracle policy, required metrics, and base/LoRA row identity
  matching before this checkpoint can move to `[x]`.
- `eval.roadmap_status` summarizes the current checkpoint statuses from the
  experiment registry and Checkpoint 3 artifact audit without running training
  or endpoint-backed evaluation.
- `docs/training_runs/direct_sql_full_lora_20260531.json` records the completed
  local full direct-SQL LoRA training artifact and its claim boundary.
- `docs/training_runs/direct_sql_full_eval_20260531.json` records the endpoint
  eval, rollout, and failure-analysis metrics while keeping the multi-megabyte
  generated results under ignored `results/`.
- `configs/experiments.yaml` marks `direct_sql_full_non_oracle_control` as
  `checkpoint3_complete_direct_sql_control`.

Latest local Checkpoint 3 run state from 2026-05-31:

- A non-`/tmp` GPU worktree prepared the direct-SQL train, proxy, and clean
  holdout JSONL files from the frozen split manifests.
- The prepared train/proxy validation path loaded 2,159 train rows and 100
  proxy rows; the train/clean-holdout validation path loaded 2,159 train rows
  and 193 clean-holdout rows.
- The RTX 5090 LoRA training path reached the active training loop only after
  running unsandboxed with `CC=/home/dan/.local/bin/cc` and Zig cache env vars
  for Triton runtime compilation.
- Full LoRA training completed for 1,620 steps / 3 epochs and saved the final
  adapter at `outputs/experiments/direct_sql_full/final`. The final adapter
  weights SHA-256 is
  `906347d66bb05c1668a8cc82979433c1c1b2056adfcd48f31ae0026a4a7ecdaf`.
- Training eval loss is language-model loss on the proxy eval split, not SQL
  execution accuracy: `0.5369` at step 1000, `0.5629` at step 1500, and
  `0.6191` at step 1620.
- Base endpoint evaluation scored `0.223` value/strict accuracy on 327 proxy
  turns and `0.218` on 680 clean-holdout turns.
- LoRA endpoint evaluation scored `0.355` value/strict accuracy on the same
  proxy turns and `0.349` on the same clean-holdout turns.
- Generated-history rollout scored `0.049` value/strict accuracy for base and
  `0.090` for LoRA on the same 680 clean-holdout turns.
- `uv run --active --no-sync python -m eval.checkpoint3_artifact_audit` passes
  when run in the local worktree that contains the ignored prepared data and
  result manifests.

Print the complete Checkpoint 3 workflow before running anything expensive:

```bash
uv run --active --no-sync python -m scripts.direct_sql_full_control
```

Run only the cheap preparation and validation stages first:

```bash
uv run --active --no-sync python -m scripts.direct_sql_full_control \
  --stage prepare \
  --stage validate \
  --source-root /home/dan/docs/multiturn-sql-finetuning \
  --run
```

When resuming GPU LoRA training in a fresh worktree, use the runner's explicit
compiler flags and run it unsandboxed so WSL CUDA is visible:

```bash
uv run --active --no-sync python -m scripts.direct_sql_full_control \
  --stage train \
  --cc /home/dan/.local/bin/cc \
  --zig-cache-dir /tmp/zig-cache \
  --train-report-to none \
  --run
```

Checkpoint 3 is complete for the configured local baseline. Re-run it when the
model, split, scorer, endpoint, or adapter changes.

Run the endpoint-backed stages after the base and LoRA endpoints are available:

```bash
uv run --active --no-sync python -m scripts.direct_sql_full_control \
  --stage eval \
  --stage rollout \
  --base-model-name <served-base-model> \
  --lora-model-name <served-lora-model> \
  --database-root data/raw/cosql_dataset/database \
  --run
```

Audit the full evidence contract with:

```bash
uv run --active --no-sync python -m scripts.direct_sql_full_control \
  --stage audit \
  --run
```

The audit is expected to pass only in a worktree that has the ignored prepared
inputs and result manifests listed in `configs/direct_sql_full_non_oracle.yaml`.

## Checkpoint 4: Let Failure Analysis Choose Methods

Status: `[x]` complete for the first clean-holdout failure-analysis pass.
`eval.clean_holdout_failure_analysis` classified the Checkpoint 3 base and LoRA
clean-holdout outputs and wrote same-row error summaries.

Do not add another method arm because the ladder has a slot for it. Classify
failures on clean validation first, then choose the method whose hypothesis
matches the observed error concentration.

Use this mapping:

- planner/schema linking: wrong tables, columns, joins, projection shape,
  aggregation, grouping, or duplicate policy;
- semantic value grounding: display-to-storage values, aliases, entities, and
  context-carried values;
- metric definitions: governed measures, dimensions, filters, grain, and
  `MEASURE(...)` preservation;
- recovery: empty result repair, syntax repair, invalid-column repair, and
  continuation after earlier generated SQL.

Tiny synthetic wins should not be scaled until they also move a real validation
slice. Synthetic fixtures are useful for isolating failure modes, not for
declaring a benchmark improvement.

The 2026-05-31 clean-holdout analysis still points first at planner/schema
linking, then value grounding and metric definitions. LoRA reduced many errors
but did not make recovery the dominant next method family:

- base method hints: planner/schema linking `420`, semantic value grounding
  `109`, metric definitions `88`, recovery `2`;
- LoRA method hints: planner/schema linking `369`, semantic value grounding
  `67`, metric definitions `61`, recovery `1`;
- base correct turns: `148` of 680; LoRA correct turns: `237` of 680.

Regenerate the failure-analysis artifact with:

```bash
uv run --active --no-sync python -m scripts.direct_sql_full_control \
  --stage analysis \
  --run
```

## Checkpoint 5: Structured Query Brief SFT

Status: `[x]` complete for the current local promotion policy. This checkpoint
is now reset around training data and benchmark comparison, not oracle-planner
gates. The old predicted-planner branch is deprecated and removed from active
evidence because it overfit the workflow to SQL-derived planner labels and did
not beat the direct-SQL control.

The active hypothesis is simpler:

> A model trained to write a compact, visible query brief before SQL can beat
> the direct-SQL finetuned control on the same multi-turn benchmark rows.

The query brief is not private chain-of-thought and not oracle reasoning. It is
a supervised, visible artifact with a small stable shape:

- user intent;
- entities and values;
- metrics or measures;
- filters;
- grouping and grain;
- joins or table families;
- final answer shape.

Training-split reference SQL may be used to build brief supervision and scoring
labels. Clean-holdout reference SQL, expected rows, future turns, gold plans,
gold DSL, and repair labels remain scorer-side only. The clean-holdout prompt
may contain only production-visible context: question, history, schema,
semantic artifacts, retrieved database values, and the model's own generated
brief if the run emits one.

Checkpoint 5 promotion no longer depends on planner F1 or the old
planner-readiness gate. A structured-brief method promotes only when a
row-matched comparison shows:

- direct-SQL control and structured-brief arm use identical clean-holdout row
  IDs;
- both arms use the same scorer, database root, oracle policy, and manifest
  shape;
- the structured-brief arm has a positive value-accuracy delta versus direct
  SQL;
- strict accuracy, syntax rate, interaction match, latency, and cost are
  reported alongside the primary value metric.

The old predicted-planner detour is deprecated and removed from active
evidence. It did not produce a promotable SQL result, and keeping its command
wrappers and JSON snapshots made the old gate-first workflow look canonical.
Checkpoint 5 now treats query decomposition as trainable structured-brief data,
not as a separate planner-readiness gate.

Current Checkpoint 5 evidence from 2026-06-02:

- `data.structured_brief_training_rows` projects the full CoSQL train split
  into 7,343 structured-brief SFT rows across 2,159 dialogs and 140 databases,
  with `split_role=train`, no current reference SQL or scorer labels in the
  model prompt, and the brief plus SQL visible only as the assistant target.
- The structured-brief adapter trained for the matched 1,620-step budget on the
  RTX 5090 and saved the final adapter at
  `outputs/experiments/structured_brief_sql/full_20260602/final`.
- The structured clean-holdout input covers 193 dialogs / 680 scored turns
  across 20 databases and keeps clean-holdout reference SQL, expected rows,
  gold plans, and scorer labels out of the model prompt.
- The same-row clean-holdout comparison beat the direct-SQL LoRA control:
  structured brief value accuracy `0.576` versus direct SQL `0.349`, strict
  accuracy `0.541` versus `0.349`, value delta `+0.228`, and strict delta
  `+0.193` on 680 comparable turns.
- The comparer marked the structured-brief result promotion-ready with no
  blockers under the current local policy. The compact checked-in comparison
  claim is
  `docs/training_runs/structured_brief_clean_holdout_comparison_20260602.json`;
  the full training and endpoint summary is
  `docs/training_runs/structured_brief_full_20260602.json`.

The result is still a local clean-holdout method win, not a hosted benchmark
claim. It also has a tradeoff: syntax accuracy is lower than direct SQL
(`0.801` versus `0.999`), and mean generation latency and token use are higher
because the model emits a visible brief before SQL. Future work should preserve
the value-accuracy gain while improving SQL well-formedness and output economy.

## Checkpoint 6: Semantic Layer And Value Grounding

Status: `[x]` complete for the current local semantic value-retrieval promotion
policy. Value labels, a non-oracle value index, semantic retrieval inputs,
alias/column context inputs, negative unpruned endpoint evidence, pruning
evidence, and a pruned clean-holdout endpoint pair exist. The pruned same-row
semantic SQL comparison beat direct SQL by a very narrow margin on the clean
holdout.

Replace mechanically generated semantic hints with versioned semantic artifacts:

- entities;
- dimensions;
- measures;
- joins;
- grain;
- aliases;
- value indexes.

Gold value labels stay scorer-side unless the run is explicitly diagnostic.
Measure retrieval coverage first, then measure SQL accuracy delta versus direct
SQL on the same rows.

The value index should come from database contents and governed metadata that
would be available at inference time. Gold SQL-derived labels are allowed for
supervision, diagnostics, and scoring, but not as production prompt context.

`eval.compare_semantic_value_retrieval` is the current promotion audit for this
checkpoint. It keeps semantic value-retrieval claims behind same-row
non-oracle direct-SQL comparison, database-derived value-index provenance, a
clean-holdout split role, enough comparable rows, a positive value-accuracy
delta, and no strict-accuracy regression. A proxy comparison can guide
iteration, but it is not a clean-holdout semantic win.

Latest Checkpoint 6 evidence from 2026-06-02:

- A clean-holdout value index was regenerated from database contents for the
  same 20 CoSQL dev databases used by `cosql_dev_clean_holdout_v1`; it produced
  12,661 non-oracle value entries.
- Semantic value-retrieval prepared input was generated for all 193
  clean-holdout dialogs / 680 assistant turns. It matched 3,611 value entries
  across 510 user turns and 155 dialogs using only user-visible text seen so far.
- `eval.run_semantic_value_retrieval_comparison --preflight-only` verified the
  direct-SQL and semantic inputs are row-identity matched and endpoint-pair
  ready on 680 turns, 193 dialogs, and 20 databases.
- The full clean-holdout endpoint pair then ran on 680 comparable turns with
  `multiturn-sql-semantic-50`. Direct SQL reached `0.650` value accuracy and
  `0.553` strict accuracy; semantic value retrieval reached `0.635` value
  accuracy and `0.540` strict accuracy. The semantic deltas were negative:
  `-0.0147` value and `-0.0132` strict.
- The comparer marked semantic promotion not ready with blockers:
  value delta must be positive, and strict delta must not regress. Evidence
  files are `docs/training_runs/semantic_value_clean_holdout_preflight_20260602.json`
  and `docs/training_runs/semantic_value_clean_holdout_full_20260602.json`.
- A paired failure analysis found 43 direct-only value-correct rows and 33
  semantic-only value-correct rows. All value-score flips occurred on rows with
  value-retrieval context; the 170 rows without retrieved values had no
  direct-only or semantic-only value flips. The evidence file is
  `docs/training_runs/semantic_value_regression_diagnosis_20260602.json`.
- Semantic value-retrieval input building now defaults to a pruned policy:
  current-turn-only retrieval, at most 4 matches per turn, and short ambiguous
  aliases pruned while numeric aliases remain eligible. This reduced the
  clean-holdout semantic input from 3,611 matched values across 510 turns to 840
  matched values across 340 turns.
- `eval.run_semantic_value_retrieval_comparison --preflight-only` verified that
  the pruned semantic input is row-identity matched and endpoint-pair ready on
  the same 680 turns, 193 dialogs, and 20 databases. The evidence file is
  `docs/training_runs/semantic_value_pruned_preflight_20260602.json`.
- The pruned full clean-holdout endpoint pair ran on the same 680 comparable
  turns with `multiturn-sql-semantic-50`. Direct SQL reached `0.651` value
  accuracy and `0.554` strict accuracy; pruned semantic value retrieval reached
  `0.653` value accuracy and `0.557` strict accuracy. The semantic deltas were
  `+0.00147` value and `+0.00294` strict, and
  `semantic_value_retrieval_promotion_ready=true`.
- The paired flip analysis shows why this should stay a narrow local claim:
  32 semantic-only value-correct rows, 31 direct-only value-correct rows, 412
  rows both value-correct, and 205 rows both value-wrong. The evidence file is
  `docs/training_runs/semantic_value_pruned_full_20260602.json`.

Checkpoint 6 is complete for the current local semantic value-retrieval
promotion policy. Future work should replicate or stress-test this narrow win,
then decide whether the value-retrieval context is worth training against or
only using as an inference-time prompt augmentation.

## Checkpoint 7: Metric DSL

Status: `[~]` in progress. Parser/evaluator code, two-row fixtures, and a
clean-holdout metric-shaped candidate slice exist, but prompt-only and 5-step
evidence are negative diagnostics rather than a DSL method win.

Expand beyond the current 2-row DSL fixture before training a serious adapter.
Parse rate and compile rate are prerequisites, not wins.

A metric-DSL method can support a benefit claim only when:

- DSL predictions are generated from prompt-visible inputs;
- generated DSL parses;
- generated DSL compiles through the same semantic model;
- governed `MEASURE(...)` intent is preserved;
- compiled DSL SQL beats direct SQL on the same metric-heavy held-out tasks.

If the DSL compiles but loses to direct SQL, the result is still useful: it says
metric intent was represented but the method did not improve SQL outcomes.

`eval.compare_metric_dsl_direct_sql` is the current promotion audit for this
checkpoint. It keeps Metric DSL claims behind same-row direct-SQL comparison,
database-backed compiled-SQL scoring, a clean-holdout split role, enough
comparable rows, full parse and compile rates, full `MEASURE(...)`
preservation, a positive value-accuracy delta, no strict-accuracy regression,
and non-oracle semantic-model provenance. Synthetic or tiny fixture evidence
can diagnose the contract, but it cannot promote the method.

Current Checkpoint 7 evidence from 2026-06-02:

- `data.metric_dsl_clean_holdout_readiness` selects metric-shaped turns from
  the prepared CoSQL clean holdout without exposing the current reference SQL,
  gold plan, or future turns in the Metric DSL prompt.
- The readiness pass found 313 candidate turns across 141 dialogs and 19
  databases with `split_role=clean_local_holdout`. Signal counts are:
  aggregation `233`, group-by `87`, having `38`, order-by `114`, and limit
  `101`.
- The compact readiness summary is
  `docs/training_runs/metric_dsl_clean_holdout_readiness_20260602.json`; the
  generated candidate JSONL remains under ignored `data/processed/metric_dsl/`,
  with provenance recorded in
  `docs/data_artifacts/metric_dsl_clean_holdout_candidates.manifest.json`.
- This is readiness evidence only. Every candidate is still blocked on
  structured scorer-side gold Metric DSL labels. After those labels exist, the
  next step is same-row Metric DSL and direct-SQL prediction generation followed
  by `eval.run_metric_dsl_comparison` and the promotion audit.

## Checkpoint 8: Generated-History Recovery

Status: `[~]` in progress. Rollout evaluators and one-row recovery diagnostics
exist; generated-history recovery has not beaten direct SQL on multi-dialog
rollout.

Move recovery evaluation from one synthetic row to multi-dialog rollout. Later
turns must see generated SQL and observed results, not prior gold SQL.

Recovery wins only if a recovery-tuned adapter beats direct SQL under
generated-history rollout. Teacher-forced history remains diagnostic: it shows
how much clean previous SQL was hiding failures, but it is not the recovery win
condition.

The minimum rollout manifest should record:

- dialog IDs and turn IDs;
- generated SQL from prior turns;
- observed result summaries from prior turns;
- scorer-visible reference SQL and expected rows;
- whether each later turn saw generated or teacher-forced history;
- value-only, strict, syntax, and interaction-level metrics.

## Checkpoint 9: Hosted And Target Benchmark Transfer

Status: `[ ]` pending. Hosted and BIRD-Interact-style transfer should wait until
a local method beats direct SQL on a clean local holdout.

Run hosted baselines only after a local method beats direct SQL on a clean local
holdout. The first real external target should be BIRD-Interact Lite or
LiveSQLBench, with Spider 2.0 and BIRD-style data used only under their allowed
protocols.

The final claim requires:

- same input rows;
- same scorer and test cases;
- same oracle policy;
- no reference SQL, expected rows, gold plans, gold DSL, repair labels, or
  future turns in production prompts;
- hosted and local manifests with model, prompt, latency, and cost metadata;
- a positive local delta against the hosted baseline under the same protocol.

## Future Repo Interfaces

The reset should converge on these interfaces:

- `docs/research_roadmap.md`: this checkpointed roadmap.
- `docs/current_research_inventory.md`: audited current-state inventory for
  Checkpoint 0.
- `configs/experiments.yaml`: compact registry for experiment hypotheses and
  run definitions.
- `data/splits/*.json`: dataset roles and frozen row IDs.
- `results/runs/<run_id>/manifest.json`: one standard manifest shape for every
  model run.

A run manifest should include:

```json
{
  "run_id": "YYYYMMDD-method-split-model",
  "hypothesis_id": "direct_sql_full_cosql",
  "dataset_role": "clean_local_holdout",
  "train_split_id": "cosql_train_v1",
  "validation_split_id": "cosql_val_v1",
  "test_split_id": "cosql_clean_holdout_v1",
  "row_ids_sha256": "...",
  "model": "unsloth/Qwen3.5-9B",
  "adapter": "outputs/experiments/<run>/final",
  "method": "direct_sql",
  "oracle_policy": "non_oracle_generation",
  "scorer": "value_and_strict_execution_v1",
  "outputs": {
    "generations": "results/runs/<run_id>/generations.jsonl",
    "scores": "results/runs/<run_id>/scores.jsonl"
  }
}
```

## Test Plan

Add tests for durable research guardrails:

- split-integrity tests proving train, validation, proxy, and test row IDs do
  not overlap;
- prompt-leakage tests for reference SQL, expected rows, future turns, gold
  plans, gold DSL, and repair labels;
- row-matched comparison tests for direct SQL versus planner, semantic, DSL, and
  recovery methods;
- scorer tests for value-only accuracy, strict accuracy, ordering, duplicates,
  empty results, and syntax errors.

Avoid broad generated-file freshness tests unless the generated file itself is a
stable contract used by code or a public artifact.

## Manual GPU Checkpoints

Use `uv run ...` for Python commands. The exact config names may change as the
experiment registry lands, but the manual checkpoints should remain:

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_smoke.jsonl \
  --validate-data-only
```

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_smoke.jsonl \
  --output-dir outputs/experiments/direct_sql_smoke \
  --max-steps 5
```

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_full_non_oracle.jsonl \
  --output-dir outputs/experiments/direct_sql_full
```

```bash
uv run --active --no-sync python -m eval.run_eval \
  --model-name <model-or-adapter-name> \
  --input data/processed/<clean_holdout>.jsonl \
  --output results/runs/<run_id>/generations.jsonl \
  --manifest-output results/runs/<run_id>/manifest.json
```

```bash
uv run --active --no-sync python -m eval.rollout_eval \
  --input data/processed/<rollout_holdout>.jsonl \
  --output results/runs/<run_id>/rollout.jsonl \
  --manifest-output results/runs/<run_id>/rollout.manifest.json
```

## Operating Rule

The next research step should be chosen by this question:

> What clean, row-matched comparison would make the current method claim more
> believable?

If the answer is "another gate" or "another generated status table," the roadmap
has drifted. If the answer is "a larger non-oracle direct-SQL control, a clean
holdout, a failure analysis, or a same-row method comparison," it is on track.
