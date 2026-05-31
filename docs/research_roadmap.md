# Research Roadmap

This is the canonical long-term roadmap for the multi-turn SQL fine-tuning
program. It replaces the old gate-heavy day-to-day direction with smaller,
row-matched method comparisons.

Last audited: 2026-05-31 on commit `2e1b804`.

Checkpoint status legend:

- `[x]` complete for the current repo state.
- `[~]` in progress with useful artifacts or code present, but the checkpoint's
  evidence requirement is not fully satisfied.
- `[ ]` pending; no sufficient current evidence exists yet.

Current checkpoint progress:

- `[x]` Checkpoint 0: Freeze The Current State.
- `[x]` Checkpoint 1: Simplify The Research Loop.
- `[x]` Checkpoint 2: Establish Honest Dataset Roles.
- `[~]` Checkpoint 3: Rebuild Baselines At Real Scale.
- `[~]` Checkpoint 4: Let Failure Analysis Choose Methods.
- `[~]` Checkpoint 5: Planner First, But Non-Oracle.
- `[~]` Checkpoint 6: Semantic Layer And Value Grounding.
- `[~]` Checkpoint 7: Metric DSL.
- `[~]` Checkpoint 8: Generated-History Recovery.
- `[ ]` Checkpoint 9: Hosted And Target Benchmark Transfer.

The publishable benchmark claim is simple:

> A local fine-tuned method must beat the right direct-SQL control on clean
> held-out multi-turn SQL tasks, then compare against hosted baselines under the
> same protocol.

The current repo is not there yet. It has useful proxy evidence, method
surfaces, and diagnostic artifacts, but most scored model evidence is still a
100-turn CoSQL proxy slice. Many method artifacts are intentionally small:
2 metric-DSL rows, 1 behavior-recovery row, and 5 synthetic fixtures.
`data/processed/train.jsonl` is absent or empty in the checked-in tree, so
current training evidence mostly comes from small prepared files and recorded
outputs under `outputs/`.

Supported non-oracle proxy claims are narrow: the base model reaches `0.590`
value accuracy on the inspected 100-turn CoSQL proxy, the 100-step LoRA reaches
`0.630`, and the best semantic prompt condition reaches `0.640`. The `0.890`
schema-pruned result is an oracle diagnostic only because reference SQL-derived
planning hints enter the run.

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

Status: `[~]` in progress. Split-based non-oracle input preparation exists for
the direct-SQL control, but current scored evidence is still proxy-scale; the
full-scale direct-SQL base and direct-SQL LoRA controls have not been rebuilt.

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

Current Checkpoint 3 preparation artifacts:

- `data.prepare_split` materializes prepared JSONL from frozen split manifests
  without oracle planning hints or oracle-pruned semantic context.
- `configs/direct_sql_full_non_oracle.yaml` names the direct-SQL full-control
  adapter and prepared train/proxy/holdout paths.
- `eval.run_eval` result manifests preserve split provenance, latency, token
  usage, and estimated generation cost when the endpoint returns usage fields.
- `eval.checkpoint3_artifact_audit` verifies the required prepared inputs,
  base/LoRA proxy and clean-holdout result manifests, generated-history rollout
  manifests, non-oracle policy, required metrics, and base/LoRA row identity
  matching before this checkpoint can move to `[x]`.
- `configs/experiments.yaml` marks `direct_sql_full_non_oracle_control` as
  `input_prep_ready`, not as a measured baseline.

Prepare the three CoSQL direct-SQL inputs with:

```bash
uv run --active --no-sync python -m data.prepare_split \
  --split-id cosql_train_v1 \
  --output data/processed/direct_sql_full/cosql_train_v1.jsonl \
  --manifest-output data/processed/direct_sql_full/cosql_train_v1.manifest.json

uv run --active --no-sync python -m data.prepare_split \
  --split-id cosql_dev_100_proxy_seen_v1 \
  --output data/processed/direct_sql_full/cosql_dev_100_proxy_seen_v1.jsonl \
  --manifest-output data/processed/direct_sql_full/cosql_dev_100_proxy_seen_v1.manifest.json

uv run --active --no-sync python -m data.prepare_split \
  --split-id cosql_dev_clean_holdout_v1 \
  --output data/processed/direct_sql_full/cosql_dev_clean_holdout_v1.jsonl \
  --manifest-output data/processed/direct_sql_full/cosql_dev_clean_holdout_v1.manifest.json
```

Then validate the training inputs before spending GPU time:

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/direct_sql_full_non_oracle.yaml \
  --data data/processed/direct_sql_full/cosql_train_v1.jsonl \
  --eval-data data/processed/direct_sql_full/cosql_dev_100_proxy_seen_v1.jsonl \
  --validate-data-only
```

Checkpoint 3 is complete only after both base and LoRA direct-SQL runs report
the required metrics on row-matched proxy and clean-holdout manifests.

Audit the full evidence contract with:

```bash
uv run --active --no-sync python -m eval.checkpoint3_artifact_audit \
  --config configs/direct_sql_full_non_oracle.yaml
```

The audit is expected to fail until the full base, LoRA, and generated-history
rollout manifests listed in `configs/direct_sql_full_non_oracle.yaml` exist.

## Checkpoint 4: Let Failure Analysis Choose Methods

Status: `[~]` in progress. Current diagnostic artifacts preserve failures and
next actions, but failure analysis on clean validation is not yet available.

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

## Checkpoint 5: Planner First, But Non-Oracle

Status: `[~]` in progress. Non-oracle planner scoring and a 24-turn negative
predicted-planner comparison exist; planner quality is not yet high enough to
promote another SQL-generation run.

Train or prompt a planner only on training-split gold labels. Evaluate planner
F1 on held-out rows before feeding predicted plans into SQL generation.

Planner outputs should cover:

- tables;
- columns;
- join path;
- query skeleton;
- projection shape;
- aggregation and grouping;
- duplicate-row policy;
- value and entity slots.

Then compare predicted-planner SQL against the same-model direct SQL control on
identical rows. Gold plans remain scorer-side labels. Reference SQL-derived
planner hints can enter prompts only in runs explicitly named as oracle
diagnostics.

## Checkpoint 6: Semantic Layer And Value Grounding

Status: `[~]` in progress. Value labels, a non-oracle value index, semantic
retrieval inputs, and alias/column context inputs exist; a same-row semantic
SQL win on a clean holdout does not.

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

## Checkpoint 7: Metric DSL

Status: `[~]` in progress. Parser/evaluator code and two-row fixtures exist,
but prompt-only and 5-step evidence are negative diagnostics rather than a DSL
method win.

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
