# Research Roadmap

This is the canonical long-term roadmap for the multi-turn SQL fine-tuning
program. It replaces the old gate-heavy day-to-day direction with smaller,
row-matched method comparisons.

Last audited: 2026-06-01 after the Checkpoint 3 endpoint evaluation,
generated-history rollout, clean-holdout failure analysis, and Checkpoint 5
planner schema/projection/table-selection repairs, planner-SFT readiness,
projection-sequence planner-SFT training, schema-label-source correction, and
corrected planner-SFT training/readiness plus alias-insensitive projection-order
scoring.

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
- `[~]` Checkpoint 5: Planner First, But Non-Oracle.
- `[~]` Checkpoint 6: Semantic Layer And Value Grounding.
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
holdout, but the structured method arms still need same-row clean-holdout wins.
Many method artifacts remain intentionally small: 2 metric-DSL rows, 1
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

## Checkpoint 5: Planner First, But Non-Oracle

Status: `[~]` in progress. Non-oracle planner scoring, a 24-turn negative
predicted-planner comparison, schema-context repair, lexical projection and
table-selection repairs, train-split planner-SFT data paths, bounded planner-SFT
readiness evidence, a projection-sequence planner adapter, a corrected
schema-label-source planner adapter/readiness run, and an alias-insensitive
projection-order scorer repair exist. The corrected adapter still misses the
ordered selected-expression threshold, so no new SQL execution pair or SQL win
is claimed.

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

`eval.planner_readiness` is the current promotion audit for this checkpoint. It
keeps planner-to-SQL endpoint runs behind explicit held-out quality checks:
enough rows, no planner parse errors, bounded low-macro-score rate, minimum
mean table/column/skeleton/projection scores, and a ready endpoint-pair
preflight. A readiness summary is not a SQL win; it only decides whether the
next same-row predicted-planner SQL comparison is worth running.

Latest Checkpoint 5 evidence from 2026-05-31:

- Split preparation now resolves repo-relative `tables_path` through
  `--source-root`, so fresh worktrees include schema and semantic model context
  in prepared CoSQL rows instead of database IDs and questions only.
- Lexical planner table F1 moved from `0.000` to `0.650` on the proxy rows and
  from `0.000` to `0.633` on the clean holdout rows.
- Macro planner score moved only from `0.575` to `0.580` on proxy and from
  `0.574` to `0.583` on clean holdout because column linking and projection
  shape remain weak.
- The evidence file is
  `docs/training_runs/planner_schema_context_repair_20260531.json`.
- The lexical planner now separates relevant columns from projected-expression
  count, splits simple compound column names such as `FullName`, and emits
  non-empty `selected_expressions` for predicted-plan prompts.
- After that projection repair, selected-count match moved from `0.187` to
  `0.847` on proxy and from `0.171` to `0.838` on clean holdout. Macro planner
  score moved to `0.662` on proxy and `0.666` on clean holdout.
- The readiness summary is still not promotable: table F1 `0.625`, column F1
  `0.155`, and skeleton F1 `0.652` on clean holdout remain below policy.
- The evidence file is
  `docs/training_runs/planner_projection_prior_20260531.json`.
- The lexical planner now treats generic column names such as `name`, `title`,
  `id`, `code`, `type`, `year`, `date`, `number`, and `amount` as table-local
  hints instead of letting them pull unrelated tables into the plan.
- After that table-selection repair, proxy table F1 moved from `0.643` to
  `0.701`, clean-holdout table F1 moved from `0.625` to `0.686`, and
  clean-holdout macro planner score moved from `0.666` to `0.675`.
- The readiness summary still blocks another endpoint comparison: clean-holdout
  column F1 `0.161` and skeleton F1 `0.656` remain below policy, and table F1
  is still just below the `0.700` threshold.
- The evidence file is
  `docs/training_runs/planner_generic_column_prior_20260531.json`.
- `data.planner_sft` now materializes train-split-only planner supervision rows.
  The prompt contains system/user context only, removes teacher-forced assistant
  SQL history, and keeps the normalized gold plan as the assistant training
  target rather than prompt context.
- A 16-row sample from `cosql_train_v1` wrote a `planner_sft_dataset` manifest
  with `split_roles={"train": 16}`, `reference_sql_visible_to_model=false`,
  `gold_plan_visible_to_model_prompt=false`, and
  `target_plan_visible_as_assistant_label=true`.
- The evidence file is
  `docs/training_runs/planner_sft_data_path_20260531.json`.
- A 1000-step train-split planner-SFT LoRA run completed on the RTX 5090 and
  saved a final adapter at
  `outputs/experiments/predicted_planner_sql/planner_sft_20260531_1000/final`.
  The full planner-SFT training dataset has 7,343 rows from 2,159 train
  dialogs, and the final adapter weights SHA-256 is
  `c27115f3d26625f02ad2b979ead303c0bbebf2afe205e2e5e612bd4be1c9bcd5`.
- Bounded proxy planner readiness now passes on 24 turns / 8 dialogs:
  macro planner score `0.877`, table F1 `0.944`, column F1 `0.826`,
  skeleton F1 `0.927`, parse error rate `0.000`, and
  `preflight_status=ready_for_endpoint_pair`.
- Bounded clean-holdout planner readiness now passes on 24 turns / 7 dialogs:
  macro planner score `0.942`, table F1 `0.972`, column F1 `0.844`,
  skeleton F1 `0.979`, parse error rate `0.000`, and
  `preflight_status=ready_for_endpoint_pair`.
- The evidence file is
  `docs/training_runs/planner_sft_1000_readiness_20260531.json`.
- The bounded clean-holdout same-row SQL pair ran on 24 turns / 7 dialogs after
  planner-SFT readiness passed. The original manifest exposed a local extraction
  problem: most generations continued into synthetic `user`/`assistant` turns
  after the first SQL statement when no semicolon was emitted.
- After correcting SQL extraction for those chat-role continuations, direct SQL
  reached value accuracy `0.875` and strict accuracy `0.792`;
  predicted-planner SQL reached value accuracy `0.667` and strict accuracy
  `0.542`.
- The corrected value-accuracy delta was `-0.2083`, so the planner path remains
  negative downstream evidence despite strong bounded planner-label quality.
- The evidence file is
  `docs/training_runs/planner_sft_1000_sql_limit24_negative_20260531.json`.
- Row-level failure analysis of the corrected 24-turn pair found 15 rows both
  arms answered correctly, 6 direct-only regressions, 1 planner-only fix, and 2
  rows both arms missed. Of the 6 direct-only regressions, 4 were projection
  order flips after predicted-plan injection, 1 was a planner-state error, and
  1 was a generated SQL execution error.
- The evidence file is
  `docs/training_runs/planner_sft_1000_sql_failure_analysis_20260531.json`.
- The planner contract now preserves `projection_shape.selected_expressions`
  order for predicted-plan prompt hints and newly generated planner-SFT targets
  instead of sorting those expressions as a set. This is a contract/data fix,
  not a SQL win for the existing `planner_sft_20260531_1000` adapter.
- The evidence file is
  `docs/training_runs/planner_projection_order_contract_20260531.json`.
- Regenerating train-split planner-SFT rows under the order-preserving contract
  produced the same 7,343-row target JSONL: output SHA-256 stayed
  `49bf4caab2478b491bddae7b52689f317f119eaba0f211df1553423673fb8502`, with
  `changed_target_plan_count=0`.
- `eval.planner_eval` and `eval.planner_readiness` now score
  `selected_expression_order_match` so selected-count success cannot hide
  ordered projection-sequence failures. Rescoring the existing 24-turn
  planner-SFT clean-holdout predictions found ordered selected-expression match
  `0.792` with 5 mismatches, below the `0.900` readiness threshold.
- The evidence file is
  `docs/training_runs/planner_projection_order_metric_20260531.json`.
- Rerunning the same 1000-step planner adapter on the 24-turn clean-holdout
  slice under the order-aware readiness policy reproduced strong table F1
  `0.972`, column F1 `0.844`, skeleton F1 `0.979`, selected-count match
  `1.000`, and parse error rate `0.000`.
- Readiness still blocks promotion: ordered selected-expression match is
  `0.792` with 5 mismatches, below the `0.900` threshold, so the
  recommendation remains `improve_planner_before_comparison`.
- The evidence file is
  `docs/training_runs/planner_orderaware_readiness_rerun_20260531.json`.
- The planner prompt now explicitly defines `selected_expressions` as the final
  answer-column sequence, tells the model to preserve user-requested order, and
  avoids generated SQL aliases. A 24-row rerun with that prompt produced the
  same predicted-prepared hash as the previous order-aware run, so prompt
  wording alone did not move the blocker.
- Ordered selected-expression match remained `0.792`, below the `0.900`
  readiness threshold. The evidence file is
  `docs/training_runs/planner_projection_sequence_prompt_20260531.json`.
- `data.planner_sft` now writes `planner_prompt_policy=
  projection_sequence_instruction_v1` into rows and manifests. Regenerating the
  full 7,343-row train-split planner-SFT dataset changed all prompt rows,
  preserved all target plans (`changed_target_plan_count=0`), and added the
  projection-sequence instruction to every prompt.
- The evidence file is
  `docs/training_runs/planner_sft_sequence_instruction_dataset_20260531.json`.
- A 1000-step planner LoRA adapter trained on that projection-sequence dataset
  and saved a final adapter at
  `outputs/experiments/predicted_planner_sql/planner_sequence_sft_20260531_1000/final`.
  Training completed with reported train loss `0.09426`; final adapter weights
  SHA-256 is
  `a319dc2541b8490117861f3eb085fec8af10c0c8e8000e70036e39757b5629c4`.
- On the 24-turn clean-holdout readiness slice, the projection-sequence adapter
  scored macro planner score `0.837`, table F1 `0.875`, column F1 `0.690`,
  skeleton F1 `0.867`, selected-count match `0.917`, and ordered
  selected-expression match `0.625`.
- Readiness blocks promotion because 2 of 24 planner generations emitted SQL
  instead of planner JSON, ordered selected-expression match is below the
  `0.900` threshold, and no endpoint-pair preflight is ready. The evidence file
  is
  `docs/training_runs/planner_sequence_sft_1000_readiness_20260531.json`.
- `eval.planner_predict` now strips teacher-forced assistant SQL history from
  planner prediction prompts so inference matches the planner-SFT system/user
  prompt distribution instead of exposing earlier gold SQL-shaped assistant
  turns.
- Rerunning the same projection-sequence adapter on the same 24-turn
  clean-holdout slice with that user-only planner prompt eliminated the parse
  errors (`parse_error_count=0`) and produced a ready endpoint-pair preflight.
  Macro planner score moved to `0.860`, table F1 to `0.931`, column F1 to
  `0.693`, skeleton F1 to `0.933`, and selected-count match to `1.000`.
- Readiness still blocks promotion because ordered selected-expression match is
  `0.583`, below the `0.900` threshold. The evidence file is
  `docs/training_runs/planner_sequence_useronly_readiness_20260531.json`.
- A label-source audit found that prepared inputs can retain stale normalized
  `gold_plans` while `schema_link_labels` preserve the current SQL-derived
  selected-expression order. Planner scoring, predicted-planner prepared
  artifacts, and planner-SFT target generation now prefer `schema_link_labels`
  when both sources are present.
- Rescoring the 24-turn user-only planner slice with schema labels as the
  answer-key source left the aggregate selected-expression order metric
  unchanged at `0.583`: one stale-label false negative and one stale-label false
  positive canceled out. The evidence file is
  `docs/training_runs/planner_schema_label_gold_source_20260601.json`.
- Regenerating the full train-split planner-SFT dataset from the corrected
  schema-label-preferred answer-key source produced 7,343 rows with unchanged
  prompts and changed target plans for 618 turns versus the prior
  projection-sequence dataset. The evidence file is
  `docs/training_runs/planner_sft_schema_label_source_dataset_20260601.json`.
- A 1000-step planner LoRA adapter trained on that corrected
  schema-label-source dataset and saved a final adapter at
  `outputs/experiments/predicted_planner_sql/planner_schema_label_sft_20260601_1000/final`.
  Training completed with reported train loss `0.09429`, runtime `6707`
  seconds, and final adapter weights SHA-256
  `1a1dc047c1ecd940cd9455ba2e1afdc66f1e4274364d49a95bd92458ff759367`.
- This is training evidence only: order-aware planner readiness has not yet
  been rerun for the corrected adapter, so no endpoint comparison or SQL win is
  claimed. The evidence file is
  `docs/training_runs/planner_schema_label_source_sft_1000_20260601.json`.
- Rerunning order-aware clean-holdout readiness for the corrected adapter on
  the same 24-turn / 7-dialog slice eliminated parse errors and produced
  `preflight_status=ready_for_endpoint_pair`. Macro planner score moved to
  `0.884`, table F1 to `0.944`, column F1 to `0.738`, skeleton F1 to `0.936`,
  and selected-count match stayed `1.000`.
- Readiness still blocks promotion because ordered selected-expression match is
  `0.708`, below the `0.900` threshold. The evidence file is
  `docs/training_runs/planner_schema_label_source_readiness_20260601.json`.
- The selected-expression order scorer now ignores SQL/table qualifiers inside
  projection expressions while table and column identity remain scored by
  `table_f1` and `column_f1`. Rescoring the same 24-turn corrected-adapter
  predictions removed alias false negatives, moved macro planner score to
  `0.897`, and moved ordered selected-expression match to `0.833`.
- Readiness still blocks promotion because four projection-shape issues remain:
  two count-column/count-distinct misses, one destination-airport distinct-count
  miss, and one group-key/aggregate order reversal. The evidence file is
  `docs/training_runs/planner_alias_normalized_readiness_20260601.json`.

The next Checkpoint 5 work should fix the remaining count-aggregate and
group/aggregation projection-order mismatches before another bounded same-row
SQL pair. A positive value-accuracy delta remains required before promoting the
planner path.

Architecture note after the alias-normalized readiness run: do not spend the
next planner iteration on another broad prompt or SFT sweep with the same flat
planner target. The remaining failures are not table-alias noise; they are
projection-slot semantics. The current contract stores answer order in
`selected_expressions` while aggregation and grouping live in neighboring lists,
which makes `count(distinct column)` versus `count(*)` and
`group_key, aggregate` versus `aggregate, group_key` too easy to blur. The next
planner architecture should model ordered output slots directly, for example
`[{kind, source_column, aggregate, distinct, display_order}]`, and keep relevant
filter/join columns separate from answer columns. Only after that contract and
scorer change should another planner-SFT dataset or endpoint SQL pair run.

## Checkpoint 6: Semantic Layer And Value Grounding

Status: `[~]` in progress. Value labels, a non-oracle value index, semantic
retrieval inputs, alias/column context inputs, and a clean-holdout semantic
value-retrieval endpoint pair exist; the full same-row semantic SQL comparison
regressed versus direct SQL on the clean holdout.

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

The next Checkpoint 6 work should diagnose the full-run regression before another
endpoint pair: compare direct-correct/semantic-wrong rows, measure whether
retrieved value context adds prompt noise, and prune retrieval or redesign
semantic context before promoting semantic value retrieval.

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

`eval.compare_metric_dsl_direct_sql` is the current promotion audit for this
checkpoint. It keeps Metric DSL claims behind same-row direct-SQL comparison,
database-backed compiled-SQL scoring, a clean-holdout split role, enough
comparable rows, full parse and compile rates, full `MEASURE(...)`
preservation, a positive value-accuracy delta, no strict-accuracy regression,
and non-oracle semantic-model provenance. Synthetic or tiny fixture evidence
can diagnose the contract, but it cannot promote the method.

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
