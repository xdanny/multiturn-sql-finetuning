# Finetuning Smoke Matrix

This file is the current-main runbook for short finetuning checks. It is not a
benchmark result table. Its job is to answer one practical question:

> Can this finetuning target load data, use the GPU path, and produce an adapter
> without crossing an oracle boundary?

Use these commands before scaling a method or writing a benchmark claim.
Use `docs/finetuning_measurement_plan.md` to decide which scored outputs and
comparison manifests are required after a smoke run.

## WSL GPU Environment

On this WSL machine, Triton needs an explicit compiler path when Linux `gcc` or
`clang` are not installed:

```bash
export CC=/home/dan/.local/bin/cc
export ZIG_LOCAL_CACHE_DIR=/tmp/zig-cache
export ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache
```

All Python commands should use `uv`.

## Direct SQL Control

This is the basic control arm. It trains directly on chat-format SQL rows.

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_smoke.jsonl \
  --eval-data data/processed/eval_cosql_smoke.jsonl \
  --output-dir outputs/experiments/direct_sql_smoke \
  --max-steps 5 \
  --report-to none
```

What it proves: the direct-SQL training loop runs. It does not prove a method
win by itself.

## Semantic Context Smoke

This checks the current semantic-context path on tracked rows.

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_semantic_smoke.jsonl \
  --eval-data data/processed/train_semantic_smoke.jsonl \
  --output-dir outputs/experiments/semantic_context_smoke \
  --max-steps 5 \
  --report-to none
```

What it proves: the model can train on rows that include semantic model hints.
It does not prove semantic-layer tuning beats direct SQL; that requires a
same-row comparison against the direct-SQL control.

## Metric DSL Status

Current `main` has two checked-in metric-DSL finetuning rows and two matching
direct-SQL control rows:

- `docs/data_artifacts/metric_dsl_training_rows.jsonl`
- `docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl`

Generate them with:

```bash
uv run --active --no-sync python -m data.metric_dsl_training_rows
```

Metric DSL smoke:

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data docs/data_artifacts/metric_dsl_training_rows.jsonl \
  --eval-data docs/data_artifacts/metric_dsl_training_rows.jsonl \
  --output-dir outputs/experiments/metric_dsl_smoke \
  --max-steps 5 \
  --report-to none
```

Same-fixture direct SQL control:

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl \
  --eval-data docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl \
  --output-dir outputs/experiments/metric_dsl_direct_sql_smoke \
  --max-steps 5 \
  --report-to none
```

After both adapters generate predictions for the same metric-heavy row
identities, use the paired runner to score and compare the arms:

```bash
uv run --active --no-sync python -m eval.run_metric_dsl_comparison \
  --metric-dsl-predictions results/metric_dsl/<run-id>.metric_predictions.jsonl \
  --direct-sql-predictions results/metric_dsl/<run-id>.direct_predictions.jsonl \
  --output-dir results/metric_dsl \
  --run-id <run-id> \
  --metric-dsl-model-name <metric-dsl-adapter> \
  --direct-sql-model-name <direct-sql-adapter>
```

What this proves: the metric-DSL and same-fixture direct-SQL training targets
both load through the current SFT path. It does not prove the DSL path wins.
That claim needs generated predictions from both adapters, database-backed
execution scoring, and a positive comparison manifest from
`eval.run_metric_dsl_comparison`.

## Behavior And Recovery Status

Current `main` has one checked-in behavior/recovery finetuning row and one
matching direct-SQL control row:

- `docs/data_artifacts/behavior_recovery_training_rows.jsonl`
- `docs/data_artifacts/behavior_recovery_direct_sql_training_rows.jsonl`

Generate them with:

```bash
uv run --active --no-sync python -m data.behavior_recovery_training_rows
```

Behavior recovery smoke:

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data docs/data_artifacts/behavior_recovery_training_rows.jsonl \
  --eval-data docs/data_artifacts/behavior_recovery_training_rows.jsonl \
  --output-dir outputs/experiments/behavior_recovery_smoke \
  --max-steps 5 \
  --report-to none
```

Same-fixture direct SQL control:

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data docs/data_artifacts/behavior_recovery_direct_sql_training_rows.jsonl \
  --eval-data docs/data_artifacts/behavior_recovery_direct_sql_training_rows.jsonl \
  --output-dir outputs/experiments/behavior_recovery_direct_sql_smoke \
  --max-steps 5 \
  --report-to none
```

Rollout evaluation is available for generated-history behavior:

```bash
uv run --active --no-sync python -m eval.rollout_eval \
  --input data/processed/eval_cosql_smoke.jsonl \
  --output results/rollout/<run-id>.jsonl \
  --manifest-output results/rollout/<run-id>.manifest.json \
  --database-root data/raw/cosql_dataset/database \
  --model-name <served-model-name>
```

What this proves: the recovery and same-fixture direct-SQL training targets
both load through the current SFT path. It does not prove recovery tuning wins.
That claim needs generated predictions from both adapters and a rollout
comparison where later turns see generated SQL history, not reference SQL.

## Reading Results

Smoke outputs belong under `outputs/experiments/`. Evaluation outputs and
comparison manifests belong under `results/`.

Do not move smoke-run manifests into `docs/data_artifacts/` unless the file is
small, canonical, and needed as an input contract for another developer.
