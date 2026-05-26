# Handover

Date: 2026-05-26

## What changed

- Verified a real GPU LoRA finetune path on WSL2 + RTX 5090 for the structured
  Stage 3 semantic-layer experiment.
- Confirmed the earlier finetune failure was environmental, not a training
  contract problem: Triton could not find a usable Linux compiler.
- Documented the working WSL compiler path in `README.md` and `train/AGENTS.md`.

## Working finetune recipe

The successful 5-step smoke run used:

```bash
CC=/home/dan/.local/bin/cc \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-cache \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data docs/data_artifacts/semantic_layer_training_rows.jsonl \
  --eval-data docs/data_artifacts/semantic_layer_training_rows.jsonl \
  --output-dir outputs/experiments/semantic_layer_5steps \
  --max-steps 5 \
  --report-to none \
  --expected-training-target semantic_layer \
  --expected-evaluation-mode non_oracle_generation \
  --expected-benchmark synthetic_semantic_layer \
  --training-manifest-output outputs/experiments/semantic_layer_5steps/training.manifest.json \
  --run-id semantic-layer-5steps
```

## Observed result

- GPU: `NVIDIA GeForce RTX 5090`
- Run status: completed
- Train rows: `5`
- Eval rows: `5`
- Train runtime: `48.24s`
- Train loss: `2.281`
- Eval loss: `1.963`
- Output adapter: `outputs/experiments/semantic_layer_5steps/final`
- Training manifest: `outputs/experiments/semantic_layer_5steps/training.manifest.json`

## Important note

- `outputs/` artifacts are intentionally not tracked by git here, so the repo
  changes for this handoff are documentation only.
- The current compiler workaround is good enough to keep experimenting without
  a system-wide `build-essential` install.

## Recommended next runs

1. Metric DSL smoke:

```bash
CC=/home/dan/.local/bin/cc \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-cache \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data docs/data_artifacts/metric_dsl_training_rows.jsonl \
  --eval-data docs/data_artifacts/metric_dsl_training_rows.jsonl \
  --output-dir outputs/experiments/metric_dsl_5steps \
  --max-steps 5 \
  --report-to none \
  --expected-training-target metric_dsl \
  --expected-evaluation-mode metric_dsl \
  --expected-benchmark synthetic_metric_dsl_bootstrap \
  --training-manifest-output outputs/experiments/metric_dsl_5steps/training.manifest.json \
  --run-id metric-dsl-5steps
```

2. Behavior recovery proxy smoke:

```bash
CC=/home/dan/.local/bin/cc \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-cache \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data docs/data_artifacts/behavior_recovery_proxy_train.jsonl \
  --eval-data docs/data_artifacts/behavior_recovery_proxy_eval.jsonl \
  --output-dir outputs/experiments/behavior_recovery_proxy_5steps \
  --max-steps 5 \
  --report-to none \
  --expected-training-target behavior_recovery \
  --expected-evaluation-mode non_oracle_generation \
  --expected-benchmark prepared \
  --training-manifest-output outputs/experiments/behavior_recovery_proxy_5steps/training.manifest.json \
  --run-id behavior-recovery-proxy-5steps
```
