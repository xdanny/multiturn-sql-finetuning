# Handover

Date: 2026-05-26

## What changed

- Verified a real GPU LoRA finetune path on WSL2 + RTX 5090 for the structured
  Stage 3 semantic-layer experiment.
- Confirmed the earlier finetune failure was environmental, not a training
  contract problem: Triton could not find a usable Linux compiler.
- Documented the working WSL compiler path in `README.md` and `train/AGENTS.md`.

## Working finetune recipe

The original successful 5-step smoke run used a local Stage 3 artifact from a
temporary branch. Current `main` should use the smoke matrix in
`docs/finetuning_smoke_matrix.md` instead. The equivalent current-main semantic
smoke command is:

```bash
CC=/home/dan/.local/bin/cc \
ZIG_LOCAL_CACHE_DIR=/tmp/zig-cache \
ZIG_GLOBAL_CACHE_DIR=/tmp/zig-global-cache \
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_semantic_smoke.jsonl \
  --eval-data data/processed/train_semantic_smoke.jsonl \
  --output-dir outputs/experiments/semantic_context_smoke \
  --max-steps 5 \
  --report-to none
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

See `docs/finetuning_smoke_matrix.md`.

Metric DSL and behavior/recovery now have small canonical finetuning rows and
same-fixture direct-SQL control rows under `docs/data_artifacts/`. They are
smoke inputs, not benchmark wins.
