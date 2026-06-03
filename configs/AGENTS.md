# Configs AGENTS

This subtree owns training, benchmark, and prompt-variant configs used by the
multi-turn SQL fine-tuning comparisons.

Primary responsibilities:

- Keep configs aligned with measured comparisons: direct SQL, semantic context,
  value retrieval, behavior recovery, local baseline, fine-tuned adapters, and
  hosted baselines.
- Keep `configs/benchmark_protocols.yaml` clear about what each benchmark can
  prove, what it cannot prove, and which leakage boundary applies.
- Keep dataset splits, row limits, model names, and output directories
  reproducible from docs and manifests.

Rules:

- Use `uv run ...` in documented commands that reference these configs.
- Do not add prompt variants that expose reference SQL, expected rows, repair
  labels, or future turns to model prompts.
- Keep output paths under `outputs/` for training products and under `results/`
  for scored generations or comparison outputs.
- Keep debug configs clearly smaller than benchmark configs. Do not let a debug
  slice support a benchmark claim.
- If a benchmark protocol changes, update `configs/benchmark_protocols.yaml`
  and the docs that describe row identity, leakage boundaries, and manifests.

When editing here, inspect:

- `README.md`
- `data/prepare.py`
- `train/finetune.py`
- `eval/run_eval.py`
- `docs/research_roadmap.md`
