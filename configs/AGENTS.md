# Configs AGENTS

This subtree owns training configs and prompt-variant configs used by the SQL
finetuning and evaluation ladder.

Primary responsibilities:

- Keep training configs aligned with the method ladder: direct SQL, structured
  query briefs, semantic context, metric DSL, behavior recovery, and
  hosted/BIRD-style comparisons.
- Keep `configs/benchmark_protocols.yaml` as the source of truth for what each
  dataset or benchmark protocol can prove, what it blocks, and which leakage
  boundary applies.
- Keep `configs/experiments.yaml` as the compact registry for roadmap method
  hypotheses, split ids, controls, scorers, and output locations.
- Make prompt variants explicit about whether they are production-style,
  diagnostic, or oracle-derived.
- Keep dataset splits, row limits, model names, and output directories
  reproducible and easy to trace from docs and manifests.
- Keep split ids in `configs/experiments.yaml` aligned with
  `data/splits/*.json`.

Rules:

- Use `uv run ...` in documented commands that reference these configs.
- Do not add prompt variants that use reference SQL, gold plans, gold metric
  DSL, repair labels, expected rows, or future turns unless the config is
  explicitly named and labeled as oracle diagnostic.
- If a prompt variant requires decomposition hints, include the relevant hint
  flag and make the instruction say whether those hints are oracle-derived,
  train-split supervised, or predicted.
- Keep output paths under `outputs/` for training products and under `results/`
  for scored generations or comparison outputs.
- When adding a dataset or split, document whether it is CoSQL proxy, SParC,
  synthetic schema-rich SQL, BIRD mini-dev, hosted baseline, or BIRD-Interact
  target work.
- Keep debug configs clearly smaller than benchmark configs. Do not let a debug
  slice support a benchmark claim.
- If a config changes a method comparison, update
  `docs/finetuning_measurement_plan.md` or `docs/finetuning_smoke_matrix.md`.
- If a roadmap experiment changes, update `configs/experiments.yaml` and keep
  control arms, split ids, and scorer names explicit.
- If a benchmark protocol changes, update `configs/benchmark_protocols.yaml`
  and the docs that describe the row identity, leakage boundary, and run
  manifest requirements.

When editing here, inspect:

- `train/finetune.py`
- `data/prepare.py`
- `eval/run_eval.py`
- `docs/finetuning_ladder.md`
- `docs/finetuning_smoke_matrix.md`
- `docs/finetuning_measurement_plan.md`
