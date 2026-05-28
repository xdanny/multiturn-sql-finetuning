# Configs AGENTS

This subtree owns training configs and prompt-variant configs used by the SQL
finetuning and evaluation ladder.

Primary responsibilities:

- Keep training configs aligned with the method ladder: direct SQL, semantic
  context, metric DSL, behavior recovery, planner runs, and hosted/BIRD-style
  comparisons.
- Make prompt variants explicit about whether they are production-style,
  diagnostic, or oracle-derived.
- Keep dataset splits, row limits, model names, and output directories
  reproducible and easy to trace from docs and manifests.

Rules:

- Use `uv run ...` in documented commands that reference these configs.
- Do not add prompt variants that use reference SQL, gold plans, gold metric
  DSL, repair labels, expected rows, or future turns unless the config is
  explicitly named and labeled as oracle diagnostic.
- If a prompt variant requires planning hints, include `requires_planning_hints`
  and make the instruction say whether those hints are oracle-derived or
  predicted.
- Keep output paths under `outputs/` for training products and under `results/`
  for scored generations or comparison outputs.
- When adding a dataset or split, document whether it is CoSQL proxy, SParC,
  synthetic schema-rich SQL, BIRD mini-dev, hosted baseline, or BIRD-Interact
  target work.
- Keep debug configs clearly smaller than benchmark configs. Do not let a debug
  slice support a benchmark claim.
- If a config changes a method comparison, update
  `docs/finetuning_measurement_plan.md` or `docs/finetuning_smoke_matrix.md`.

When editing here, inspect:

- `train/finetune.py`
- `data/prepare.py`
- `eval/run_eval.py`
- `eval/method_readiness.py`
- `docs/finetuning_ladder.md`
- `docs/finetuning_smoke_matrix.md`
- `docs/finetuning_measurement_plan.md`
