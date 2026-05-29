# Train AGENTS

This subtree owns `train.finetune`, training-time validation, and run policy.

Primary responsibilities:

- Keep finetuning entry points aligned with the structured ladder in
  `docs/finetuning_ladder.md`.
- Reject misleading training inputs before GPU time is spent.
- Make run naming, stage naming, and mixture intent explicit in documentation
  and command examples.
- Keep command examples aligned with the actual `train.finetune` CLI on
  current `main`.

Rules:

- Preserve the `allow-oracle-diagnostic-data` guard. Oracle rows are diagnostic
  unless the run is explicitly labeled that way.
- Prefer `uv run python -m train.finetune ...` in repo docs and examples.
- On this WSL machine, if Linux `gcc` / `clang` are absent but
  `/home/dan/.local/bin/cc` exists, export `CC=/home/dan/.local/bin/cc`
  before `train.finetune` so Triton can compile its launchers. In constrained
  shells, also point Zig caches at writable directories such as `/tmp`.
- Treat dataset mixture and stage naming as part of experiment meaning, not as
  optional metadata.
- Before committing generated data under `docs/data_artifacts/`, check
  `docs/data_artifacts/README.md`. Most run-specific files belong under
  `outputs/` or `results/`, not in the source tree.
- Current `train.finetune` supports `--validate-data-only`, `--dry-run`,
  `--max-steps`, `--output-dir`, `--report-to`, and
  `--allow-oracle-diagnostic-data`. Do not document unimplemented flags as if
  they already exist.
- `train.finetuning_steps` validates `configs/finetuning_steps.yaml` and emits
  the concrete train/evaluate/compare sequence for method arms. Keep it aligned
  with `configs/finetuning_methods.yaml` and `configs/benchmark_protocols.yaml`.
- If a new finetuning target is added, document which control it is expected to
  beat and which eval command clears that claim.

When editing here, inspect:

- `train/finetune.py`
- `docs/finetuning_ladder.md`
- `docs/finetuning_smoke_matrix.md`
- `docs/research_goal.md`
- `tests/test_train_finetune.py`
