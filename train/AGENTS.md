# Train AGENTS

This subtree owns `train.finetune`, training-time validation, and run policy.

Primary responsibilities:

- Keep finetuning entry points aligned with the structured ladder in
  `docs/finetuning_ladder.md`.
- Reject misleading training inputs before GPU time is spent.
- Make run naming, stage naming, and mixture intent explicit in documentation
  and command examples.
- Emit a machine-checkable training manifest before and after runs so later
  eval artifacts can be tied back to an exact training input.
- Keep `data.finetuning_program_registry` aligned with any new training target
  or benchmark contract added here.

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
- When a finetuning stage has a fixed contract, pass
  `--expected-training-target`, `--expected-evaluation-mode`, and
  `--expected-benchmark` so mislabeled datasets fail before training.
- Use `--run-id` and `--training-manifest-output` when a run is intended to
  feed a benchmark claim or a same-row method comparison.
- If a new finetuning target is added, document which control it is expected to
  beat and which eval command clears that claim.

When editing here, inspect:

- `train/finetune.py`
- `docs/finetuning_ladder.md`
- `data/finetuning_program_registry.py`
- `docs/research_goal.md`
- `tests/test_train_finetune.py`
