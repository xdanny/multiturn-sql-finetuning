# Train AGENTS

This subtree owns `train.finetune`, training-time validation, and run policy.

Primary responsibilities:

- Keep finetuning entry points aligned with `docs/research_roadmap.md` and
  `configs/experiments.yaml`.
- Reject misleading training inputs before GPU time is spent.
- Make run naming, stage naming, and mixture intent explicit in documentation
  and command examples.
- Keep command examples aligned with the actual `train.finetune` CLI on
  current `main`.

Rules:

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
  `--max-steps`, `--output-dir`, and `--report-to`. Do not document
  unimplemented flags as if they already exist.
- If a new finetuning target is added, document which control it is expected to
  beat and which eval command clears that claim.

When editing here, inspect:

- `train/finetune.py`
- `docs/research_roadmap.md`
- `configs/experiments.yaml`
- `tests/test_train_finetune.py`
