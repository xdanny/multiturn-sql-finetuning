# Train AGENTS

This subtree owns `train.finetune`, training-time validation, and run policy.

Primary responsibilities:

- Keep fine-tuning entry points aligned with the measured repo workflow:
  prepared chat JSONL, Qwen LoRA training, endpoint/local evaluation, and result
  manifests.
- Make run naming, data source, step count, adapter path, and mixture intent
  explicit in docs and command examples.
- Keep command examples aligned with the actual `train.finetune` CLI.

Rules:

- Prefer `uv run python -m train.finetune ...` in repo docs and examples.
- On this WSL machine, if Linux `gcc` / `clang` are absent but
  `/home/dan/.local/bin/cc` exists, export `CC=/home/dan/.local/bin/cc`
  before `train.finetune` so Triton can compile launchers. In constrained
  shells, also point Zig caches at writable directories such as `/tmp`.
- Treat dataset mixture and stage naming as part of experiment meaning, not as
  optional metadata.
- Before committing generated data under `docs/data_artifacts/`, check
  `docs/data_artifacts/README.md`. Most run-specific files belong under
  `outputs/` or `results/`, not in the source tree.
- Current `train.finetune` supports `--validate-data-only`, `--dry-run`,
  `--max-steps`, `--output-dir`, and `--report-to`.
- If a new fine-tuning target is added, document which control it is expected to
  beat and which eval command clears that claim.

When editing here, inspect:

- `README.md`
- `data/prepare.py`
- `train/finetune.py`
- `docs/research_roadmap.md`
- `tests/test_train_finetune.py`
