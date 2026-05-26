# Train AGENTS

This subtree owns `train.finetune`, training-time validation, and run policy.

Primary responsibilities:

- Keep finetuning entry points aligned with the structured ladder in
  `docs/finetuning_ladder.md`.
- Reject misleading training inputs before GPU time is spent.
- Make run naming, stage naming, and mixture intent explicit in documentation
  and command examples.

Rules:

- Preserve the `allow-oracle-diagnostic-data` guard. Oracle rows are diagnostic
  unless the run is explicitly labeled that way.
- Prefer `uv run python -m train.finetune ...` in repo docs and examples.
- Treat dataset mixture and stage naming as part of experiment meaning, not as
  optional metadata.
- When a finetuning stage has a fixed contract, pass
  `--expected-training-target`, `--expected-evaluation-mode`, and
  `--expected-benchmark` so mislabeled datasets fail before training.
- If a new finetuning target is added, document which control it is expected to
  beat and which eval command clears that claim.

When editing here, inspect:

- `train/finetune.py`
- `docs/finetuning_ladder.md`
- `docs/research_goal.md`
- `tests/test_train_finetune.py`
