# Tests AGENTS

This subtree owns regression tests for data contracts, scoring, comparison
logic, leakage checks, and small runnable examples.

Primary responsibilities:

- Protect shared behavior that would make a benchmark or fine-tuning comparison
  misleading if it regressed.
- Keep tests focused on row identity, scorer correctness, manifest provenance,
  and leakage boundaries.
- Prefer one or two narrow tests for a new evaluator or artifact contract over
  broad repo-shape tests.
- Keep GPU, endpoint, and long-running training checks out of default unit
  tests unless they are explicitly marked as smoke or manual workflows.

Rules:

- Use `uv run ... pytest ...` in documented verification commands.
- Do not add tests that require future predictions, private credentials, hosted
  model access, or live endpoint availability.
- When a method arm changes, test the contract that matters: same rows, same
  control, correct metric, and no answer leakage.
- When a generated artifact changes, test the generator or summary function,
  not every byte of generated output.

When editing here, inspect:

- `README.md`
- `docs/research_roadmap.md`
- `eval/result_manifest.py`
- `data/prepare.py`
