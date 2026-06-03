# Data AGENTS

This subtree owns prepared data, synthetic fixtures, value artifacts, and
semantic context used by fine-tuning and evaluation.

Primary responsibilities:

- Make every training or eval row explicit about source, target, split role, and
  evaluation mode when that metadata is available.
- Keep generated artifacts reproducible from commands, but avoid checking in
  run-specific outputs.
- Maintain small canonical inputs that let another developer inspect a method
  before running a larger experiment.

Rules:

- `reference_sql`, expected rows, gold metric DSL, repair labels, and future
  turns are scorer-side data during validation, holdout, and hosted comparison.
- Database-derived value indexes and schema introspection context are valid
  model inputs when generated without evaluation answers.
- Before adding files under `docs/data_artifacts/`, check
  `docs/data_artifacts/README.md`.
- Prefer updating an existing artifact family over creating a parallel one with
  overlapping meaning.

When editing here, inspect:

- `docs/data_artifacts/README.md`
- `data/prepare.py`
- `data/semantic_value_retrieval_inputs.py`
- `data/value_artifacts.py`
- `data/value_index.py`
- `data/synthetic_method_fixtures.py`
