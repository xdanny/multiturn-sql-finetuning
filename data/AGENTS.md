# Data AGENTS

This subtree owns prepared data, synthetic fixtures, value artifacts, and
semantic context used by finetuning and evaluation.

Primary responsibilities:

- Make every training or eval row explicit about benchmark, target, and
  evaluation mode.
- Keep oracle-derived labels separate from non-oracle retrieval artifacts.
- Keep generated artifacts reproducible from commands, but avoid checking in
  run-specific outputs.
- Maintain small canonical inputs that let another developer inspect a method
  before running a larger experiment.
- Maintain `data/splits/*.json` as the frozen split-role manifests used by the
  roadmap experiment registry.

Rules:

- `reference_sql`, expected rows, gold plans, gold metric DSL, repair labels,
  and future turns are scorer-side data unless a run is explicitly diagnostic.
- Database-derived value indexes can be non-oracle; answer-derived labels are
  diagnostic or supervision labels.
- Before adding files under `docs/data_artifacts/`, check
  `docs/data_artifacts/README.md`.
- Prefer updating an existing artifact family over creating a parallel one
  with overlapping meaning.

When editing here, inspect:

- `docs/data_artifacts/README.md`
- `data/prepare.py`
- `data/semantic_value_retrieval_inputs.py`
- `data/value_artifacts.py`
- `data/value_index.py`
- `data/synthetic_method_fixtures.py`
- `data/split_manifest.py`
- `docs/evidence_contract.md`
