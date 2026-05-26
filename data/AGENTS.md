# Data AGENTS

This subtree owns prepared data, intermediate artifact generation, semantic
context, value retrieval, and synthetic fixtures.

Primary responsibilities:

- Define prepared data contracts that training and evaluation can both consume.
- Keep every artifact auditable: prepared data, value index, semantic artifact,
  synthetic fixture, and planner labels should be reproducible from commands.
- Keep oracle and non-oracle provenance explicit in every artifact.
- Keep the cross-stage registry current: `data.finetuning_program_registry`
  should reflect the actual training targets and benchmark surfaces that the
  repo supports.

Rules:

- Prefer prepared data with explicit metadata over ad hoc prompt snippets.
- Treat artifact shape as part of the public contract; update tests with schema
  changes.
- Preserve semantic distinctions: raw schema text, semantic context, value
  index, and gold labels are not interchangeable.
- If a field carries oracle meaning, name it directly and keep the artifact
  diagnostic.

When editing here, inspect:

- `docs/finetuning_ladder.md`
- `data/plan_contract.py`
- `data/semantic_layer_dataset.py`
- `data/semantic_layer_direct_sql_dataset.py`
- `data/semantic_proxy_dataset.py`
- `data/semantic_proxy_direct_sql_dataset.py`
- `data/behavior_recovery_dataset.py`
- `data/behavior_recovery_direct_sql_dataset.py`
- `data/behavior_recovery_proxy_dataset.py`
- `data/hosted_baseline_dataset.py`
- `data/bird_interact_transfer_dataset.py`
- `data/metric_dsl.py`
- `data/finetuning_program_registry.py`
- `data/value_artifacts.py`
- `data/value_index.py`
