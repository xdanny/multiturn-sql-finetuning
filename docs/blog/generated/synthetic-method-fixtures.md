| artifact | metric | value | interpretation |
| --- | --- | --- | --- |
| synthetic_method_fixtures_summary.json | fixture_count | 5 | Curated synthetic rows that isolate method-specific multi-turn failures before endpoint spend. |
| synthetic_method_fixtures_summary.json | schema_count | 1 | Number of deterministic SQLite schemas in the fixture pack. |
| synthetic_method_fixtures_summary.json | non_oracle_fixture_count | 5 | Rows designed so reference SQL is used for scoring, not prompt context. |
| synthetic_method_fixtures_summary.json | failure_modes | context_carryover, duplicate_row_policy, entity_resolution, grain_change, grain_fanout, measure_preservation, recovery, semantic_model, value_normalization | Failure classes covered by synthetic rows: values, entity resolution, grain/fanout, MEASURE() preservation, and recovery. |
| synthetic_method_fixtures_summary.json | training_targets | behavior_recovery, metric_dsl, planner_first_sql, semantic_layer | Candidate tuning targets exercised before larger CoSQL or BIRD-Interact endpoint runs. |
| synthetic_method_fixtures_summary.json | required_artifacts | entity_resolution_labels, generated_history_trace, grain_fanout_fixtures, semantic_model_manifest, value_index | Data artifacts the fixtures are meant to validate or supervise. |
