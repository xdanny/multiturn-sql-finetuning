| claim_id | claim_status | evaluation_mode | allowed_public_claim | evidence |
| --- | --- | --- | --- | --- |
| qwen35_9b_base_cosql_dev_100turns | supported_proxy | non_oracle_generation | local proxy result only | value=0.59, strict=0.37, rows=100.0 |
| multiturn_sql_100_cosql_dev_100turns | supported_proxy | non_oracle_generation | local proxy result only | value=0.63, strict=0.53, rows=100.0 |
| semantic_prompt_minimal_executable_cosql_dev_100turns | supported_proxy | non_oracle_generation | local proxy result only | value=0.64, strict=0.42, rows=100.0 |
| schema_pruned_trained100_oracle_cosql_dev_100turns | diagnostic_upper_bound | oracle_planner_diagnostic | oracle planner diagnostic only | value=0.89, strict=0.82, rows=100.0 |
| metric-dsl-bootstrap.metric_dsl | supported_metric_dsl_quality | metric_dsl | metric-DSL quality only, not direct SQL superiority | value=1.0, strict=0.0, rows=2.0 |
| value_index_coverage | supported_value_retrieval_coverage | value_retrieval_coverage | value/entity retrieval coverage only | value=nan, strict=nan, rows=12661.0 |
| model_generated_history_rollout | pending | non_oracle_generation | no generated-history rollout result yet | no model-generated-history rollout manifest |
| rollout_beats_teacher_forced_history | pending | not_run | no generated-history diagnostic comparison yet | no side-by-side rollout-vs-teacher-forced comparison |
| hosted_sota_same_protocol | pending | not_run | no hosted/SOTA comparison yet | no same-protocol hosted-model manifest |
| local_beats_hosted_same_protocol | pending | not_run | no local-vs-hosted win yet | no local-vs-hosted comparison with positive value delta |
| bird_interact_local_vs_hosted | pending | not_run | no BIRD-Interact claim yet | no BIRD-Interact result manifest |
| metric_dsl_beats_direct_sql | pending | metric_dsl | no metric-DSL vs direct-SQL improvement claim yet | no non-oracle direct-SQL comparison with positive metric-DSL value delta |
| semantic_value_retrieval_improves_sql | pending | non_oracle_generation | no semantic value-retrieval SQL win yet | no row-matched semantic value-retrieval SQL comparison |
