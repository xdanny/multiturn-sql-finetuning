| system | fine_tuning_target | value_accuracy | context_carryover_accuracy | value_grounding_accuracy | measure_preservation_rate | recovery_success_rate | lab_takeaway |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct_sql_baseline | assistant SQL | 0.250 | 0.750 | 0.500 | 0.000 | 0.000 | This is not a benchmark result; it shows which intermediate behavior the target isolates in the shareable lab. |
| planner_first_sql | query plan then SQL | 0.250 | 1.000 | 0.250 | 0.000 | 0.000 | This is not a benchmark result; it shows which intermediate behavior the target isolates in the shareable lab. |
| semantic_value_sql | semantic state then SQL | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | This is not a benchmark result; it shows which intermediate behavior the target isolates in the shareable lab. |
| semantic_dsl_planner | MEASURE-preserving DSL then SQL | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | This is not a benchmark result; it shows which intermediate behavior the target isolates in the shareable lab. |
| behavior_recovery_sql | execution feedback then repair | 0.750 | 1.000 | 0.750 | 1.000 | 1.000 | This is not a benchmark result; it shows which intermediate behavior the target isolates in the shareable lab. |
