| artifact | metric | value | interpretation |
| --- | --- | --- | --- |
| planner_readiness_cosql_dev_100.json | endpoint_pair_ready | True | Whether direct and predicted prepared inputs have matching row identities before endpoint execution. |
| planner_readiness_cosql_dev_100.json | claim_boundary | readiness only; no SQL execution claim | What this artifact is allowed to prove. |
| planner_readiness_cosql_dev_100.json | row_count | 100 | Number of planner-evaluated turns in the readiness report. |
| planner_readiness_cosql_dev_100.json | dialog_count | 32 | Number of dialogs covered by the planner readiness report. |
| planner_readiness_cosql_dev_100.json | database_count | 15 | Number of databases covered by the planner readiness report. |
| planner_readiness_cosql_dev_100.json | column_zero_rate | 0.790 | Share of turns where the predicted planner recovered no gold columns. |
| planner_readiness_cosql_dev_100.json | selected_count_mismatch_rate | 0.820 | Share of turns where predicted projection width does not match the reference plan. |
| planner_readiness_cosql_dev_100.json | empty_projection_expression_rate | 1.000 | Share of turns where the planner produced no explicit projection expressions. |
| planner_readiness_cosql_dev_100.json | join_zero_when_gold_join_rate | 0.210 | Share of all turns where a gold join exists and the predicted join score is zero. |
| planner_readiness_cosql_dev_100.json | group_by_zero_when_gold_group_by_rate | 0.170 | Share of all turns where gold grouping exists and predicted group-by score is zero. |
| planner_readiness_cosql_dev_100.json | recommendation | improve_planner_before_claim | Decision for the next planner step before making SQL claims. |
| planner_readiness_cosql_dev_100.json | top_risks | empty_projection_expression, projection_shape, column_linking, low_macro_score | Largest planner weaknesses to address before endpoint claims. |
