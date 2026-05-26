| run | source_artifact | schema_mismatch_rows | unknown_column | wrong_table_column | ambiguous_unqualified_column | unknown_table | example_turn | example_diagnostic |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Base Qwen 3.5 9B | results/rescored/vllm_qwen35_9b_base_cosql_dev_100turns.jsonl | 10 | 6 | 0 | 4 | 0 | prepared-0:1 | repairable: unknown_column; unknown_columns=popularity |
| 100-step LoRA | results/rescored/vllm_qwen35_9b_lora100_cosql_dev_100turns.jsonl | 39 | 32 | 5 | 3 | 0 | prepared-4:0 | repairable: wrong_table_column; wrong_table_columns=concert.name |
| Semantic 50-step + minimal executable | results/rescored/minimal_executable.jsonl | 38 | 35 | 1 | 3 | 0 | prepared-17:2 | repairable: wrong_table_column; wrong_table_columns=model_list.fullname |
