| artifact | metric | value | interpretation |
| --- | --- | --- | --- |
| value_grounding_labels_cosql_dev_100.jsonl | carried_from_prior_sql_count | 0 | The value is inherited from a prior assistant SQL turn rather than the current user text. |
| value_grounding_labels_cosql_dev_100.jsonl | database_count | 17 | Number of databases touched by the value-label artifact. |
| value_grounding_labels_cosql_dev_100.jsonl | dialog_turn_count | 94 | Number of dialog turns with at least one value predicate. |
| value_grounding_labels_cosql_dev_100.jsonl | exact_in_current_turn_count | 82 | The user stated the stored value directly in the current turn. |
| value_grounding_labels_cosql_dev_100.jsonl | exact_in_history_count | 21 | The current SQL depends on a value mentioned in an earlier user turn. |
| value_grounding_labels_cosql_dev_100.jsonl | missing_from_user_text_count | 3 | The stored literal is not present in user text and needs a value index or display-to-storage normalization. |
| value_grounding_labels_cosql_dev_100.jsonl | requires_context_carryover_count | 21 | Rows where the value-grounding target depends on previous turns. |
| value_grounding_labels_cosql_dev_100.jsonl | requires_value_normalization_count | 3 | Rows where exact text matching is not enough to recover the stored value. |
| value_grounding_labels_cosql_dev_100.jsonl | value_reference_count | 106 | Gold SQL literal predicates turned into value-grounding labels on the fixed CoSQL proxy slice. |
