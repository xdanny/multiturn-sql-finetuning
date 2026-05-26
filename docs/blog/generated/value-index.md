| artifact | metric | value | interpretation |
| --- | --- | --- | --- |
| value_index_cosql_dev_100_summary.json | artifact_type | non_oracle_value_index_summary | Summary type for the generated value-index artifact. |
| value_index_cosql_dev_100_summary.json | index_source | database_contents | Where index entries come from; database contents means no reference SQL. |
| value_index_cosql_dev_100_summary.json | entry_count | 12661 | Number of distinct database values indexed under the per-column cap. |
| value_index_cosql_dev_100_summary.json | database_count | 20 | Number of CoSQL databases scanned from the fixed proxy input. |
| value_index_cosql_dev_100_summary.json | table_count | 80 | Number of tables with indexed values. |
| value_index_cosql_dev_100_summary.json | column_count | 411 | Number of columns with indexed values. |
| value_index_cosql_dev_100_summary.json | alias_count | 19334 | Raw and normalized aliases available before entity-resolution enrichment. |
| value_index_cosql_dev_100_summary.json | resolved_value_indexed_rate | 0.811 | Share of gold SQL literal values present in the database-derived index. |
| value_index_cosql_dev_100_summary.json | mention_alias_indexed_rate | 0.783 | Share of user-visible mentions already recoverable as value-index aliases. |
| value_index_cosql_dev_100_summary.json | resolved_value_indexed_count | 86 | Count of gold resolved values found in the index. |
| value_index_cosql_dev_100_summary.json | mention_alias_indexed_count | 83 | Count of user mentions found as index aliases. |
| value_index_cosql_dev_100_summary.json | label_count | 106 | Gold value labels used only for coverage evaluation. |
