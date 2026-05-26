| metric | why_it_matters | status |
| --- | --- | --- |
| metric_dsl_parse_rate | The generated artifact can be parsed as the DSL target. | pending_manifest |
| metric_dsl_compile_rate | The parsed DSL can compile through a governed semantic model. | pending_manifest |
| measure_preservation | The model kept MEASURE(name) instead of expanding metric SQL early. | pending_manifest |
| value_execution_accuracy | The compiled SQL returns the right values when reference SQL and a database are available. | pending_manifest |
| metric_dsl_value_delta_vs_direct_sql | The DSL-first path must beat a direct-SQL baseline before it supports a superiority claim. | pending_comparison |
| compiled_sql_execution_evaluated_rows | Execution accuracy is meaningful only over database-backed attempts. | pending_manifest |
| semantic_model_sha256s | Metric results must identify the semantic model used for compilation. | pending_manifest |
