# Current Research Inventory

Last audited: 2026-05-31 on commit `2e1b804`.

This inventory completes Roadmap Checkpoint 0. It names the checked-in rows,
manifests, adapters, outputs, and claims that are currently available after the
old gate workflow cleanup. Runtime outputs are expected under `outputs/` and
`results/`; neither directory is checked in on current `main`.

## Dataset Roles

| Dataset or slice | Current role | Current evidence |
| --- | --- | --- |
| CoSQL raw train | `train` candidate | The roadmap records 2,159 train dialogs / 7,343 train turns in the local raw archive. Raw data is not checked in. |
| CoSQL raw dev | source for `proxy_dev_seen` and future split work | The roadmap records 293 dev dialogs / 1,007 dev turns in the local raw archive. Raw data is not checked in. |
| CoSQL dev 100 | `proxy_dev_seen` | Fixed inspected proxy slice. Prepared inputs are generated under `data/processed/`, not checked in. Summary artifacts record 100 rows, 327 assistant turns, and 20 databases. |
| SParC | future split manifest target | No split manifest or checked-in prepared rows yet. |
| BIRD mini-dev | future external/single-turn harness target | No split manifest or checked-in prepared rows yet. |
| Synthetic method fixtures | diagnostic fixture pack | `docs/data_artifacts/synthetic_method_fixtures.jsonl` has 5 rows across value normalization, entity resolution, grain/fanout, metric DSL, and recovery failure modes. |

`data/processed/train.jsonl` is not checked in on current `main`. Treat that as
no durable full-scale training evidence from the default training path.

## Checked-In Canonical Inputs

| Artifact | Rows or entries | Role |
| --- | ---: | --- |
| `docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl` | 106 value references | Gold SQL-derived labels for supervision, coverage, and scorer-side analysis. Not production prompt context. |
| `docs/data_artifacts/value_index_cosql_dev_100.jsonl` | 12,661 entries | Non-oracle database-content value index: 20 databases, 80 tables, 411 columns, and 19,334 aliases. |
| `docs/data_artifacts/semantic_value_retrieval_inputs_summary.json` | 100 rows / 327 assistant turns | Non-oracle semantic value retrieval inputs matched only from user-visible text and the database-derived value index. |
| `docs/data_artifacts/alias_column_context_inputs_summary.json` | 100 rows / 327 assistant turns | Non-oracle schema-introspection alias and column-role context inputs. |
| `docs/data_artifacts/metric_dsl_training_rows.jsonl` | 2 rows | Metric-DSL training target fixture rows. |
| `docs/data_artifacts/metric_dsl_direct_sql_training_rows.jsonl` | 2 rows | Direct-SQL control rows for the metric-DSL fixture identities. |
| `docs/data_artifacts/metric_dsl_prediction_inputs.jsonl` | 2 rows | Prompt-visible metric-DSL prediction inputs. |
| `docs/data_artifacts/metric_dsl_direct_sql_prediction_inputs.jsonl` | 2 rows | Direct-SQL prediction inputs for the same metric-heavy fixture identities. |
| `docs/data_artifacts/behavior_recovery_training_rows.jsonl` | 1 row | Recovery target fixture row. |
| `docs/data_artifacts/behavior_recovery_direct_sql_training_rows.jsonl` | 1 row | Direct-SQL control for the same recovery fixture. |
| `docs/data_artifacts/behavior_recovery_rollout_inputs.jsonl` | 1 row | Generated-history rollout diagnostic input. |
| `docs/data_artifacts/value_schema_repair_rollout_inputs.jsonl` | 1 row | Prompt-only value/schema repair diagnostic input. |
| `docs/data_artifacts/value_choice_consistency_rollout_inputs.jsonl` | 1 row | Prompt-only value-choice consistency diagnostic input. |
| `docs/data_artifacts/alias_column_validity_rollout_inputs.jsonl` | 1 row | Prompt-only alias/column-validity diagnostic input. |

Each checked-in canonical input has a matching manifest or summary under
`docs/data_artifacts/`.

## Evidence Snapshots

The current checked-in evidence snapshots are compact summaries under
`docs/training_runs/`:

- `gpu_finetuning_evidence.json`: records local RTX 5090 access, a fresh
  one-step smoke, and previous adapter evidence. The supported non-oracle proxy
  runs remain the 100-step direct-SQL LoRA at `0.630` value accuracy and the
  semantic-50/minimal-executable condition at `0.640` value accuracy on CoSQL
  dev 100. The `0.890` schema-pruned run is explicitly an oracle diagnostic.
- `lexical_predicted_planner_limit24.json`: negative non-oracle planner evidence
  on 24 matched turns. The lexical planner regressed value accuracy by `-0.0417`
  versus direct SQL and is not promoted.
- `metric_dsl_prompt_baseline.json` and `metric_dsl_arm_5steps.json`: tiny
  two-row metric-DSL diagnostics. Both show that DSL output obedience and metric
  semantics are not learned yet.
- `behavior_recovery_5steps.json`: one-row generated-history recovery
  diagnostic. The recovery adapter did not beat the direct-SQL control.
- `value_schema_repair_prompt.json`, `value_choice_consistency_prompt.json`, and
  `alias_column_validity_prompt.json`: one-row prompt-only repair diagnostics
  showing the sequence from value/schema failure, to correct storage-value choice
  with invalid column use, to a solved alias/column-validity prompt fixture.
- `alias_column_context_prompt_limit8.json` and
  `alias_column_context_prompt_limit24.json`: bounded CoSQL proxy prompt
  validations. Alias/column context changed generations but did not improve
  value or strict accuracy versus direct SQL.
- `hypothesis_arm_rollup.json`: compact rollup of the tiny diagnostic arms. It
  is not benchmark evidence.

## Runtime Outputs And Adapters

Checked-in docs refer to local runtime paths such as:

- `outputs/qwen35_9b_multiturn_sql_100steps/`
- `outputs/qwen35_9b_multiturn_sql_semantic_50steps/`
- `outputs/qwen35_9b_multiturn_sql_schema_pruned_100steps/`
- `outputs/gpu_access_smoke_1step/`
- `results/predicted_planner/`
- `results/rescored/`

These paths are not checked in on current `main`. Their recorded hashes and
metrics live in the compact evidence snapshots above.

## Active Claim Boundary

Supported claims are proxy-only:

- Base Qwen reaches `0.590` value accuracy on the inspected CoSQL dev 100 proxy.
- The 100-step direct-SQL LoRA reaches `0.630` value accuracy on that proxy.
- The best semantic prompt condition reaches `0.640` value accuracy on that
  proxy.

Not supported yet:

- Full-scale non-oracle direct-SQL control evidence.
- A clean local holdout claim.
- A promoted predicted-planner, semantic-layer, metric-DSL, or recovery win.
- Hosted baseline or BIRD-Interact/SOTA comparison claims.
