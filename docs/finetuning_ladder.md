# Finetuning Ladder

This file is the short operational map for the SQL finetuning program. The
research narrative lives in `docs/research_goal.md`; the broader evaluation
rules live in `docs/methodology.md`, `docs/evidence_contract.md`, and
`docs/finetuning_measurement_plan.md`.

The ladder exists so new work does not turn into a pile of unrelated runs. Each
step says what the model is supposed to learn, what it should be compared
against, and what evidence would make the step useful.

The executable registry for these method arms is
this document and the measurement plan. Keep them aligned when adding a method,
changing a control arm, or tightening the evidence requirement. For the
practical per-method runbook, use `docs/finetuning_method_runbook.md`.

## Ground Rules

- Direct SQL is the control arm.
- Structured methods must beat the right control on the same rows.
- Oracle hints are diagnostic only.
- Future turns, reference SQL, gold plans, gold metric DSL, expected rows, and
  repair labels must not enter production-style prompts.
- Training outputs belong in `outputs/`; scored generations and comparison
  manifests belong in `results/`.
- Checked-in artifacts should be small canonical inputs. See
  `docs/data_artifacts/README.md`.

## Stage 0: Direct SQL Control

Question: can ordinary supervised finetuning improve SQL generation on the fixed
multi-turn proxy?

Input shape: chat-format rows with schema and visible dialog history, but no
oracle planning hints.

Control arm: none. This is the control arm for later stages.

Win condition: value-only execution accuracy improves on the fixed CoSQL proxy
slice without oracle prompt inputs.

Current status: measured on the CoSQL proxy, but still not a hosted or
BIRD-Interact claim.

## Stage 1: Structured Query Brief SQL

Question: does a compact visible query brief before SQL improve execution?

Input shape: the model emits a short brief covering intent, entities and values,
metrics or measures, filters, grouping and grain, table families or joins, and
final answer shape before the final SQL. Training-split reference SQL may
supervise the brief target; clean-holdout reference SQL and expected rows stay
scorer-side only.

Control arm: same model, same rows, direct SQL prompt.

Win condition: structured-brief SQL beats direct SQL on value-only execution
accuracy in a same-row comparison from `eval.run_structured_brief_comparison`.

Current status: the comparison scorer exists. The repo still needs train-split
brief supervision and a clean-holdout endpoint comparison.

## Stage 2: Semantic-Layer Tuning

Question: does governed semantic context help with entities, dimensions,
measures, grain, joins, and value meaning that raw schema text misses?

Input shape: SQL rows with non-oracle semantic context or retrieval artifacts.

Control arm: same rows without the semantic-layer context.

Win condition: semantic-layer tuning or prompting improves same-row SQL
execution without oracle pruning.

Current status: semantic context has helped some prompt conditions, and the
value-index coverage artifact is tracked. The SQL win is still not proven until
`eval.compare_semantic_value_retrieval` shows a positive same-row delta versus
direct SQL.

## Stage 3: `MEASURE()` Metric DSL

Question: should the model preserve metric intent first and let a compiler
expand the governed SQL later?

Input shape: the model predicts a small metric DSL such as `MEASURE(revenue)`
rather than raw SQL directly. The evaluator scores parse rate, metric
preservation, compilation, and optional SQL execution.

Control arm: direct SQL on the same metric-heavy rows.

Win condition: compiled metric-DSL SQL beats direct SQL on same-row value
accuracy, while preserving the governed metric intent.

Current status: the parser/evaluator exists. The repo still needs real
checkpoint-generated metric-DSL evidence before making a win claim.

## Stage 4: Behavior And Recovery

Question: can the model continue after its own earlier outputs instead of
depending on clean teacher-forced history?

Input shape: generated-history rollout, where later turns see previous model
SQL rather than prior reference SQL.

Control arm: direct SQL trained on the same generated-history repair rows.
Teacher-forced history is a diagnostic comparison for rollout viability, not
the recovery method's win condition.

Win condition: rollout-vs-teacher-forced metrics explain the diagnostic gap,
then the recovery-tuned adapter beats the direct-SQL control adapter under
generated-history rollout.

Current status: the rollout evaluator exists, and tiny recovery/control SFT rows
exist for smoke testing. Recovery is not proven until generated predictions from
the recovery-tuned adapter clear side-by-side rollout comparison manifests.

## Stage 5: Hosted And BIRD-Interact Gate

Question: can the best local candidate compete with hosted baselines on the
target interaction protocol?

Input shape: frozen interactive benchmark inputs with result manifests, model
metadata, latency, and cost accounting.

Control arm: hosted model baselines and the best local direct-SQL candidate.

Win condition: the local finetuned model beats the hosted baseline under the
same interaction protocol and scorer.

Current status: target direction only. Current evidence remains a CoSQL proxy,
not a hosted SOTA or BIRD-Interact claim.

## How To Add A New Step

Do not start by adding many generated files. Start with one small PR that states:

- the method hypothesis,
- the canonical input rows or fixture family,
- the control arm,
- the scorer or comparison command,
- the leakage boundary,
- and where runtime outputs should be written.

Only add tests when the change touches shared behavior, scoring correctness,
row matching, or leakage prevention.
