# Research Goal

This repository is no longer just "fine-tune Qwen on SQL." The research question is:

> Can a small specialized local model learn the behavior and semantic concepts needed
> to outperform state-of-the-art general models on multi-turn analytical SQL?

The motivation is the gap between single-turn benchmark progress and real analytical
work. Large models are increasingly strong on single-shot text-to-SQL benchmarks such
as BIRD, but multi-turn analysis asks for more than one correct query. It requires
state, ambiguity handling, schema understanding, value grounding, metric semantics,
repair, and recovery.

The repo should therefore read like an experiment, not like a collection of SQL
utilities. The starting claim is narrow:

> Zero-shot prompting can look strong on single-turn SQL while still failing the
> behaviors that make data analysis conversational.

The project is valuable only if it can show whether specialization changes that
behavior. A local model does not need to be generally smarter than a hosted model.
It needs to learn the intermediate decisions that hosted zero-shot prompting often
leaves implicit: what the user is still asking about, which values and entities are
meant, which metric definition is governed, and which query grain is allowed.

## What Would Count As Success

A serious success claim needs all of these:

1. A fixed interactive evaluation protocol, first on CoSQL/SParC-style proxy tasks and
   then on BIRD-Interact-style tasks.
2. Hosted-model baselines run through the same prepared inputs, execution scorer,
   result manifest, and latency/cost accounting.
3. A local model manifest compared against those hosted rows with a positive value
   delta; merely publishing the hosted baseline is not the win.
4. A local model path that wins on multi-turn execution outcomes, not only on isolated
   SQL syntax.
5. Evidence that the win comes from learnable intermediate behavior: planning,
   semantic-layer reasoning, value grounding, clarification, or repair.
6. No oracle planning hints in production-style claims.

## Experiment Ladder

The comparison should move in this order. Skipping a rung makes the result hard to
interpret.

1. **Direct SQL control**: same rows, same local endpoint path, no oracle hints. This
   says whether ordinary SFT helps the fixed multi-turn proxy.
2. **Structured query brief before SQL**: train a compact visible brief from
   training-split supervision only, then compare brief-first SQL against the
   direct SQL control on row-matched manifests.
4. **Semantic-layer target**: retrieve or predict governed entities, dimensions,
   measures, grain, and value aliases. Score those artifacts before mixing them into
   generation prompts.
5. **`MEASURE()`-preserving DSL**: predict metric intent first, compile it through a
   semantic model, then compare compiled SQL against direct SQL on the same metric-heavy
   rows.
6. **Generated-history rollout**: stop teacher-forcing clean previous SQL and test
   whether the model can continue after its own earlier misses.
7. **Hosted and BIRD-Interact comparison**: only after the local protocol is stable,
   run hosted baselines and BIRD-Interact-style tasks with the same scorer, manifest,
   latency, and cost accounting.

Each rung should produce an artifact the blog can cite directly: a prepared
input manifest, a structured-brief comparison manifest, a semantic-artifact
report, a metric-DSL comparison, a rollout manifest, or a hosted comparison. A
prose claim without one of those artifacts is not ready for publication.

The hosted comparison rung must spell out the same input rows, scorer, oracle
boundary, hosted model manifest, local model manifest, generated-history
rollout, latency and cost accounting, and BIRD-Interact transfer evidence
required before a hosted/SOTA claim can move.

Each method should name the module and command surface that can produce or check
its evidence. A method is not "next" unless there is an executable path for
preparing, scoring, or comparing it against the right control.

## Fine-Tuning Methods To Compare

The repo should compare training methods as first-class hypotheses:

| Method | Hypothesis | Required evidence |
| --- | --- | --- |
| Direct SQL SFT | A small model can learn the target SQL distribution directly from chat-format SQL rows. | Non-oracle execution accuracy improves on fixed proxy and BIRD-Interact-style evals. |
| Structured brief or DSL first, SQL second | The model should first produce a compact visible query brief or typed DSL, then generate or compile SQL. | Same-row SQL execution beats direct SQL, with the intermediate artifact scored separately. |
| Semantic-layer tuning | The model should learn governed entities, dimensions, measures, grain, and allowed joins. | Semantic artifact retrieval and use improves metric and join correctness. |
| MEASURE()-preserving metric DSL | The model should preserve governed metrics such as `MEASURE(revenue)` instead of expanding metric SQL too early. | Metric DSL accuracy and compiled SQL execution beat raw SQL generation on metric-heavy tasks. |
| Behavior/recovery tuning | The model should learn to clarify, inspect values, repair failures, and recover after its own earlier errors. | Recovery-adapter rollout beats the direct-SQL control under generated-history evaluation; rollout-vs-teacher-forced remains a diagnostic comparison. |

## Why The Current Work Is Incomplete

The current repo has a useful local loop, but it is still mostly a proxy:

- CoSQL gives a reproducible multi-turn slice, but not the final BIRD-Interact claim.
- SParC adds related context-dependent SQL signal, but it is not the same as an
  interactive generated-history evaluation.
- BIRD mini-dev is useful for single-turn execution-harness checks, but it cannot
  prove multi-turn data-analysis behavior by itself.
- Synthetic schema-rich SQL should be used for controlled value-grounding, grain,
  join-fanout, and `MEASURE()` preservation fixtures, not blended into one
  anonymous fine-tuning pile.
- Teacher-forced history tests whether the model can use clean prior SQL, not whether it
  can recover from its own mistakes.
- Oracle planner diagnostics show that decomposition can help, but they do not prove
  a deployable model can produce or use that decomposition.
- Semantic context exists, but it is derived from schema metadata rather than a governed
  semantic model with explicit measures, dimensions, grain, and joins.
- DSPy has been used for prompt variants; it still needs to optimize structured
  brief, semantic, and recovery programs, not just final SQL wording.

## Method Decision Rules

The repo should not rank fine-tuning methods by vibes, prompt length, or isolated
accuracy numbers. Each candidate has to beat the right control arm on the same
rows, with the same scorer, the same prompt boundary, and the same oracle policy:

- **Direct SQL SFT** is the control arm. It defines the baseline every structured
  target must beat; it does not answer the hosted-SOTA question alone.
- **Structured brief or DSL first, SQL second** wins only if the structured
  intermediate artifact can be produced without clean-holdout labels and the
  resulting SQL beats direct SQL on matching rows. Checkpoint 5 uses
  `eval.run_structured_brief_comparison` for that comparison so row identity,
  scorer, and oracle policy stay aligned.
- **Semantic-layer tuning** wins only if versioned semantic artifacts improve value
  grounding, entity resolution, joins, and grain without simply flooding the prompt.
- **MEASURE()-preserving metric DSL** wins only if generated DSL parses, compiles,
  preserves governed `MEASURE()` intent, and the compiled SQL matches or beats
  direct SQL on metric-heavy rows.
- **Behavior/recovery tuning** wins only in generated-history rollout, where the
  model has to recover from its own prior bad SQL or empty result instead of reading
  clean teacher-forced history.

These rules should change only when evaluation code and comparison artifacts can
enforce the new rule.

## Data Artifact Contract

The next work is data engineering as much as model training. The repo needs a
`data_artifact_contract` for the intermediate state a multi-turn SQL model must learn:

| Artifact | Failure it isolates | Why it matters |
| --- | --- | --- |
| Value index | Display-to-storage mismatches such as `France -> FR`, aliases, abbreviations, dates, and entity names. | Prevents executable SQL from silently returning empty or wrong rows. |
| Entity-resolution labels | Follow-up mentions bind to the wrong entity, table role, value, or previous turn. | Separates context resolution from final SQL generation. |
| Grain and fanout fixtures | Bridge tables, duplicated child rows, and many-to-many joins inflate metrics. | Tests whether a model understands analytical grain, not only syntax. |
| Semantic model manifest | Governed measures, dimensions, joins, grain, and `MEASURE()` definitions. | Makes semantic-layer and metric-DSL claims reproducible. |
| Schema/alias validator output | Wrong-table columns and ambiguous aliases collapse into generic execution errors. | Turns bad SQL into repairable planner/generator labels. |
| Generated-history trace | Teacher-forced history hides failures after the model's own bad turns. | Makes behavior and recovery tuning measurable. |

These artifacts should be versioned and referenced by manifests before the blog claims
that a method learned the corresponding behavior.

The first landed label artifact is
`docs/data_artifacts/value_grounding_labels_cosql_dev_100.jsonl`, with a matching
summary and manifest. It is intentionally narrow: it derives value-grounding
labels from gold/reference SQL on the fixed CoSQL proxy slice. That makes it
useful for supervision, coverage, and evaluation, but it does not prove a
production system can find those values without the answer.

The first non-oracle companion artifact is
`docs/data_artifacts/value_index_cosql_dev_100.jsonl`, generated from SQLite
database contents rather than gold SQL. Its coverage summary now shows the
remaining semantic-layer gap directly: stored values are often present, but
user-facing aliases are not always recoverable. The next step is entity/alias
expansion and retrieval scoring before SQL generation.

The first synthetic schema-rich fixture pack is
`docs/data_artifacts/synthetic_method_fixtures.jsonl`. It is not a benchmark and
does not support a hosted-SOTA claim. It is a curated data-engineering surface for
testing the method hypotheses before endpoint runs: value normalization, entity
resolution, bridge-table fanout, governed `MEASURE()` preservation, and
empty-result recovery. Each row keeps reference SQL out of prompt-visible inputs
and records the artifacts needed to score the failure mode.

## Blog-Attached Code Lab Contract

The public post should link to the attached codebase and one published HTML lab
at `/labs/local-multiturn-sql-finetuning/`. The same walkthrough remains
runnable from the codebase:

1. `notebooks/labs/local_multiturn_sql_lab.py`
2. `notebooks/labs/local_multiturn_sql_lab.ipynb`

The post explains the narrative, but every public claim should name a lab
checkpoint, run manifest, or canonical data artifact that backs it. The lab
auto-selects CUDA, MPS, or XPU when PyTorch detects an available
accelerator and falls back to CPU. It should not require the full GPU training
setup, dependency installation cells, or vLLM serving path.

This matters because the blog should not be a static story written after the fact. It
should behave like a lab walkthrough inside the attached codebase:

1. Start with the benchmark gap: zero-shot BIRD-style SQL strength does not imply
   robust multi-turn analysis.
2. Define the evaluation protocol and dataset roles before making model claims.
3. Compare fine-tuning targets directly: direct SQL, structured query briefs,
   semantic-layer, `MEASURE()` DSL, and behavior/recovery.
4. Separate production-style proxy results from oracle diagnostics.
5. Turn the remaining failures into next repo artifacts.

Every public claim should name a lab section, run manifest, or comparison
artifact that produced it. The public post should be readable
on its own, but it should also let readers rerun the companion lab to see the
boundary between current proxy evidence, oracle diagnostics, and future
BIRD-Interact or hosted-model claims.
Run manifests and comparison artifacts are the bridge between toy lab behaviors
and model evidence, so readers can see which targets are supported, pending, or
only diagnostic.

Before the blog treats structured query briefs as the next result, the repo must
show that brief-first SQL beats the direct-SQL control on the same clean-holdout
rows.

Expensive model serving and GPU training stay in scripts. The public reader path
stays focused on the published lab, the attached codebase, and cited run
manifests.

## First DSL Experiment Surface

The first implementation slice is the metric DSL in `data.metric_dsl` plus the
offline evaluator in `eval.metric_dsl_eval`. It is small on purpose: parse
`MEASURE(...)` intent, compile it through a semantic model, score whether a
model preserved the governed metric rather than expanding raw SQL too early, and
write a manifest that can later be compared against direct SQL.

This gives the next fine-tuning run a concrete target:

1. Train direct SQL.
2. Train metric DSL first.
3. Compile DSL to SQL through the same semantic model.
4. Compare both execution accuracy and semantic-intent scores.
