# Research Goal

This repository is no longer just "fine-tune Qwen on SQL." The research question is:

> Can a small specialized local model learn the behavior and semantic concepts needed
> to outperform state-of-the-art general models on multi-turn analytical SQL?

The motivation is the gap between single-turn benchmark progress and real analytical
work. Large models are increasingly strong on single-shot text-to-SQL benchmarks such
as BIRD, but multi-turn analysis asks for more than one correct query. It requires
state, ambiguity handling, schema understanding, value grounding, metric semantics,
repair, and recovery.

## What Would Count As Success

A serious success claim needs all of these:

1. A fixed interactive evaluation protocol, first on CoSQL/SParC-style proxy tasks and
   then on BIRD-Interact-style tasks.
2. Hosted-model baselines run through the same prepared inputs, execution scorer,
   result manifest, and latency/cost accounting.
3. A local model path that wins on multi-turn execution outcomes, not only on isolated
   SQL syntax.
4. Evidence that the win comes from learnable intermediate behavior: planning,
   semantic-layer reasoning, value grounding, clarification, or repair.
5. No oracle planning hints in production-style claims.

## Fine-Tuning Methods To Compare

The repo should compare training methods as first-class hypotheses:

| Method | Hypothesis | Required evidence |
| --- | --- | --- |
| Direct SQL SFT | A small model can learn the target SQL distribution directly from chat-format SQL rows. | Non-oracle execution accuracy improves on fixed proxy and BIRD-Interact-style evals. |
| Planner/DSL first, SQL second | The model should first predict a typed plan or DSL, then compile or generate SQL. | Planner F1 improves and predicted-plan SQL execution beats direct SQL. |
| Semantic-layer tuning | The model should learn governed entities, dimensions, measures, grain, and allowed joins. | Semantic artifact retrieval and use improves metric and join correctness. |
| MEASURE()-preserving metric DSL | The model should preserve governed metrics such as `MEASURE(revenue)` instead of expanding metric SQL too early. | Metric DSL accuracy and compiled SQL execution beat raw SQL generation on metric-heavy tasks. |
| Behavior/recovery tuning | The model should learn to clarify, inspect values, repair failures, and recover after its own earlier errors. | Rollout evaluation with model-generated history beats teacher-forced history evaluation. |

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
- Oracle planner diagnostics show that planning is valuable, but they do not prove the
  model can produce the plan.
- Semantic context exists, but it is derived from schema metadata rather than a governed
  semantic model with explicit measures, dimensions, grain, and joins.
- DSPy has been used for prompt variants; it still needs to optimize planner and semantic
  programs, not just final SQL wording.

## Method Decision Rules

The repo should not rank fine-tuning methods by vibes, prompt length, or isolated
accuracy numbers. Each candidate has to beat the right control arm on the same
rows, with the same scorer, the same prompt boundary, and the same oracle policy:

- **Direct SQL SFT** is the control arm. It defines the baseline every structured
  target must beat; it does not answer the hosted-SOTA question alone.
- **Planner/DSL first, SQL second** wins only if a non-oracle planner improves
  planner labels and the resulting SQL beats direct SQL execution on matching rows.
- **Semantic-layer tuning** wins only if versioned semantic artifacts improve value
  grounding, entity resolution, joins, and grain without simply flooding the prompt.
- **MEASURE()-preserving metric DSL** wins only if generated DSL parses, compiles,
  preserves governed `MEASURE()` intent, and the compiled SQL matches or beats
  direct SQL on metric-heavy rows.
- **Behavior/recovery tuning** wins only in generated-history rollout, where the
  model has to recover from its own prior bad SQL or empty result instead of reading
  clean teacher-forced history.

The generated `method-decision-rules.md` and `method-priority-backlog.md` assets are
the public form of these rules. They should change only when the claim ledger and
evaluation code can enforce the new rule.

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

## Blog-Attached Marimo Lab Contract

The public post should link to the attached codebase and one primary Marimo
walkthrough:

1. `notebooks/labs/local_multiturn_sql_lab.py`
2. `notebooks/labs/local_multiturn_sql_lab.ipynb`

The post explains the narrative, but every public claim should name a lab
checkpoint or generated evidence artifact that backs it. The lab
auto-selects CUDA, MPS, or XPU when PyTorch detects an available
accelerator and falls back to CPU. It should not require the full GPU training
setup, dependency installation cells, or vLLM serving path.

This matters because the blog should not be a static story written after the fact. It
should behave like a lab walkthrough inside the attached codebase:

1. Start with the benchmark gap: zero-shot BIRD-style SQL strength does not imply
   robust multi-turn analysis.
2. Define the evaluation protocol and dataset roles before making model claims.
3. Compare fine-tuning targets directly: direct SQL, planner-first, semantic-layer,
   `MEASURE()` DSL, and behavior/recovery.
4. Separate production-style proxy results from oracle diagnostics.
5. Turn the remaining failures into next repo artifacts.

Every public claim should name a lab notebook section or generated evidence artifact
that produced it. The public post should be readable
on its own, but it should also let readers rerun the companion Marimo lab to see the
boundary between current proxy evidence, oracle diagnostics, and future
BIRD-Interact or hosted-model claims.
The generated `target-evidence-matrix.md` is the public bridge between the toy lab
behaviors and manifest-backed model evidence, so readers can see which targets are
supported, pending, or only diagnostic.

Expensive model serving and GPU training stay in scripts. The public reader path stays
focused on the shareable Marimo lab and generated evidence assets.

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
