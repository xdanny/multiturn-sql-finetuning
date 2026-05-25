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
- Teacher-forced history tests whether the model can use clean prior SQL, not whether it
  can recover from its own mistakes.
- Oracle planner diagnostics show that planning is valuable, but they do not prove the
  model can produce the plan.
- Semantic context exists, but it is derived from schema metadata rather than a governed
  semantic model with explicit measures, dimensions, grain, and joins.
- DSPy has been used for prompt variants; it still needs to optimize planner and semantic
  programs, not just final SQL wording.

## Notebook-Driven Blog Contract

Every blog chapter in `docs/blog/` should have a matching marimo notebook in
`notebooks/blog/`. The post explains the narrative. The notebook reruns the relevant
tables, plots, and claim checks from tracked artifacts.

This matters because the blog should not be a static story written after the fact. It
should be a guided run through the evidence:

1. Load the artifact.
2. Compute the table or graph.
3. State what the evidence supports.
4. State what it does not support.
5. Point to the next experiment.

The notebooks are intentionally lightweight at first. Expensive model serving and GPU
training stay in scripts, while notebooks read tracked manifests, summaries, results,
and failure taxonomies. As the project matures, each notebook can grow from analysis
artifact into a reproducible evaluation app.

## First DSL Experiment Surface

The first implementation slice is the metric DSL in `data.metric_dsl`. It is small on
purpose: parse `MEASURE(...)` intent, compile it through a semantic model, and score
whether a model preserved the governed metric rather than expanding raw SQL too early.

This gives the next fine-tuning run a concrete target:

1. Train direct SQL.
2. Train metric DSL first.
3. Compile DSL to SQL through the same semantic model.
4. Compare both execution accuracy and semantic-intent scores.
