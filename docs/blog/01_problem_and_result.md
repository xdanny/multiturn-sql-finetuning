# Can a Local 9B Model Compete on Multi-Turn SQL?

Notebook: `notebooks/blog/01_problem_and_result.py`

The project starts from a concrete benchmark question:

> Can a locally fine-tuned Qwen 3.5 9B model become useful enough on
> BIRD-Interact-style multi-turn SQL tasks to compete with much larger hosted
> models?

That is different from a generic "fine-tune a model on SQL chats" project. The
benchmark target matters, the local serving constraint matters, and the
comparison class matters. CoSQL is currently the small reproducible proxy slice;
BIRD-Interact is the direction of the actual series.

That distinction matters. Single-turn text-to-SQL is already a schema-linking
problem. Multi-turn text-to-SQL adds state. A later question may say "what about
the top five by revenue?" or "only include the ones from last year." The model
has to carry forward the previous target table, filters, grouping level, metric,
and sometimes the user's intended comparison. If it loses any of that state, the
SQL can still look plausible while answering the wrong question.

This repo now has a complete local loop:

- Prepare chat-format SQL data from CoSQL, SParC, and synthetic schema-rich SQL.
- Fine-tune `unsloth/Qwen3.5-9B` with LoRA on an RTX 5090.
- Serve the base model and adapters through a local vLLM OpenAI-compatible API.
- Evaluate on the same 100 assistant turns from CoSQL dev, spanning 32 dialogs.
- Report both per-turn execution accuracy and dialog-level metrics.

This loop does not yet produce a BIRD-Interact score. It proves the local
training and evaluation path is sensitive enough to iterate before the harder
benchmark harness is finished.

The best non-oracle adapter so far is still the 100-step non-semantic run. It
moved the vLLM endpoint from `0.370` to `0.530` execution accuracy on the fixed
100-turn slice. Later oracle-labelled experiments reach higher numbers, but
those use planning labels extracted from reference SQL and should be read as
upper-bound diagnostics, not production-path results.

| Model | Execution accuracy | Dialog execution accuracy | Interaction match rate |
| --- | ---: | ---: | ---: |
| Base `unsloth/Qwen3.5-9B` | 0.370 | 0.347 | 0.0625 |
| `multiturn-sql-50` | 0.420 | 0.382 | 0.09375 |
| `multiturn-sql-128each-50` | 0.420 | 0.389 | 0.12500 |
| `multiturn-sql-semantic-50` | 0.420 | 0.384 | 0.09375 |
| `multiturn-sql-100` | 0.530 | 0.515 | 0.15625 |

This is not a leaderboard result. It is a small, controlled engineering result.
The value is that the loop now produces a quality gradient. A 50-step adapter
beats the base model. A 100-step adapter beats the 50-step adapter. Doubling the
data without increasing steps does not automatically help. Adding automatically
derived semantic hints improves prompt structure and syntax reliability, but in
this first small run it does not improve per-turn execution accuracy over the
plain 50-step adapter.

That last point is the most important lesson from this phase. "Add a semantic
layer" is not magic. The useful version is more exacting: define entities,
grain, measures, dimensions, allowed joins, and metric semantics well enough
that the model can learn stable decisions from them. A shallow semantic summary
derived mechanically from `tables.json` is better than no structure, but it is
not yet the same thing as a governed analytics model.

## Why CoSQL Is the First Proxy Slice

CoSQL is a conversational text-to-SQL dataset built from Wizard-of-Oz dialogs
over unseen databases. The official description emphasizes 30k+ turns, 10k+
annotated SQL queries, 3k dialogs, 200 databases, and 138 domains. More
importantly, it frames conversational text-to-SQL as dialogue state tracking
where the state is SQL, not a domain-specific slot map.

That makes CoSQL a useful engineering proxy for this repo. It exercises:

- Carrying context across turns.
- Resolving follow-up references.
- Generalizing to unseen database schemas.
- Handling ambiguous or unanswerable user questions.
- Reporting dialog-level quality, not only per-turn quality.

The current evaluator is intentionally teacher-forced. Previous gold assistant
SQL stays in the prompt history while the model predicts the current SQL turn.
That does not simulate a fully autonomous agent, because model errors do not
compound across the conversation. It does isolate a narrower skill: given clean
history, can the model use it?

That is the right intermediate gate for fine-tuning. If a model cannot use gold
history reliably on CoSQL, it will not survive a harder BIRD-Interact-style loop
where it has to handle dynamic interaction and recover from its own earlier
mistakes.

## What The Numbers Do And Do Not Say

Per-turn execution accuracy answers: "Did this generated SQL return the expected
result for this turn on this SQLite database?"

Dialog execution accuracy answers: "How well did the model do when scores are
aggregated by conversation?"

Interaction match rate answers: "For how many dialogs did every evaluated turn
score correctly?"

The interaction metric is harsh, and it should be. In an analytics assistant,
one bad follow-up answer can invalidate the user's thread of reasoning. A model
that is 53% correct per turn but only perfect on 15.6% of dialogs is not ready
to be trusted. It is, however, good enough to show where the next engineering
work should go.

The next phase should not chase another small prompt tweak first. It should
improve the data substrate:

- Better semantic models with real grain and metric definitions.
- Standalone resolved questions for follow-up turns.
- Explicit schema-linking labels or relevant-table supervision.
- Join-path and fanout tests.
- Execution error taxonomies.
- Regression suites for schema drift and metric drift.

The core research direction is consistent with QURG, RAT-SQL, RESDSQL, DIN-SQL,
and semantic-layer systems like Cube and dbt: a text-to-SQL model needs more
than raw DDL. It needs structured hints about what the user means and what the
database is allowed to mean.

Sources:

- Qwen 3.5 9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
- BIRD-Interact project page: https://bird-interact.github.io/
- BIRD-Interact paper: https://arxiv.org/abs/2510.05318
- CoSQL project page: https://yale-lily.github.io/cosql
- CoSQL paper: https://arxiv.org/abs/1909.05378
- SParC paper: https://arxiv.org/abs/1906.02285
- Cube semantic layer introduction: https://docs.cube.dev/docs/introduction
- dbt Semantic Layer docs: https://docs.getdbt.com/docs/build/semantic-models
