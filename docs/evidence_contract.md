# Evidence Contract

This repo is testing whether a local Qwen 3.5 9B model can become useful on
BIRD-Interact-style multi-turn SQL. The current reproducible evidence is still a
CoSQL proxy slice, not a BIRD-Interact score and not a hosted-model comparison.
The broader methodology, dataset roles, training boundaries, and benchmark
rules are defined in `docs/methodology.md`.

## Evaluation Modes

| Mode | Inference inputs | Allowed public claim |
| --- | --- | --- |
| `non_oracle_generation` | Question, conversation history, schema, semantic context, and non-oracle prompt variants | A deployable path improved on the fixed proxy slice. |
| `predicted_planner` | The same inputs plus a plan predicted without reference SQL | A production-style planner-to-SQL path can be evaluated. |
| `oracle_planner_diagnostic` | Gold SQL-derived planning hints or schema pruning from those hints | A ceiling test for how much planning/schema linking matters. |

Gold SQL-derived labels may be used as scorer targets in every mode. They may
enter the prompt only in `oracle_planner_diagnostic`.

## Claim Ledger

| Claim | Status | Artifact | Mode | Allowed in blog |
| --- | --- | --- | --- | --- |
| Base local Qwen 3.5 9B reaches `0.370` strict and `0.590` value accuracy on the fixed CoSQL proxy turns. | Supported | `docs/result_manifests/cosql_dev_100_proxy.json` | `non_oracle_generation` | Yes, as a proxy result. |
| The 100-step LoRA reaches `0.530` strict and `0.630` value accuracy on the same proxy turns. | Supported | `docs/result_manifests/cosql_dev_100_proxy.json` | `non_oracle_generation` | Yes, as a proxy result. |
| The best non-oracle prompt/result currently reaches `0.640` value accuracy. | Supported | `docs/result_manifests/cosql_dev_100_proxy.json` | `non_oracle_generation` | Yes, if labeled value-only. |
| A prompt-only oracle diagnostic reaches `0.850` value accuracy. | Supported | `docs/result_manifests/cosql_dev_100_proxy.json` | `oracle_planner_diagnostic` | Yes, only as a ceiling test. |
| Gold SQL-derived planning hints can push the best diagnostic run to `0.890` value accuracy. | Supported | `docs/result_manifests/cosql_dev_100_proxy.json` | `oracle_planner_diagnostic` | Yes, only as a ceiling test. |
| The lexical planner baseline has macro score `0.571`, table F1 `0.599`, column F1 `0.117`, and skeleton F1 `0.648`. | Supported | `docs/planner_baseline_cosql_dev_100_summary.json` | planner scoring | Yes, as planner quality, not SQL accuracy. |
| A non-oracle predicted planner improves SQL execution. | Pending | `data/processed/eval_cosql_dev_predicted_planner_100.jsonl` can now be generated | `predicted_planner` | No, until endpoint SQL results exist. |
| Local 9B competes with hosted large models. | Pending | none | not run | No. |
| Local 9B competes on real BIRD-Interact/Multi-BIRD. | Pending | none | not run | No. |

## Reproducible Proxy Commands

Create a CoSQL-only prepared artifact:

```bash
python -m data.prepare \
  --config configs/cosql_dev_planner.yaml \
  --section eval \
  --limit 100 \
  --output data/processed/eval_cosql_dev_100.jsonl \
  --manifest-output data/processed/eval_cosql_dev_100.manifest.json
```

Create the first 100-turn predicted-planner artifact:

```bash
python -m eval.planner_eval \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --limit 100 \
  --predicted-prepared-output data/processed/eval_cosql_dev_predicted_planner_100.jsonl \
  --output results/planner_eval_cosql_dev_100.jsonl \
  --summary-output results/planner_eval_cosql_dev_100_summary.json
```

Future endpoint runs through `eval.run_eval` write a manifest next to the JSONL
output by default. Historical manifest snapshots for the currently cited proxy
numbers live in `docs/result_manifests/cosql_dev_100_proxy.json`.

## Blog Rule

A blog sentence can make a benchmark claim only if it names one of:

- `non_oracle_generation` for deployable proxy results;
- `predicted_planner` for non-oracle planner-to-SQL results;
- `oracle_planner_diagnostic` for ceiling tests;
- `pending` for work that has not been run.

If the sentence cannot be labeled with one of those modes, it is probably too
vague to publish.
