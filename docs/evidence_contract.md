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

The claim ledger turns this contract into a checked artifact. It reads result
manifests, verifies referenced hashes, scans input/output rows for oracle
planning leakage, joins classified failure counts, and writes one row per
claimable artifact plus explicit pending rows for missing evidence:

```bash
python -m eval.claim_ledger
```

Tracked outputs:

- `docs/claim_ledgers/cosql_dev_100.jsonl`
- `docs/claim_ledgers/cosql_dev_100_summary.csv`

The current ledger is intentionally conservative:

- non-oracle CoSQL results are `supported_proxy`;
- oracle-planner rows are `diagnostic_upper_bound`;
- planner summaries are `supported_planner_quality`, not SQL accuracy;
- generated-history rollout existence, rollout-vs-teacher-forced improvement,
  predicted-planner SQL execution, hosted/SOTA comparison, and BIRD-Interact
  comparison remain `pending` until same-protocol result manifests exist.

Any hash mismatch, missing manifest field, non-oracle oracle marker, or
predicted-planner manifest whose output rows are not also marked
`predicted_planner` is downgraded to `pending` with a blocker.

## Claim Ledger Rows

| Claim | Status | Artifact | Mode | Allowed in blog |
| --- | --- | --- | --- | --- |
| Base local Qwen 3.5 9B reaches `0.370` strict and `0.590` value accuracy on the fixed CoSQL proxy turns. | `supported_proxy` | `docs/result_manifests/cosql_dev_100_proxy.json` | `non_oracle_generation` | Yes, as a proxy result. |
| The 100-step LoRA reaches `0.530` strict and `0.630` value accuracy on the same proxy turns. | `supported_proxy` | `docs/result_manifests/cosql_dev_100_proxy.json` | `non_oracle_generation` | Yes, as a proxy result. |
| The best non-oracle prompt/result currently reaches `0.640` value accuracy. | `supported_proxy` | `docs/result_manifests/cosql_dev_100_proxy.json` | `non_oracle_generation` | Yes, if labeled value-only. |
| A prompt-only oracle diagnostic reaches `0.850` value accuracy. | `diagnostic_upper_bound` | `docs/result_manifests/cosql_dev_100_proxy.json` | `oracle_planner_diagnostic` | Yes, only as a ceiling test. |
| Gold SQL-derived planning hints can push the best diagnostic run to `0.890` value accuracy. | `diagnostic_upper_bound` | `docs/result_manifests/cosql_dev_100_proxy.json` | `oracle_planner_diagnostic` | Yes, only as a ceiling test. |
| The lexical planner baseline has macro score `0.571`, table F1 `0.599`, column F1 `0.117`, and skeleton F1 `0.648`. | `supported_planner_quality` | `docs/planner_baseline_cosql_dev_100_summary.json` | planner scoring | Yes, as planner quality, not SQL accuracy. |
| A non-oracle predicted planner improves SQL execution. | Pending | `data/processed/eval_cosql_dev_predicted_planner_100.jsonl` can now be generated | `predicted_planner` | No, until endpoint SQL results exist. |
| A generated-history rollout result exists for the fixed CoSQL proxy. | Pending | `eval.rollout_eval` is implemented | `non_oracle_generation` | No, until a rollout result manifest exists. |
| Generated-history rollout beats teacher-forced history for the same model/input. | Pending | none | not run | No, until side-by-side comparison metrics exist. |
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

Run generated-history rollout without teacher-forcing prior gold SQL:

```bash
python -m eval.rollout_eval \
  --model-name <served-model-name> \
  --endpoint http://127.0.0.1:8000/v1 \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --database-root data/raw/cosql_dataset/database \
  --output results/rollout/<run-id>.jsonl
```

Rollout rows use `history_policy=model_generated_sql_rollout`. A rollout result
can support a proxy rollout claim, but not a behavior/recovery improvement claim
until the ledger also has same-model teacher-forced comparison metrics.

## Blog Rule

A blog sentence can make a benchmark claim only if it names one of:

- `non_oracle_generation` for deployable proxy results;
- `predicted_planner` for non-oracle planner-to-SQL results;
- `oracle_planner_diagnostic` for ceiling tests;
- `pending` for work that has not been run.

If the sentence cannot be labeled with one of those modes, it is probably too
vague to publish.
