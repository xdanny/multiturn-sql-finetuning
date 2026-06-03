# Multi-Turn SQL Fine-Tuning

Fine-tune and evaluate small local models for multi-turn analytical SQL. The
goal is not one perfect demo query. The goal is a measured comparison between
raw local models, local fine-tunes, and hosted frontier models on follow-up data
questions.

The current proxy benchmark is CoSQL because it is small, reproducible, and
conversational. The longer target is a BIRD-Interact or LiveSQLBench-style
comparison with the same rows, same scorer, same leakage policy, hosted
baselines, latency, and cost.

## Current Result

The latest matrix compares raw Qwen, saved local fine-tuning adapters, and
OpenRouter Claude Sonnet 4.6 on the same 43 generated-history CoSQL turns.
Generated history means later turns use the model's earlier SQL, not a perfect
reference history. Value accuracy checks whether returned values match. Strict
accuracy also cares about the exact result shape.

| Arm | Value accuracy | Strict accuracy | Claim boundary |
| --- | ---: | ---: | --- |
| OpenRouter Claude Sonnet 4.6 | `0.674` | `0.395` | Hosted comparator. |
| Schema-pruned 100-step adapter | `0.651` | `0.465` | Diagnostic only; it uses answer-derived schema pruning. |
| Semantic 50-step adapter | `0.581` | `0.326` | Best clean semantic local adapter. |
| Direct-SQL 50-step adapter | `0.581` | `0.302` | Best clean direct-SQL local adapter by value accuracy. |
| Raw Qwen 9B | `0.558` | `0.302` | Local baseline with no adapter. |
| Behavior-recovery 5-step adapter | `0.558` | `0.302` | Clean run, tied raw Qwen here. |

The observed clean local gain is small on this slice: `+0.023` value accuracy
over raw Qwen. The schema-pruned diagnostic result is the strongest research
signal. It suggests schema selection may be a major bottleneck, but it does not
prove a deployable local model can choose that schema without help.

Evidence:

- `docs/training_runs/all_strategy_cp9_matrix_20260603.json`
- `docs/training_runs/hosted_transfer_openrouter_sonnet_4_6_20260603.json`
- `docs/research_roadmap.md`

## Rules

Results only count when the comparison is honest:

- split roles are explicit;
- prompts do not expose future turns, reference SQL, expected rows, gold plans,
  gold Metric DSL, repair labels, or answer-derived schema pruning;
- claimable schema selection must come only from inference-time-visible inputs:
  the user question, prior conversation, schema, database-derived indexes, and
  allowed retrieval artifacts;
- local and hosted runs use the same row identities;
- every reported run has a manifest with input hash, output hash, model,
  endpoint, mode, command, and metrics;
- diagnostic runs stay labeled as diagnostics.

## Repo Map

- `docs/research_roadmap.md`: checkpointed roadmap and current claim boundary.
- `docs/current_research_inventory.md`: audited adapters, manifests, and
  evidence snapshots.
- `configs/experiments.yaml`: compact registry of tested hypotheses.
- `data/splits/`: frozen split manifests for train, proxy, clean holdout,
  synthetic fixtures, and pending external targets.
- `eval.run_eval`: prepared SQL evaluation against local or endpoint models.
- `eval.run_behavior_recovery_comparison`: generated-history rollout evaluation.
- `eval.run_metric_dsl_comparison`: Metric DSL versus direct-SQL comparison.
- `docs/data_artifacts/README.md`: rules for checked-in data artifacts.
- `docs/blog/README.md`: public post and lab attachment notes.

## Common Commands

Check roadmap status:

```bash
uv run --active --no-sync python -m eval.roadmap_status --format markdown
```

Run tests:

```bash
uv run --active --no-sync pytest -q
```

Prepare data:

```bash
uv run --active --no-sync python -m data.prepare \
  --config configs/qwen35_9b_5090.yaml \
  --limit 100 \
  --output data/processed/train.jsonl
```

Run endpoint evaluation:

```bash
uv run --active --no-sync python -m eval.run_eval \
  --benchmark prepared \
  --endpoint http://127.0.0.1:8000/v1 \
  --model-name multiturn-sql-100 \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --limit 100 \
  --max-tokens 192 \
  --database-root data/raw/cosql_dataset/database \
  --output results/vllm_qwen35_9b_lora100_cosql_dev_100turns.jsonl
```

Run generated-history rollout comparison:

```bash
uv run --active --no-sync python -m eval.run_behavior_recovery_comparison \
  --backend local \
  --model-name unsloth/Qwen3.5-9B \
  --adapter-path outputs/qwen35_9b_multiturn_sql_semantic_50steps/final \
  --input docs/data_artifacts/behavior_recovery_rollout_inputs.jsonl \
  --output-dir results/example_rollout \
  --run-id semantic_50step_example \
  --database-root data/raw/cosql_dataset/database
```

## Current Constraints

- CoSQL comes from the official Yale/Google Drive archive, not the stale
  `alpineai/cosql` Hugging Face ID.
- The accessible `jellyChiru/SParC` mirror is flattened to question/query rows,
  not full dialogs.
- BIRD mini-dev uses splits such as `mini_dev_sqlite`, not `validation`.
- vLLM serving is separate from training: `.venv` is for Unsloth training and
  `.venv-vllm` is for serving.
- Qwen thinking mode must be disabled during endpoint evaluation.
- WSL PyTorch CUDA works on this machine, but source builds may still need a
  full CUDA toolkit.

## Blog-Attached Lab

The public blog post points readers to the attached codebase and a published
HTML lab at `/labs/local-multiturn-sql-finetuning/`. The lab is a lightweight
reader path, not the evidence source for benchmark claims.

Run it locally with Marimo:

```bash
uv run --active --no-sync marimo edit notebooks/labs/local_multiturn_sql_lab.py
```
