# Multi-Turn SQL Fine-Tuning

This repo tests one practical question:

Can a local Qwen LoRA learn enough multi-turn analytical SQL behavior to beat
its raw base model, and how far is it from a strong hosted model when every
model is scored on the same rows?

The project is not a generic text-to-SQL demo. It is a measured research
workspace for multi-turn SQL: follow-up questions, carried context, value
grounding, generated-history recovery, and same-row comparisons.

`docs/research_roadmap.md` is the canonical roadmap. It replaces the old
gate-heavy workflow with four guardrails:

- split integrity;
- prompt leakage prevention;
- row identity matching;
- run manifests.

## Current Results

The current evidence is useful but still scoped to local CoSQL splits and a
bounded hosted comparison. It is not yet a BIRD-Interact or LiveSQLBench claim.

| Comparison | Rows | Value accuracy | Strict accuracy | What it means |
| --- | ---: | ---: | ---: | --- |
| Raw Qwen, clean holdout | 680 | 0.218 | 0.218 | direct-SQL base control |
| Direct-SQL LoRA, clean holdout | 680 | 0.349 | 0.349 | full local direct-SQL control improved |
| Structured-brief LoRA, clean holdout | 680 | 0.576 | 0.541 | strongest clean held-out SFT result listed here |
| Raw Qwen, generated-history slice | 43 | 0.558 | 0.302 | CP9 local base control |
| Recovery LoRA, same generated-history slice | 43 | 0.558 | 0.302 | tied raw Qwen; no transfer win |
| Claude Sonnet 4.6 through OpenRouter, same slice | 43 | 0.674 | 0.395 | current hosted target gap |

What we learned:

- Direct SQL fine-tuning helped, but did not solve multi-turn SQL.
- Structured query briefs produced the strongest clean held-out SFT result in
  the top-line table, at the cost of longer outputs and lower syntax rate than
  direct SQL.
- Semantic value retrieval has a narrow clean-holdout win, but it still needs a
  larger and simpler validation story.
- Metric DSL failed its clean generated-output comparison. That is recorded as
  negative evidence, not hidden.
- Generated-history recovery is much harder than teacher-forced evaluation. The
  best recovery adapter tied raw Qwen on the hosted-transfer slice.
- The old planner/oracle path was overbuilt for the claim we need. Useful label
  extraction remains as scorer-side metadata, but planner prompt injection is no
  longer an active workflow.

Inconclusive evidence stays visible. Failed or tied runs record what was
learned and what remains unproven so future writeups do not turn a dead end
into a success story.

## Repo Shape

- `data/prepare.py`: convert datasets into chat-style JSONL.
- `data/prepare_split.py`: prepare rows from frozen split manifests.
- `train/finetune.py`: Qwen LoRA fine-tuning with Unsloth and TRL.
- `eval/run_eval.py`: endpoint evaluation and result manifests.
- `eval/rollout_eval.py`: generated-history rollout evaluation.
- `eval/local_benchmark.py`: local adapter evaluation.
- `eval/compare_hosted_baseline.py`: same-row local-vs-hosted comparison.
- `configs/experiments.yaml`: compact registry of current experiments.
- `docs/research_roadmap.md`: checkpoint status, evidence, and next direction.
- `docs/training_runs/`: small checked-in summaries of important runs.
- `docs/data_artifacts/README.md`: rules for checked-in data artifacts.

Run-specific outputs live under `outputs/` and `results/`. Checked-in docs
should summarize evidence, not replace the raw run artifacts.

## Reproduce

Use `uv run --active --no-sync ...` for Python commands.

Audit roadmap status:

```bash
uv run --active --no-sync python -m eval.roadmap_status --format markdown
```

Prepare a frozen CoSQL split:

```bash
uv run --active --no-sync python -m data.prepare_split \
  --split-id cosql_dev_clean_holdout_v1 \
  --output data/processed/direct_sql_full/cosql_dev_clean_holdout_v1.jsonl \
  --manifest-output data/processed/direct_sql_full/cosql_dev_clean_holdout_v1.manifest.json \
  --source-root .
```

Validate training rows without loading GPU libraries:

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/direct_sql_full_non_oracle.yaml \
  --data data/processed/direct_sql_full/cosql_train_v1.jsonl \
  --validate-data-only
```

Run a bounded fine-tuning smoke:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run --active --no-sync python -m train.finetune \
  --config configs/direct_sql_full_non_oracle.yaml \
  --data data/processed/direct_sql_full/cosql_train_v1.jsonl \
  --max-steps 1 \
  --output-dir outputs/gpu_access_smoke_1step \
  --report-to none
```

Evaluate an OpenAI-compatible endpoint:

```bash
uv run --active --no-sync python -m eval.run_eval \
  --benchmark prepared \
  --endpoint http://localhost:8000/v1 \
  --model-name direct_sql_full_non_oracle_lora \
  --input data/processed/direct_sql_full/cosql_dev_clean_holdout_v1.jsonl \
  --output results/runs/direct_sql_full_non_oracle_control/lora_clean_holdout.jsonl \
  --manifest-output results/runs/direct_sql_full_non_oracle_control/lora_clean_holdout.manifest.json \
  --database-root data/raw/cosql_dataset/database
```

Compare a hosted run only after both manifests are row-matched:

```bash
uv run --active --no-sync python -m eval.compare_hosted_baseline \
  --local-manifest results/generated_history_recovery_clean_holdout/behavior_recovery_5steps_cp8_limit12.rollout.manifest.json \
  --hosted-manifest results/hosted_transfer_cp9/openrouter_claude_sonnet_4_6_cp9_limit12.rollout.manifest.json \
  --output results/hosted_transfer_cp9/behavior_recovery_5steps_cp8_limit12.vs_openrouter_claude_sonnet_4_6.manifest.json
```

Run tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --active --no-sync pytest
```

## Data And Leakage Rules

Model prompts may use the current question, visible conversation history,
schema, semantic context, database-derived value indexes, and generated prior
SQL when the run is explicitly a generated-history rollout.

Model prompts must not include future turns, reference SQL, expected rows, gold
Metric DSL, repair labels, or answer-key-derived planner hints. Scorer-side
labels may exist in expanded rows for auditing and failure analysis, but they
are not model-facing context.

Clean claims require:

1. frozen split roles;
2. no answer-key leakage in prompts;
3. identical row identity between method and control;
4. result manifests with hashes, metrics, model names, commands, and costs when
   available.

## GPU Notes

This repo is run from WSL. GPU visibility can differ between the sandbox and the
real shell. Before diagnosing training, check `/usr/lib/wsl/lib/nvidia-smi` and
a PyTorch CUDA probe from the real WSL environment. `uv.lock` is local
environment churn here and should not be committed.

## Direction

The next credible claim is a clean benchmark story, not another complicated
planner abstraction:

1. keep scaling direct-SQL and structured-brief training on clean split roles;
2. compare against raw Qwen and hosted models on the same multi-turn rows;
3. use failure analysis to decide whether semantic context, value retrieval,
   Metric DSL, or recovery deserves the next training run;
4. move to BIRD-Interact-style evaluation only after the local protocol is
   stable and honest.
