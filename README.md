# Multi-Turn SQL Fine-Tuning

This repo tests a practical question:

Can a local Qwen LoRA answer follow-up analytical SQL questions better than the
raw model, and can we measure that against stronger hosted models without
leaking the answer key?

The current evidence is useful but still early. The best clean local result in
this checkout is a 100-step direct-SQL LoRA on a fixed 100-turn CoSQL dev proxy
slice:

| Run | Rows | Value accuracy | Strict accuracy | Syntax accuracy | What it means |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw Qwen 9B baseline | 100 | 0.590 | 0.370 | not recorded here | baseline control |
| Direct SQL LoRA, 100 steps | 100 | 0.630 | 0.530 | 1.000 | best clean local adapter recorded here |
| Semantic-context LoRA, 50 steps | 100 | 0.640 | 0.420 | 1.000 | slightly better value score, worse strict score |
| Answer-key-pruned diagnostic LoRA, 100 steps | 100 | 0.890 | 0.820 | 1.000 | diagnostic only; not a production claim |

The diagnostic row is important but should not be confused with the clean
result. It shows that table/column/value narrowing can matter a lot, but it used
reference-derived context that would not be available at inference time.

## What This Repo Does

- Prepares CoSQL, SParC, BIRD, and small synthetic rows into chat-style JSONL.
- Fine-tunes Qwen 3.5 9B with LoRA through Unsloth and TRL.
- Evaluates raw models and adapters through local generation or an
  OpenAI-compatible endpoint.
- Scores SQL by execution against SQLite when database files are available.
- Records value accuracy, strict accuracy, syntax validity, row identity, and
  run manifests for same-row comparisons.
- Keeps small synthetic artifacts for method smoke tests: Metric DSL, value
  retrieval, alias/column context, and generated-history recovery.

The core loop is intentionally simple now:

1. Prepare clean model-facing rows.
2. Train or serve one model variant.
3. Evaluate on the same row IDs as the control.
4. Rescore and classify failures.
5. Compare against raw Qwen and hosted models under the same protocol.

## What We Tried

Direct SQL fine-tuning helped on the 100-turn CoSQL proxy slice: value accuracy
moved from `0.590` to `0.630`, and strict accuracy moved from `0.370` to
`0.530`.

Semantic context helped less cleanly. The 50-step semantic adapter reached
`0.640` value accuracy, but strict accuracy stayed lower at `0.420`. That makes
it a lead, not a settled win.

Metric DSL and recovery are not yet benchmark wins. The current DSL and recovery
artifacts are tiny smoke tests that clarify failure modes. They should grow only
after they improve a real validation slice.

The answer-key-pruned diagnostic reached `0.890` value accuracy. That is the
most interesting negative/diagnostic evidence in the repo: if a model is handed
the right narrowed schema/value context, many failures disappear. The next clean
work is to derive useful context from training data, database contents, and
visible user text rather than from the evaluation answer.

## Reproduce

Use `uv run --active --no-sync ...` for repo commands.

Prepare a fixed CoSQL proxy eval slice:

```bash
uv run --active --no-sync python -m data.prepare \
  --config configs/qwen35_9b_5090.yaml \
  --section eval \
  --limit 100 \
  --output data/processed/eval_cosql_dev_100.jsonl \
  --manifest-output data/processed/eval_cosql_dev_100.manifest.json
```

Prepare a training slice:

```bash
uv run --active --no-sync python -m data.prepare \
  --config configs/qwen35_9b_5090.yaml \
  --section train \
  --limit 100 \
  --output data/processed/train_cosql_100.jsonl \
  --manifest-output data/processed/train_cosql_100.manifest.json
```

Validate training data without loading GPU libraries:

```bash
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_cosql_100.jsonl \
  --validate-data-only
```

Run a one-step GPU smoke on WSL:

```bash
CC=/home/dan/.local/bin/cc UV_CACHE_DIR=/tmp/uv-cache \
uv run --active --no-sync python -m train.finetune \
  --config configs/qwen35_9b_5090.yaml \
  --data data/processed/train_cosql_smoke.jsonl \
  --max-steps 1 \
  --output-dir outputs/gpu_access_smoke_1step \
  --report-to none
```

Evaluate an OpenAI-compatible endpoint:

```bash
uv run --active --no-sync python -m eval.run_eval \
  --benchmark prepared \
  --endpoint http://localhost:8000/v1 \
  --model-name multiturn-sql-100 \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --output results/multiturn_sql_100_cosql_dev.jsonl \
  --manifest-output results/multiturn_sql_100_cosql_dev.manifest.json \
  --database-root data/raw/cosql_dataset/database
```

Compare generated outputs with a hosted baseline only after both manifests are
row-matched and use the same scorer:

```bash
uv run --active --no-sync python -m eval.compare_hosted_baseline \
  --local-manifest results/multiturn_sql_100_cosql_dev.manifest.json \
  --hosted-manifest results/hosted/<run-id>.manifest.json \
  --output results/hosted/<run-id>.compared.manifest.json
```

## GPU Notes

This repo was run from WSL. GPU visibility can differ between the sandbox and
the real shell. The recorded fresh check in
`docs/training_runs/gpu_finetuning_evidence.json` saw an RTX 5090 outside the
sandbox, while sandboxed NVML access was blocked. Treat sandbox CUDA failures as
environment visibility failures until `/usr/lib/wsl/lib/nvidia-smi` and a
PyTorch CUDA probe are checked outside the sandbox.

## Where To Look

- `docs/research_roadmap.md`: current roadmap and checkpoint definitions.
- `docs/training_runs/gpu_finetuning_evidence.json`: recorded adapter metrics
  and GPU smoke evidence.
- `data/prepare.py`: dataset-to-chat JSONL preparation.
- `train/finetune.py`: Qwen LoRA fine-tuning entry point.
- `eval/run_eval.py`: endpoint evaluation and result manifest writing.
- `eval/local_benchmark.py`: local adapter evaluation.
- `eval/compare_hosted_baseline.py`: same-row local-vs-hosted comparison.
- `docs/data_artifacts/README.md`: small checked-in method artifacts.

## Current Research Direction

The next credible claim is not another prompt trick. It is a clean comparison:

1. Build split manifests with clear train, validation, proxy-dev, and holdout
   roles.
2. Train direct-SQL adapters on more than tiny samples.
3. Evaluate raw Qwen, best local LoRA, and a hosted SOTA model on the same
   multi-turn rows.
4. Use failure analysis to decide whether semantic context, value retrieval,
   Metric DSL, or recovery should be scaled.

Negative results should stay in the repo. If a method does not beat direct SQL,
that is still evidence about which idea is not yet worth scaling.
