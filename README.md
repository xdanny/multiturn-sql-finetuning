# Multi-Turn SQL Fine-Tuning

Fine-tune **Qwen 3.5 9B** on multi-turn conversational SQL with clarification handling. Train locally on RTX 5090 with Unsloth, serve with vLLM, evaluate with RAGAS, benchmark against frontier API models.

## The Problem

Single-shot text-to-SQL is largely solved — frontier models hit 80%+ on Spider. But real production SQL assistants handle **multi-turn conversations**: vague questions → clarification → refinement → follow-ups. Most portfolio projects ignore this.

This repo asks: **can a fine-tuned Qwen 3.5 9B match frontier API models on multi-turn SQL at a fraction of the cost?**

## Stack

| Layer | Tool | Notes |
|-------|------|-------|
| Base model | `Qwen/Qwen3.5-9B` | Released Feb 2026, Apache 2.0 |
| Training | Unsloth + TRL SFTTrainer | bf16 LoRA (QLoRA unstable on Qwen 3.5) |
| Serving | vLLM 0.19+ with `--enable-lora` | FP8 native on Blackwell |
| Evaluation | RAGAS 0.4.3 | `DataCompyScore`, `ToolCallAccuracy`, `AgentGoalAccuracy` |
| Tracking | Weights & Biases | Loss curves, hyperparams, eval metrics |
| Compute | Local RTX 5090 (32GB) | Colab Pro L4 as backup for debug |

## Datasets

| Dataset | Role | HF ID |
|---------|------|-------|
| CoSQL | Multi-turn SQL with clarification | `alpineai/cosql` |
| SParC | Multi-turn Spider extension | `jellyChiru/SParC` |
| gretelai/synthetic_text_to_sql | Single-turn reinforcement (20-30% mix) | `gretelai/synthetic_text_to_sql` |
| BIRD mini_dev | Single-turn execution accuracy baseline | `birdsql/bird_mini_dev` |
| BIRD-Interact Lite | Stretch goal (ICLR 2026 Oral) | `birdsql/bird-interact-lite` |

## Evaluation

**Single-turn metrics:**
- `DataCompyScore` — execution-based result DataFrame comparison (primary signal)
- `SQLSemanticEquivalence` — semantic equivalence tiebreaker
- Latency (TTFT, TPOT via vLLM `benchmark_serving.py`)
- $ per successful query

**Multi-turn metrics:**
- `ToolCallAccuracy` — sequence-level correctness
- `AgentGoalAccuracyWithReference` — dialog-level success
- `TopicAdherence` — drift detection
- Interaction Match (IM) — all turns in dialog correct

**Baselines:** Claude Haiku 4.5, GPT-4o-mini, Gemini Flash, Qwen 3.5 9B base (ablation)

## Quick Start

```bash
# 1. Verify Blackwell toolchain
python scripts/verify_blackwell.py

# 2. Install dependencies (one-shot)
bash scripts/setup_5090.sh

# 3. Prepare data
python -m data.prepare

# 4. Train (local 5090)
python -m train.finetune --config configs/qwen35_9b_5090.yaml

# 5. Serve with vLLM
bash serve/serve_vllm.sh

# 6. Evaluate
python -m eval.run_eval --benchmark cosql
python -m eval.baselines --model gpt-4o-mini
python -m eval.plot_pareto
```

## Hardware Target

**Primary:** RTX 5090 (Blackwell, sm_120, 32GB GDDR7)
- CUDA 12.8+, driver 575.64.03+, Linux/WSL2
- PyTorch nightly cu128 (stock wheels lack sm_120 kernels)
- vLLM built from source for sm_120 support
- Unsloth ≥ 2026.4.0 (required for Qwen 3.5 `fused_ce_loss` fix)

**Backup:** Google Colab Pro L4 (22GB) with Qwen 3.5 4B for rapid debug cycles.

## VRAM Budget (32GB 5090)

| Config | VRAM | Throughput | Status |
|--------|------|------------|--------|
| 2048 ctx, bs=1, bf16 LoRA | ~23GB | ~100-200 tok/s | ✅ Primary target |
| 4096 ctx, bs=1, bf16 LoRA | ~26GB | ~50 tok/s | ⚠️ Tight, works with GC |
| 8192 ctx, bs=1, bf16 LoRA | ~22GB (with fused_ce_loss fix) | ~25-30 tok/s | ✅ Requires Unsloth 2026.4+ |

## Status

🚧 **In development** — targeting public launch end of Week 2.

## License

MIT
