#!/usr/bin/env bash
# Serve fine-tuned Qwen 3.5 9B with vLLM on RTX 5090
# Prerequisites: .venv-vllm with vLLM installed, or vLLM on PATH.

set -euo pipefail

MODEL="${MODEL:-unsloth/Qwen3.5-9B}"
LORA_PATH="${LORA_PATH:-outputs/qwen35_9b_multiturn_sql_50steps/final}"
LORA_NAME="${LORA_NAME:-multiturn-sql-50}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

# WSL2/Blackwell notes:
# - Triton needs a C compiler for JIT kernels.
# - FlashInfer sampler JIT currently expects nvcc; disabling it keeps serving
#   working without a system CUDA Toolkit install.
if [[ -z "${CC:-}" && -x "$HOME/.local/bin/cc" ]]; then
    export CC="$HOME/.local/bin/cc"
fi
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

vllm serve "$MODEL" \
    --enable-lora \
    --lora-modules "${LORA_NAME}=${LORA_PATH}" \
    --max-lora-rank 64 \
    --max-loras 1 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size 1 \
    --dtype bfloat16 \
    --max-num-seqs "$MAX_NUM_SEQS"
