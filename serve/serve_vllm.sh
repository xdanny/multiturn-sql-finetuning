#!/usr/bin/env bash
# Serve fine-tuned Qwen 3.5 9B with vLLM on RTX 5090
# Prerequisites: vLLM built from source with TORCH_CUDA_ARCH_LIST="12.0" (see scripts/setup_5090.sh)

set -euo pipefail

MODEL="${MODEL:-unsloth/Qwen3.5-9B}"
LORA_PATH="${LORA_PATH:-outputs/qwen35_9b_multiturn_sql/final}"
LORA_NAME="${LORA_NAME:-multiturn-sql}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# FA3 not supported on Blackwell — force FA2
export VLLM_FLASH_ATTN_VERSION=2

vllm serve "$MODEL" \
    --enable-lora \
    --lora-modules "${LORA_NAME}=${LORA_PATH}" \
    --max-lora-rank 64 \
    --max-loras 1 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size 1 \
    --dtype bfloat16
