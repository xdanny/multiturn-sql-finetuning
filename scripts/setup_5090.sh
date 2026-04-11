#!/usr/bin/env bash
# One-shot install script for Qwen 3.5 9B fine-tuning + vLLM serving on RTX 5090 (Blackwell, sm_120)
# Assumes: Linux or WSL2, CUDA 12.8+ already installed, NVIDIA driver 575.64.03+
# Run scripts/verify_blackwell.py first to confirm prerequisites.

set -euo pipefail

echo "=== Blackwell / RTX 5090 toolchain setup ==="
echo

# Require Python 3.11+
python_version=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
echo "Python: ${python_version}"

# Create virtualenv with uv if available, else venv
if command -v uv &> /dev/null; then
    echo "Using uv for virtualenv"
    uv venv .venv --python 3.12
    # shellcheck source=/dev/null
    source .venv/bin/activate
    PIP="uv pip"
else
    echo "uv not found — using python venv + pip"
    python3 -m venv .venv
    # shellcheck source=/dev/null
    source .venv/bin/activate
    PIP="pip"
fi

echo
echo "=== Step 1: PyTorch nightly cu128 (required for sm_120) ==="
${PIP} install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

echo
echo "=== Step 2: Core training stack ==="
${PIP} install \
    "transformers>=5.0.0" \
    "trl>=0.12.0" \
    "peft>=0.14.0" \
    "accelerate>=1.2.0" \
    "datasets>=3.0.0" \
    "bitsandbytes>=0.45.3" \
    "sentencepiece" \
    "protobuf" \
    "wandb"

echo
echo "=== Step 3: Unsloth (must be 2026.4.0+ for Qwen 3.5 fused_ce_loss fix) ==="
${PIP} install "unsloth>=2026.4.0"

echo
echo "=== Step 4: Evaluation stack (RAGAS + datacompy + sqlglot) ==="
${PIP} install \
    "ragas>=0.4.3" \
    "datacompy>=0.13.0" \
    "sqlparse>=0.5.0" \
    "sqlglot>=25.0.0" \
    "pandas>=2.2.0" \
    "matplotlib>=3.9.0" \
    "seaborn>=0.13.0"

echo
echo "=== Step 5: API baselines ==="
${PIP} install \
    "anthropic>=0.40.0" \
    "openai>=1.50.0" \
    "google-genai>=0.3.0"

echo
echo "=== Step 6: vLLM from source (required for sm_120) ==="
echo "Note: this takes 15-30 minutes to build."
read -p "Build vLLM from source now? [y/N] " -r build_vllm
if [[ $build_vllm =~ ^[Yy]$ ]]; then
    VLLM_DIR="${VLLM_DIR:-$HOME/src/vllm}"
    mkdir -p "$(dirname "$VLLM_DIR")"
    if [ ! -d "$VLLM_DIR" ]; then
        git clone https://github.com/vllm-project/vllm.git "$VLLM_DIR"
    fi
    pushd "$VLLM_DIR"
    git pull
    python use_existing_torch.py
    ${PIP} install -r requirements/build.txt
    export VLLM_FLASH_ATTN_VERSION=2
    export TORCH_CUDA_ARCH_LIST="12.0"
    export MAX_JOBS="${MAX_JOBS:-6}"
    ${PIP} install --no-build-isolation -e .
    popd
else
    echo "Skipping vLLM build. Run later with:"
    echo "  cd \$HOME/src/vllm && VLLM_FLASH_ATTN_VERSION=2 TORCH_CUDA_ARCH_LIST=12.0 ${PIP} install --no-build-isolation -e ."
fi

echo
echo "=== Setup complete ==="
echo "Run: python scripts/verify_blackwell.py"
