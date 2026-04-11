#!/usr/bin/env python3
"""
Blackwell / RTX 5090 toolchain compatibility probe.

Checks everything needed for Qwen 3.5 9B bf16 LoRA training with Unsloth
and vLLM serving on sm_120 (Blackwell).

Run this BEFORE installing anything — it tells you what's missing.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass


# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    fix: str | None = None


def check_nvidia_driver() -> CheckResult:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True,
        ).strip()
        name, driver = [x.strip() for x in out.split(",")]
        major = int(driver.split(".")[0])
        if major >= 575:
            return CheckResult(
                "NVIDIA driver",
                True,
                f"{name} driver {driver}",
            )
        return CheckResult(
            "NVIDIA driver",
            False,
            f"{name} driver {driver} — need 575.64.03+",
            "Update via NVIDIA website or apt/dnf",
        )
    except FileNotFoundError:
        return CheckResult(
            "NVIDIA driver",
            False,
            "nvidia-smi not found",
            "Install NVIDIA driver 575.64.03+",
        )
    except Exception as e:
        return CheckResult("NVIDIA driver", False, f"error: {e}")


def check_cuda_toolkit() -> CheckResult:
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        # e.g. "Cuda compilation tools, release 12.9, V12.9.86"
        for line in out.splitlines():
            if "release" in line:
                version_str = line.split("release")[1].split(",")[0].strip()
                major, minor = map(int, version_str.split(".")[:2])
                if (major, minor) >= (12, 8):
                    return CheckResult(
                        "CUDA toolkit",
                        True,
                        f"nvcc {version_str}",
                    )
                return CheckResult(
                    "CUDA toolkit",
                    False,
                    f"nvcc {version_str} — need 12.8+",
                    "Install CUDA 12.8+ from developer.nvidia.com/cuda-toolkit",
                )
        return CheckResult("CUDA toolkit", False, "could not parse nvcc output")
    except FileNotFoundError:
        return CheckResult(
            "CUDA toolkit",
            False,
            "nvcc not found in PATH",
            "Install CUDA 12.8+ and add /usr/local/cuda/bin to PATH",
        )


def check_python() -> CheckResult:
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 11):
        return CheckResult("Python", True, f"{major}.{minor}")
    return CheckResult(
        "Python",
        False,
        f"{major}.{minor} — need 3.11+",
        "Use pyenv or uv to install Python 3.11+",
    )


def check_pytorch() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult(
            "PyTorch",
            False,
            "not installed",
            "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
        )

    version = torch.__version__
    cuda_version = getattr(torch.version, "cuda", None)

    if not torch.cuda.is_available():
        return CheckResult(
            "PyTorch",
            False,
            f"{version} — CUDA not available",
            "Reinstall with cu128 wheel",
        )

    if cuda_version is None or not cuda_version.startswith("12.8"):
        return CheckResult(
            "PyTorch",
            False,
            f"{version} (CUDA {cuda_version}) — need cu128",
            "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128",
        )

    return CheckResult("PyTorch", True, f"{version} (CUDA {cuda_version})")


def check_gpu_compute_capability() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("GPU compute capability", False, "PyTorch not installed")

    if not torch.cuda.is_available():
        return CheckResult("GPU compute capability", False, "no CUDA GPU")

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    cap_str = f"{cap[0]}.{cap[1]} (sm_{cap[0]}{cap[1]})"

    if cap == (12, 0):
        return CheckResult(
            "GPU compute capability",
            True,
            f"{name} — {cap_str} (Blackwell)",
        )
    if cap == (10, 0):
        return CheckResult(
            "GPU compute capability",
            True,
            f"{name} — {cap_str} (Hopper H100)",
        )
    if cap >= (8, 0):
        return CheckResult(
            "GPU compute capability",
            True,
            f"{name} — {cap_str} (Ampere/Ada — this project targets Blackwell but fallback OK)",
        )
    return CheckResult(
        "GPU compute capability",
        False,
        f"{name} — {cap_str} (too old)",
        "Need sm_80+ GPU",
    )


def check_vram() -> CheckResult:
    try:
        import torch
    except ImportError:
        return CheckResult("GPU VRAM", False, "PyTorch not installed")

    if not torch.cuda.is_available():
        return CheckResult("GPU VRAM", False, "no CUDA GPU")

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    total_gb = total_bytes / (1024**3)

    if total_gb >= 30:
        return CheckResult("GPU VRAM", True, f"{total_gb:.1f} GB (fits Qwen 3.5 9B bf16 LoRA)")
    if total_gb >= 20:
        return CheckResult(
            "GPU VRAM",
            True,
            f"{total_gb:.1f} GB (fits Qwen 3.5 4B; tight for 9B)",
        )
    return CheckResult(
        "GPU VRAM",
        False,
        f"{total_gb:.1f} GB (too small for Qwen 3.5 9B)",
        "Use Qwen 3.5 0.8B/2B, or run on Colab L4",
    )


def check_unsloth() -> CheckResult:
    try:
        import unsloth
    except ImportError:
        return CheckResult(
            "Unsloth",
            False,
            "not installed",
            "uv pip install unsloth --torch-backend=auto",
        )

    version = getattr(unsloth, "__version__", "unknown")
    # Parse e.g. "2026.4.4"
    try:
        parts = version.split(".")
        year = int(parts[0])
        month = int(parts[1])
        if year > 2026 or (year == 2026 and month >= 4):
            return CheckResult(
                "Unsloth",
                True,
                f"{version} (Qwen 3.5 fused_ce_loss fix present)",
            )
        return CheckResult(
            "Unsloth",
            False,
            f"{version} — need 2026.4.0+ for Qwen 3.5 long-context fix",
            "pip install -U unsloth",
        )
    except (ValueError, IndexError):
        return CheckResult(
            "Unsloth",
            False,
            f"{version} — unknown version format",
            "pip install -U unsloth",
        )


def check_transformers() -> CheckResult:
    try:
        import transformers
    except ImportError:
        return CheckResult("transformers", False, "not installed", "pip install transformers")

    version = transformers.__version__
    major = int(version.split(".")[0])
    if major >= 5:
        return CheckResult("transformers", True, f"{version}")
    return CheckResult(
        "transformers",
        False,
        f"{version} — need v5.0+ for Qwen 3.5",
        "pip install -U transformers",
    )


def check_trl() -> CheckResult:
    try:
        import trl
    except ImportError:
        return CheckResult("TRL", False, "not installed", "pip install trl")
    return CheckResult("TRL", True, trl.__version__)


def check_bitsandbytes() -> CheckResult:
    try:
        import bitsandbytes as bnb
    except ImportError:
        return CheckResult(
            "bitsandbytes",
            False,
            "not installed (needed for adamw_8bit optimizer)",
            "pip install bitsandbytes",
        )
    return CheckResult("bitsandbytes", True, bnb.__version__)


def check_vllm() -> CheckResult:
    try:
        import vllm
    except ImportError:
        return CheckResult(
            "vLLM",
            False,
            "not installed (needed only for serving phase)",
            "Build from source: see scripts/setup_5090.sh",
        )
    return CheckResult("vLLM", True, vllm.__version__)


def check_ragas() -> CheckResult:
    try:
        import ragas
    except ImportError:
        return CheckResult(
            "RAGAS",
            False,
            "not installed",
            "pip install ragas",
        )
    return CheckResult("RAGAS", True, ragas.__version__)


def main() -> int:
    print(f"\n{BOLD}Blackwell / RTX 5090 toolchain probe{RESET}\n")

    checks = [
        check_nvidia_driver(),
        check_cuda_toolkit(),
        check_python(),
        check_pytorch(),
        check_gpu_compute_capability(),
        check_vram(),
        check_unsloth(),
        check_transformers(),
        check_trl(),
        check_bitsandbytes(),
        check_vllm(),
        check_ragas(),
    ]

    all_passed = True
    for r in checks:
        mark = f"{GREEN}✓{RESET}" if r.passed else f"{RED}✗{RESET}"
        print(f"  {mark} {BOLD}{r.name:26s}{RESET} {r.detail}")
        if not r.passed:
            all_passed = False
            if r.fix:
                print(f"      {YELLOW}→ {r.fix}{RESET}")

    print()
    if all_passed:
        print(f"{GREEN}{BOLD}All checks passed — ready to train.{RESET}\n")
        return 0
    print(
        f"{YELLOW}{BOLD}Some checks failed — run `bash scripts/setup_5090.sh` to fix.{RESET}\n"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
