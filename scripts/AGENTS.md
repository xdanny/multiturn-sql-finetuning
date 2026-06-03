# Scripts AGENTS

This subtree owns environment checks, setup helpers, and operational scripts
that support finetuning or serving.

Primary responsibilities:

- Keep environment checks actionable for WSL, RTX 5090, Unsloth, Triton, and
  vLLM workflows.
- Separate setup diagnostics from benchmark evidence.
- Prefer scripts that verify prerequisites or run a bounded smoke path over
  scripts that silently mutate training data or result artifacts.
- Document any machine-specific workaround next to the script and, when it
  affects research evidence, in the relevant checked-in run summary.

Rules:

- Use `uv run ...` for Python script examples unless the script is explicitly a
  shell-only setup helper.
- Do not read, print, or copy private SSH keys, API keys, or 1Password secrets.
- Do not make GPU activity the default path for a script that is also used in
  docs or CI-style verification.
- If a script writes outputs, keep run-specific files under `outputs/` or
  `results/`; checked-in canonical inputs belong under `docs/data_artifacts/`
  only when the data artifact guide allows it.
- Keep compiler or cache workarounds explicit, especially `CC`, `ZIG_*_CACHE`,
  and WSL-specific CUDA notes.

When editing here, inspect:

- `scripts/verify_blackwell.py`
- `scripts/setup_5090.sh`
- `docs/research_roadmap.md`
- `README.md`
- `train/AGENTS.md`
