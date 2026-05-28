# Serve AGENTS

This subtree owns local model serving helpers for OpenAI-compatible endpoint
evaluation.

Primary responsibilities:

- Keep serving scripts aligned with the evaluation commands that consume the
  endpoint.
- Make adapter names, ports, max sequence length, LoRA settings, and GPU memory
  choices explicit.
- Treat serving as an operational prerequisite, not as evidence that a method
  improved.
- Keep vLLM-specific WSL and RTX 5090 workarounds visible in scripts and docs.

Rules:

- Do not change serving defaults in a way that changes benchmark comparability
  without updating result-manifest docs and runbooks.
- Do not mix several LoRA adapters in one serving example unless that exact
  setup has been verified on this machine.
- Endpoint evaluation should still write manifests through `eval.run_eval` or a
  paired comparison runner; serving logs are not benchmark artifacts.
- Keep secrets out of commands. Use local endpoint defaults such as
  `api_key=EMPTY` when the server expects a placeholder.

When editing here, inspect:

- `serve/serve_vllm.sh`
- `README.md`
- `docs/5090_benchmark_report.md`
- `docs/finetuning_smoke_matrix.md`
- `eval/run_eval.py`
