# Docs AGENTS

This subtree owns structured explanation of the repo's research program.

Primary responsibilities:

- Keep `research_goal`, `methodology`, and the finetuning ladder aligned.
- Document claim boundary, benchmark meaning, and experiment sequencing in a
  way that matches the code and manifests.
- Prefer disciplined repo language over generic blog-style prose.

Rules:

- If a method changes, update the ladder and inspect whether methodology and
  research_goal also need the same change.
- Keep claim boundary explicit. Supported, pending, and diagnostic states should
  read the same way they are enforced.
- Benchmark names should stay precise: CoSQL proxy, SParC signal,
  BIRD-Interact target, and synthetic fixtures each have different roles.
- Use `uv run ...` in command examples unless a command is intentionally not
  Python or not managed by uv.

When editing here, inspect:

- `docs/research_goal.md`
- `docs/methodology.md`
- `docs/finetuning_ladder.md`
- `docs/metric_dsl_contract.md`
- `docs/rollout_eval_contract.md`
