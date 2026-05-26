# Repository AGENTS

This repository is organized around executable research claims, not generic SQL
utilities.

Read these first:

- `docs/research_goal.md`
- `docs/finetuning_ladder.md`
- `docs/methodology.md`

Global invariants:

- Keep same rows and same scorer whenever comparing one method against another.
- Keep oracle and non-oracle paths explicit. Do not smuggle oracle information
  through renamed fields, prompt prose, or generated artifacts.
- Prefer claim-ledger-backed manifests over informal summary prose.
- Treat direct SQL as the control arm unless a narrower benchmark says
  otherwise.
- If you change a method claim, also inspect the corresponding comparison,
  manifest, and ledger boundary.

Working areas:

- `data/`: prepared data contracts, semantic artifacts, metric DSL, value
  retrieval, and synthetic fixtures.
- `eval/`: manifests, scorers, comparisons, readiness gates, and claim
  enforcement.
- `train/`: `train.finetune`, run policy, and training-time guardrails.
- `notebooks/`: lab and reader-facing walkthroughs that explain repo evidence.
- `docs/`: structured explanation of research goals, methodology, finetuning
  ladder, and claim boundary.
