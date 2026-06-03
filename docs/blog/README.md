# Blog-Attached Evidence

The public post links to this repository as the attached codebase for the
multi-turn SQL fine-tuning work. Keep the reader path focused on the research
question, the measured comparisons, and the evidence files that support the
claims.

The post should cite `README.md`, `docs/research_roadmap.md`, run summaries
under `docs/training_runs/`, and small canonical data artifacts under
`docs/data_artifacts/`. It should not turn local setup, vLLM serving, generated
evidence bundles, or environment notes into the public reader path.

The intended pattern is:

1. The post frames a claim.
2. The matching lab section runs the smallest executable version of the claim or
   diagnostic.
3. Run manifests or canonical data artifacts support larger endpoint and method
   results.
4. The post states what the artifact proves and what it does not prove.

Inconclusive results should be cited with the same discipline as wins and
failures. Before the post uses an inconclusive result, confirm that the matching
artifact records the row scope, control or readiness policy, scorer, blocker,
and remaining uncertainty. The post should explain the consideration it creates
for future work instead of recasting it as either success or failure.
