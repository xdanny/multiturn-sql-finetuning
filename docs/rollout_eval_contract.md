# Generated-History Rollout Evaluation

The standard prepared evaluator is teacher-forced: turn `N` sees prior gold SQL
assistant messages from the dataset. That is useful for clean-history SQL
generation, but it does not test recovery after the model's own earlier output.

`eval.rollout_eval` adds the missing evaluation mode. It runs a prepared dialog
sequentially:

1. Send the current prompt to the model.
2. Keep the gold SQL only as `reference_sql` for scoring.
3. Insert the model's generated SQL as the assistant message in the in-memory
   dialog history.
4. Use that generated assistant history for the next turn.

The output rows use:

- `history_policy=model_generated_sql_rollout`
- `original_history_policy` for the source artifact's history policy
- the exact `messages` sent for each turn
- `reference_sql` for scoring only
- `generated_sql` for the model output propagated to later turns

This means a wrong or invalid first turn is still carried into turn two. That is
the point: rollout evaluation measures error propagation and recovery instead of
clean-history conditioning.

## Command

```bash
python -m eval.rollout_eval \
  --model-name <served-model-name> \
  --endpoint http://127.0.0.1:8000/v1 \
  --input data/processed/eval_cosql_dev_100.jsonl \
  --database-root data/raw/cosql_dataset/database \
  --output results/rollout/<run-id>.jsonl
```

The runner writes a result manifest next to the output unless
`--manifest-output` is supplied.

## Compare Against Teacher-Forced History

Rollout accuracy by itself does not prove behavior or recovery improved. The
comparison must use the same model and exact same prepared input hash under the
teacher-forced evaluator:

```bash
python -m eval.compare_rollout_history \
  --rollout-manifest results/rollout/<run-id>.manifest.json \
  --teacher-forced-manifest results/teacher_forced/<run-id>.manifest.json \
  --output results/rollout/<run-id>.compared.manifest.json
```

The comparison command refuses mismatched models, mismatched input hashes,
mismatched row counts, mismatched output row identities, oracle diagnostics,
non-rollout manifests, and teacher-forced artifacts that do not use
`history_policy=gold_sql_teacher_forced`. The output is an augmented rollout
manifest with:

- `teacher_forced_comparison_run_id`
- `teacher_forced_model_name`
- `teacher_forced_input_sha256`
- `teacher_forced_value_execution_accuracy`
- `teacher_forced_strict_execution_accuracy`
- `teacher_forced_comparable_row_count`
- `rollout_value_delta_vs_teacher_forced`
- `rollout_strict_delta_vs_teacher_forced`

## Local Same-Checkpoint Runner

For local adapter evaluation, the repo also has a wrapper that runs both the
teacher-forced prepared benchmark and the generated-history rollout benchmark
for the same checkpoint and input manifest:

```bash
uv run python -m eval.run_local_rollout_comparison \
  --training-manifest outputs/<run>/training.manifest.json \
  --output-dir results/rollout \
  --run-id <run-id> \
  --model-name <local-model-name> \
  --adapter-path <adapter-dir> \
  --database-root data/raw/cosql_dataset/database
```

The training manifest must point at `benchmark=prepared` and
`evaluation_mode=non_oracle_generation`. The runner then writes:

- `<run-id>.teacher_forced.manifest.json`
- `<run-id>.rollout.manifest.json`
- `<run-id>.compared.manifest.json`

This is the local Stage 5 prepared-dialog contract: same checkpoint, same
prepared input, different history policy.

For endpoint-free validation, the same runner can also write a readiness
artifact without loading a model:

```bash
uv run python -m eval.run_local_rollout_comparison \
  --training-manifest docs/data_artifacts/behavior_recovery_proxy_training_run.manifest.json \
  --output-dir /tmp/behavior-recovery-rollout-unused \
  --run-id behavior-recovery-proxy \
  --model-name unsloth/Qwen3.5-9B \
  --preflight-output docs/behavior_recovery_rollout_preflight.json \
  --preflight-only
```

That preflight is bounded to `preflight only; no rollout execution claim`. It
proves the current prepared behavior/recovery manifest still points at a valid
non-oracle prepared input before a fresh rollout comparison is run.

## Claim Boundary

A generated-history rollout result can support only a proxy rollout claim until
there is a side-by-side comparison against the same model and input under
teacher-forced history. The claim ledger tracks these separately:

- `model_generated_history_rollout`: pending until a valid rollout manifest
  exists.
- `rollout_beats_teacher_forced_history`: pending until comparison metrics show
  a same-model, same-input, same-row rollout result beating the referenced
  teacher-forced manifest with a positive value-accuracy delta.

Oracle inputs are rejected by default. `--allow-oracle-plan` is diagnostic only
and sets `oracle_allowed=true` in the manifest, so those rows cannot become a
production behavior/recovery claim.
