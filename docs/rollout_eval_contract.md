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

## Claim Boundary

A generated-history rollout result can support only a proxy rollout claim until
there is a side-by-side comparison against the same model and input under
teacher-forced history. The claim ledger tracks these separately:

- `model_generated_history_rollout`: pending until a valid rollout manifest
  exists.
- `rollout_beats_teacher_forced_history`: pending until comparison metrics show
  a same-model rollout result beating the matching teacher-forced result.

Oracle inputs are rejected by default. `--allow-oracle-plan` is diagnostic only
and sets `oracle_allowed=true` in the manifest, so those rows cannot become a
production behavior/recovery claim.
