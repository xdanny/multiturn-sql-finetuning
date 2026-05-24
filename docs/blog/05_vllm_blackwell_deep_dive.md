# Serving LoRA Adapters Locally with vLLM on Blackwell WSL

Serving was the point where model work became an engineering loop instead of a
notebook experiment. Once adapters can be exposed through an OpenAI-compatible
endpoint, the evaluator no longer needs to know whether it is talking to the
base model, a LoRA adapter, or a future production service.

The local serving environment is:

- WSL2 on an RTX 5090.
- vLLM `0.21.0`.
- `torch 2.11.0+cu130`.
- CUDA 13.0 runtime from wheels.
- No system `nvcc`.
- A user-local compiler at `/home/dan/.local/bin/cc`.

Training remains separate in `.venv`. Serving lives in `.venv-vllm`.

## Why vLLM Instead of Local Transformers for Every Benchmark

The repo still supports local Transformers evaluation, and that path is useful
for smoke tests. vLLM is the better comparison path once adapters exist:

- It is the same API shape many apps already use.
- It makes latency visible.
- It can serve named LoRA adapters.
- It exercises prompt formatting and chat-template behavior through a realistic
  endpoint.
- It keeps evaluation code independent of model-loading details.

This matters for multi-turn SQL because prompt length, chat templates, adapter
selection, and decoding settings all affect behavior. An endpoint benchmark is
closer to how a real analytics assistant would call the model.

## Working Single-Adapter Command

The most reliable current path is serving one adapter at a time:

```bash
CC=/home/dan/.local/bin/cc \
VLLM_USE_FLASHINFER_SAMPLER=0 \
PATH="$PWD/.venv-vllm/bin:$PATH" \
vllm serve unsloth/Qwen3.5-9B \
  --enable-lora \
  --lora-modules multiturn-sql-semantic-50=outputs/qwen35_9b_multiturn_sql_semantic_50steps/final \
  --max-lora-rank 64 \
  --max-loras 1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-num-seqs 64
```

Then the evaluator calls:

```bash
.venv/bin/python -m eval.run_eval \
  --benchmark prepared \
  --endpoint http://127.0.0.1:8000/v1 \
  --model-name multiturn-sql-semantic-50 \
  --input data/processed/eval_100_each.jsonl \
  --limit 100 \
  --max-tokens 192 \
  --database-root data/raw/cosql_dataset/database \
  --output results/vllm_qwen35_9b_lora_semantic50_cosql_dev_100turns.jsonl
```

## Failure Modes Worth Remembering

The useful failures were:

- No compiler: Triton JIT paths failed until `CC` pointed at the user-local
  compiler.
- FlashInfer sampler JIT: without system `nvcc`, disabling that sampler path was
  the practical choice.
- Default concurrency: vLLM's default sequence count was too high for this
  model/cache shape.
- Multi-LoRA serving: trying to keep several adapters mounted hit cache and CUDA
  graph memory pressure on this machine.
- Thinking mode: Qwen could emit reasoning prose unless
  `chat_template_kwargs.enable_thinking=false` was passed.

None of these are intellectually interesting model problems. They are still
worth writing down because they are exactly the kind of environment details that
can waste a day and make benchmark results look inconsistent.

## Single-LoRA Serving Is Enough For The Current Benchmark

Multi-adapter serving would be convenient, but it is not required for a clean
comparison. The benchmark input, output file, and evaluator stay the same. Only
the served adapter name changes.

That has one advantage: memory pressure is lower, and a failed comparison is
less likely to be a serving artifact. For this phase, correctness and
repeatability matter more than serving every adapter simultaneously.

## Latency Interpretation

The semantic adapter had much higher mean latency in the endpoint summary:

| Model | Mean latency |
| --- | ---: |
| Base `unsloth/Qwen3.5-9B` | 292.79 ms |
| `multiturn-sql-50` | 360.07 ms |
| `multiturn-sql-100` | 392.88 ms |
| `multiturn-sql-semantic-50` | 2285.79 ms |

The semantic run used longer prompts because semantic model context was injected
into CoSQL examples. Longer context is not free. This is one reason a production
semantic layer should not simply dump every cube, dimension, measure, and join
into every prompt. It needs retrieval and pruning:

- Include cubes likely relevant to the current turn.
- Include prior-turn cubes and measures.
- Include join paths only when they connect relevant entities.
- Include metric definitions when a metric phrase is present.
- Exclude unrelated schema sections aggressively.

The benchmark now exposes that tradeoff. A prompt feature that improves syntax
but does not improve execution accuracy may still be too expensive in its
current form.

## The Serving Contract

The serving layer should be treated as a contract:

- Model names identify base or adapter behavior.
- Prompts are SQL-only and thinking mode is disabled.
- Results are written to versioned JSONL files.
- Plotting includes only selected result files.
- The benchmark slice stays fixed unless a new experiment explicitly changes it.

That contract is what lets the project compare model changes rather than
accidentally comparing environments.

Sources:

- vLLM GPU installation docs: https://docs.vllm.ai/getting_started/installation/gpu.html
- vLLM LoRA docs: https://docs.vllm.ai/features/lora.html
- Qwen model card: https://huggingface.co/Qwen/Qwen3.5-9B
