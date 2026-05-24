# WSL2, RTX 5090, and the Boring Parts That Made the Loop Work

The useful outcome of the environment work is not that a particular command ran
once. It is that the repo now has a repeatable split between training, serving,
and evaluation on a local RTX 5090 under WSL2.

That split is deliberate:

- `.venv` is the training environment. It runs Unsloth, TRL, and
  `torch 2.10.0+cu128`.
- `.venv-vllm` is the serving environment. It runs vLLM `0.21.0`,
  `torch 2.11.0+cu130`, and the OpenAI-compatible endpoint.
- The evaluator calls either local Transformers or the vLLM endpoint, but it
  does not care which virtualenv launched the server.

Trying to force training and serving into one Python environment cost more time
than it saved. The two stacks move at different speeds and have different CUDA
pressure. Splitting them keeps the engineering surface smaller: train adapters
in one place, serve adapters in another, evaluate through a stable API.

## Verified Local Constraints

The machine sees an RTX 5090 with about 31.8 GB of VRAM. PyTorch CUDA works for
training, and the training verifier confirms Blackwell compute capability
through the installed wheel stack.

The system does not have `nvcc` installed. That matters for source builds and
some JIT paths, but it does not block the wheel-based vLLM serving path used for
these runs. The working workaround is:

```bash
CC=/home/dan/.local/bin/cc
VLLM_USE_FLASHINFER_SAMPLER=0
```

`CC` points Triton at a user-local compiler. Disabling the FlashInfer sampler
avoids a path that otherwise wants the CUDA toolkit compiler. This is not a
philosophical choice; it is just the shortest stable path for the current WSL2
machine.

## The Serving Command That Worked

The endpoint experiments use vLLM with LoRA enabled:

```bash
CC=/home/dan/.local/bin/cc \
VLLM_USE_FLASHINFER_SAMPLER=0 \
PATH="$PWD/.venv-vllm/bin:$PATH" \
vllm serve unsloth/Qwen3.5-9B \
  --enable-lora \
  --lora-modules multiturn-sql-100=outputs/qwen35_9b_multiturn_sql_100steps/final \
  --max-lora-rank 64 \
  --max-loras 1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82 \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-num-seqs 64
```

The important flags are not decorative:

- `--max-model-len 4096` keeps multi-turn prompts realistic without blowing up
  cache memory.
- `--max-num-seqs 64` avoids the default concurrency target, which exceeded the
  available cache budget in this environment.
- `--enable-lora` and `--lora-modules` let the same base model serve adapters by
  model name.
- `--dtype bfloat16` is the stable precision for this stack.

When I tried to mount several adapters at once, the current machine hit cache
and CUDA graph memory limits. Serving a single LoRA at a time is less elegant,
but it is reliable and adequate for benchmark iteration.

## Qwen-Specific Evaluation Detail

Endpoint evaluation needs Qwen thinking mode disabled:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

Without that, the model can produce reasoning text before the SQL. The system
prompt asks for SQL only, but for this model family the chat template setting is
the stronger control. SQL evaluation should not depend on scraping a query out
of prose.

## Credential and Repo Access Notes

The repo also records a practical connector caveat. A GitHub app or connector
can inspect repository metadata and files when installed and authorized, but it
does not replace the local WSL `git`, SSH agent, or 1Password CLI path. Private
repo cloning and pushing still depend on the local credential setup.

For this workspace, the reliable path is local Git plus the user's configured
SSH or HTTPS credentials. The connector remains useful for metadata and review
work, but it is not a substitute for local repository access.

## What This Setup Buys

The main payoff is fast iteration discipline:

1. Generate a bounded training file.
2. Run a bounded LoRA training job.
3. Serve the resulting adapter by name.
4. Run the same endpoint benchmark.
5. Plot the same summary files.

That discipline matters more than any one command. The project is now able to
separate environment failures from model-quality failures. When the semantic
50-step adapter tied the plain 50-step adapter, that was a data/modeling signal,
not a serving-stack mystery.

Sources:

- vLLM GPU installation docs: https://docs.vllm.ai/getting_started/installation/gpu.html
- vLLM LoRA serving docs: https://docs.vllm.ai/features/lora.html
- Qwen 3.5 9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
