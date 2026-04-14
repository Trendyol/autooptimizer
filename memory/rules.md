# Experimentation Rules

Each experiment starts a vLLM server, runs a fixed benchmark, then kills the server.

## What You CAN Do

- Modify `project/edit/serve_config.sh` — this is the only file you edit
- Use **ANY** vLLM serve flag — no restrictions on parameters
- Try quantization methods (fp8, awq, gptq, bitsandbytes, marlin, etc.)
- Try engine parameters (gpu-memory-utilization, max-num-seqs, max-num-batched-tokens, block-size, swap-space, etc.)
- Try scheduler settings (enable-chunked-prefill, scheduler-delay-factor, num-scheduler-steps, etc.)
- Try speculative decoding (speculative-model, num-speculative-tokens, etc.)
- Try compilation flags, parallelism settings, kv-cache-dtype, anything
- Discover new flags by running `vllm serve --help`

## What You CANNOT Do

- **Change the model** — always use the model specified in `memory/overview.md`
- **Modify `project/no_edit/benchmark.py`** — it is read-only (contains the fixed benchmark and scoring function)
- **Install new packages** or add dependencies (only use what's in `pyproject.toml`)
- **Change the port** — always use `--port 8000` (benchmark.py expects this)
- **Change the host** — always use `--host 127.0.0.1`

## Scoring

```
score = throughput_tok_per_sec / (1 + mean_e2el_ms / 1000)
```

Higher is better. This is the ONLY metric that matters for keep/discard decisions.

## GPU Memory

Soft constraint. The server must not OOM. Some increase in memory usage is acceptable for meaningful score gains.
