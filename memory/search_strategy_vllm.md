# Search Strategy (vLLM)

## Official Documentation References

Use these links to look up details on any parameter, feature, or optimization technique:

- **Engine Arguments (full flag reference):** https://docs.vllm.ai/en/stable/configuration/engine_args/
- **Optimization & Tuning Guide:** https://docs.vllm.ai/en/stable/configuration/optimization/
- **CLI Reference (`vllm serve`):** https://docs.vllm.ai/en/stable/cli/serve.html
- **Quantization:** https://docs.vllm.ai/en/stable/quantization/
- **Speculative Decoding:** https://docs.vllm.ai/en/stable/features/speculative_decoding/
- **Automatic Prefix Caching:** https://docs.vllm.ai/en/stable/features/automatic_prefix_caching.html
- **Benchmarking:** https://docs.vllm.ai/en/stable/design/benchmarking/
- **GitHub Repo (source + latest docs):** https://github.com/vllm-project/vllm

## Hypothesis Prioritization

Order experiments by expected impact. Use this priority formula:

```
priority = uplift_expectation - 0.5 * effort
```

Where:
- `uplift_expectation`: estimated score improvement (high/medium/low → 3/2/1)
- `effort`: complexity of the change (high/medium/low → 3/2/1)

## High-Impact Areas (try first)

### 1. Quantization (highest impact)
- `--quantization fp8` — often 1.5-2x throughput boost with minimal quality loss
- `--quantization awq` — requires AWQ-quantized model variant
- `--quantization gptq` — requires GPTQ-quantized model variant
- `--kv-cache-dtype fp8_e5m2` — compress KV cache, more room for batching

### 2. Batching parameters
- `--max-num-seqs` — increase concurrent sequences (try 128, 256, 512, 1024)
- `--max-num-batched-tokens` — increase tokens per iteration (try 4096, 8192, 16384, 32768)
- These directly control throughput

### 3. Memory utilization
- `--gpu-memory-utilization` — try 0.85, 0.90, 0.95 (more memory = more KV cache = more concurrent requests)
- `--swap-space` — CPU offloading (try 0, 4, 8, 16 GiB)

### 4. Scheduling
- `--enable-chunked-prefill` — better interleaving of prefill and decode
- `--enable-prefix-caching` — reuse KV cache for shared prefixes
- `--num-scheduler-steps` — multi-step scheduling (try 1, 5, 10)
- `--scheduler-delay-factor` — accumulate requests for better batching

## Medium-Impact Areas (try second)

### 5. CUDA graphs
- `--enforce-eager` — disable CUDA graphs (sometimes faster for small models)
- Default (CUDA graphs enabled) — usually better for larger models

### 6. Block size
- `--block-size` — KV cache block size (try 8, 16, 32)
- Affects memory fragmentation and allocation efficiency

### 7. Speculative decoding
- `--speculative-model <draft_model>` — use a smaller model to speculate
- `--num-speculative-tokens 5` — how many tokens to speculate ahead
- Can significantly improve latency but adds complexity

## Low-Impact Areas (try last)

### 8. Data types
- `--dtype half` vs `--dtype bfloat16` vs `--dtype auto`
- Usually `auto` is fine, but worth testing

### 9. Parallelism
- `--tensor-parallel-size` — only useful with multiple GPUs
- `--pipeline-parallel-size` — only useful with multiple GPUs

### 10. Advanced / experimental
- `--max-model-len` — restrict sequence length to save memory
- Compilation flags via `--compilation-config`
- Any new flags discovered via `vllm serve --help`

## Strategy Tips

1. **Start with baseline** — always run the default config first
2. **One change at a time** — isolate the effect of each parameter
3. **Combine winners** — after finding individual improvements, combine them
4. **Binary search on continuous params** — for things like gpu-memory-utilization, narrow down the optimal value
5. **Read error messages** — crashes often reveal the boundary of what's possible
6. **Check `vllm serve --help` periodically** — you might discover flags you missed
