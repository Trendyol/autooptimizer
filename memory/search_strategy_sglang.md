# Search Strategy (SGLang)

## Official Documentation References

Use these links to look up details on any parameter, feature, or optimization technique:

- **Server Arguments (full flag reference):** https://sgl-project.github.io/advanced_features/server_arguments.html
- **Hyperparameter Tuning Guide:** https://sgl-project.github.io/advanced_features/hyperparameter_tuning.html
- **Benchmarking Guide:** https://sgl-project.github.io/developer_guide/bench_serving.html
- **Quantization:** https://sgl-project.github.io/advanced_features/quantization.html
- **Speculative Decoding:** https://sgl-project.github.io/advanced_features/speculative_decoding.html
- **Data Parallelism / Model Gateway:** https://sgl-project.github.io/advanced_features/sgl_model_gateway.html
- **GitHub Repo (source + latest docs):** https://github.com/sgl-project/sglang

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

### 2. Memory management
- `--mem-fraction-static` — fraction of GPU memory for KV cache (try 0.80, 0.85, 0.90, 0.95)
- `--chunked-prefill-size` — max tokens per prefill chunk (try 4096, 8192, 16384, 32768)
- Higher memory fraction = more KV cache = more concurrent requests

### 3. Batching parameters
- `--max-running-requests` — max concurrent sequences (try 128, 256, 512, 1024)
- `--max-total-tokens` — total token budget across all sequences
- `--schedule-policy` — scheduling algorithm: `lpm` (longest prefix match), `fcfs` (first come first served), `random`

### 4. Attention backend
- `--attention-backend flashinfer` — FlashInfer backend (default, usually fastest)
- `--attention-backend triton` — Triton backend (try if FlashInfer has issues)
- This can have significant impact depending on model architecture

## Medium-Impact Areas (try second)

### 5. Speculative decoding
- `--speculative-algorithm` — speculative decoding strategy
- `--num-speculative-tokens` — how many tokens to speculate ahead (try 3, 5, 8)
- Can improve latency significantly, may reduce throughput

### 6. Radix cache
- SGLang's unique prefix caching system, enabled by default
- `--disable-radix-cache` — test impact of disabling it
- Helps when many requests share common prefixes

### 7. CUDA graphs
- `--disable-cuda-graph` — disable CUDA graph capture (sometimes faster for small models)
- `--cuda-graph-max-bs` — max batch size for CUDA graphs (try 1, 4, 8, 16, 32)

## Low-Impact Areas (try last)

### 8. Data types
- `--dtype half` vs `--dtype bfloat16` vs `--dtype auto`
- Usually `auto` is fine, but worth testing

### 9. Parallelism
- `--tp-size` — tensor parallel size (only useful with multiple GPUs)
- `--dp-size` — data parallel size (only useful with multiple GPUs)

### 10. Advanced / experimental
- `--max-prefill-tokens` — limit prefill tokens per step
- `--enable-torch-compile` — torch.compile optimization
- `--disable-flashinfer` — fall back to default attention
- Any new flags discovered via `python -m sglang.launch_server --help`

## Strategy Tips

1. **Start with baseline** — always run the default config first
2. **One change at a time** — isolate the effect of each parameter
3. **Combine winners** — after finding individual improvements, combine them
4. **Binary search on continuous params** — for things like mem-fraction-static, narrow down the optimal value
5. **Read error messages** — crashes often reveal the boundary of what's possible
6. **Check `python -m sglang.launch_server --help` periodically** — you might discover flags you missed
