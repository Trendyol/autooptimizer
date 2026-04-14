# Autooptimizer Overview

Automated vLLM serving parameter optimization using LLM agents.

## Target Model

```
MODEL=Qwen/Qwen3.5-27B-FP8
FRAMEWORK=vllm
```

**Do NOT change the model.** All experiments must use this exact model. The goal is to find the best vLLM serving parameters for this specific model on this specific hardware.

## Best Known Configuration

```bash
vllm serve Qwen/Qwen3.5-27B-FP8 --reasoning-parser qwen3 \
                          --enable-auto-tool-choice \
                          --tool-call-parser qwen3_coder \
                          --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
                          --enable-chunked-prefill \
                          --max-num-batched-tokens 16384 \
                          --enable-prefix-caching \
                          --kv-cache-dtype fp8 \
                          --gpu-memory-utilization 0.90
```

> When the user provides a better configuration from previous experiments, it will be updated here.

## Goal

**Get the highest `score`** — higher is better.

```
score = throughput_tok_per_sec / (1 + mean_e2el_ms / 1000)
```

This rewards high token throughput while penalizing high end-to-end response latency (total time from request to completion).

## Files

| File | Purpose | Editable |
|------|---------|----------|
| `project/edit/serve_config.sh` | vLLM serve command with parameters | **Yes** |
| `project/no_edit/benchmark.py` | Benchmark runner, scoring, server management | No |
| `artifacts/hypothesis_backlog.tsv` | Prioritized experiment queue | Yes (untracked) |
| `artifacts/results.tsv` | Experiment results log | Yes (untracked) |
| `logs/<experiment>.log` | Per-experiment logs | Auto-generated (untracked) |

## Output Format

Benchmark uses a **fixed seed (42)**, fixed input/output lengths (512/128), disabled shuffle, and zero range ratio to ensure **deterministic, reproducible** results across runs.

After a benchmark, `benchmark.py` prints:

```
---
score:                1234.56
throughput_req/s:     45.23
throughput_tok/s:     5789.12
mean_ttft_ms:         12.34
median_ttft_ms:       10.56
p99_ttft_ms:          45.67
mean_tpot_ms:         3.45
median_tpot_ms:       3.12
p99_tpot_ms:          8.90
mean_itl_ms:          3.21
mean_e2el_ms:         456.78
p99_e2el_ms:          890.12
completed_requests:   500
total_output_tokens:  64000
benchmark_duration_s: 11.1
peak_gpu_memory_mb:   45060.2
```

Extract key metric: `grep "^score:" logs/<experiment>.log`
