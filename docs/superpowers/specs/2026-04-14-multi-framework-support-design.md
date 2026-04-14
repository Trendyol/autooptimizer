# Multi-Framework Support Design

**Date:** 2026-04-14
**Status:** Approved
**Scope:** Add SGLang support alongside vLLM, with architecture that enables future TensorRT-LLM integration

## Context

Autooptimizer currently optimizes vLLM serving parameters exclusively. The serve script, benchmark tool, agent strategy docs, and experiment loop are all hardwired to vLLM. This design extends the system to support multiple inference frameworks, starting with SGLang.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework optimization mode | Per-framework first, cross-framework comparison later | Simpler to build and validate |
| Execution environment | All local installs, no Docker | Matches current vLLM setup |
| Benchmark strategy | Each framework uses its own native bench tool (primary); unified HTTP benchmark available via `--unified` flag for cross-framework comparison | Native tools stress framework-specific features; unified enables apples-to-apples comparison when needed |
| Agent knowledge organization | Separate strategy files per framework | Clean separation, agent loads only the relevant one |
| Project layout | Keep `edit/` vs `no_edit/` split, namespace by framework | Preserves the "what the agent can touch" boundary |
| Framework selection | `FRAMEWORK=` line in `memory/overview.md` | Consistent with existing `MODEL=` pattern |
| Initial scope | vLLM + SGLang | Architecturally similar (Python, OpenAI-compatible API, similar CLI patterns); TRT-LLM later |
| Approach | Shell-script-per-framework + thin Python routing | Minimal change from current design; preserves simplicity |

## Architecture

### Overview

`benchmark.py` becomes the routing layer. On startup it reads `FRAMEWORK` from `memory/overview.md` and uses it to select the correct serve script and adapter. The agent continues to edit shell scripts — one per framework — and the experiment loop is unchanged.

All frameworks serve on `127.0.0.1:8000` with an OpenAI-compatible API. Only one framework runs at a time.

### File Layout

```
project/
├── edit/
│   ├── vllm_serve_config.sh          # renamed from serve_config.sh
│   └── sglang_serve_config.sh        # new
├── no_edit/
│   ├── benchmark.py                  # modified: reads FRAMEWORK, routes
│   └── adapters/
│       ├── __init__.py
│       ├── vllm.py                   # new: vllm-specific bench cmd, health, result parsing
│       └── sglang.py                 # new: sglang-specific bench cmd, health, result parsing
memory/
├── overview.md                       # modified: adds FRAMEWORK= line
├── experiment_loop.md                # modified: framework-generic language
├── rules.md                          # modified: framework-generic language
├── setup.md                          # modified: framework-generic language
├── search_strategy_vllm.md           # renamed from search_strategy.md
└── search_strategy_sglang.md         # new
```

### Framework Adapter (`project/no_edit/adapters/`)

Each adapter is a Python module in `no_edit/` (not editable by the agent) that encapsulates framework differences. An adapter provides:

- **`SERVE_SCRIPT`** — path to the framework's editable serve config shell script
- **`HEALTH_ENDPOINT`** — the HTTP path used to health-check the server (e.g. `/health`)
- **`STRATEGY_FILE`** — path to the framework's search strategy markdown
- **`get_native_benchmark_cmd(model, base_url) -> list[str]`** — returns the shell command for the framework's native benchmark tool
- **`parse_native_result(duration) -> BenchmarkResult | None`** — reads the framework's benchmark output file and maps it into the common `BenchmarkResult` dataclass

#### vLLM Adapter

```python
SERVE_SCRIPT = "project/edit/vllm_serve_config.sh"
HEALTH_ENDPOINT = "/health"
STRATEGY_FILE = "memory/search_strategy_vllm.md"

def get_native_benchmark_cmd(model, base_url):
    # Calls: python -m vllm.entrypoints.cli.main bench serve ...
    # Same command as today's benchmark.py

def parse_native_result(duration):
    # Reads bench_result.json (vllm's output format)
    # Maps to BenchmarkResult dataclass
```

#### SGLang Adapter

```python
SERVE_SCRIPT = "project/edit/sglang_serve_config.sh"
HEALTH_ENDPOINT = "/health"
STRATEGY_FILE = "memory/search_strategy_sglang.md"

def get_native_benchmark_cmd(model, base_url):
    # Calls: python -m sglang.bench_serving ...
    # With same workload parameters (500 prompts, input=512, output=128, seed=42)

def parse_native_result(duration):
    # Reads SGLang's benchmark output format
    # Maps to BenchmarkResult dataclass
```

### Benchmark Strategy

**Default (native):** Each framework uses its own benchmark tool via the adapter. This is what the agent uses in the experiment loop. Scores are comparable within a framework but not directly across frameworks.

**Unified (`--unified` flag):** A framework-agnostic HTTP load generator built into `benchmark.py`. Sends requests to `/v1/completions`, measures TTFT/TPOT/ITL/E2EL/throughput from HTTP response timing and streaming chunks. Same workload parameters (500 prompts, input_len=512, output_len=128, seed=42, request_rate=inf). Used only when explicitly comparing across frameworks.

**CLI:**
```bash
uv run project/no_edit/benchmark.py run                # native bench tool (default)
uv run project/no_edit/benchmark.py run --unified       # unified HTTP benchmark
```

**Unchanged:** `compute_score()`, `BenchmarkResult` dataclass, result printing, TSV logging, GPU memory measurement.

### `benchmark.py` Changes

The main changes to `benchmark.py`:

1. **Read `FRAMEWORK`** from `memory/overview.md` at startup
2. **Import the correct adapter** based on `FRAMEWORK` value
3. **`run_benchmark()`** calls `adapter.get_native_benchmark_cmd()` by default, or the unified benchmark if `--unified` is passed
4. **`parse_result()`** delegates to `adapter.parse_native_result()` for native mode
5. **Start server command** references `adapter.SERVE_SCRIPT` instead of hardcoded path

Server lifecycle (health check, wait, kill via `lsof` on port) remains framework-agnostic.

### SGLang Serve Script

`project/edit/sglang_serve_config.sh`:

```bash
#!/bin/bash
MODEL="Qwen/Qwen3.5-27B-FP8"

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 8192
```

Same rules as vLLM: agent can change any flag, can't change model/host/port.

### SGLang Search Strategy

`memory/search_strategy_sglang.md` covers SGLang-specific tuning knobs, prioritized by expected impact:

**High-impact:**
- Memory management: `--mem-fraction-static`, `--chunked-prefill-size`
- Batching: `--max-running-requests`, `--max-total-tokens`, `--schedule-policy` (lpm/fcfs/random)
- Quantization: `--quantization fp8/awq/gptq`, `--kv-cache-dtype fp8_e5m2`

**Medium-impact:**
- Attention backends: `--attention-backend flashinfer/triton`
- Speculative decoding: `--speculative-algorithm`, `--num-speculative-tokens`
- Radix cache: `--disable-radix-cache` and tuning

**Low-impact:**
- CUDA graphs: `--disable-cuda-graph`, `--cuda-graph-max-bs`
- Data types, parallelism, advanced/experimental flags

### Memory File Updates

**`memory/overview.md`:**
- Add `FRAMEWORK=vllm` (or `sglang`) line in the Target Model section
- "Best Known Configuration" section shows config for the active framework

**`memory/rules.md`:**
- Replace vLLM-specific references with framework-generic language
- "Modify the active framework's serve config in `project/edit/`"
- "Use ANY flag supported by the active framework"
- "Discover available flags by running the framework's serve command with `--help`"
- Constraints unchanged: don't change model, port, host, or benchmark.py

**`memory/experiment_loop.md`:**
- Step 2: "Modify the active framework's serve config"
- Step 5: "Start the server" picks the right script via FRAMEWORK
- All other steps (git commit, benchmark, score check, keep/revert) unchanged

**`memory/setup.md`:**
- Step 3: read the active framework's strategy file instead of hardcoded `search_strategy.md`
- Step 5: "Check framework flags" — run the active framework's serve command with `--help` instead of hardcoded `vllm serve --help`
- Step 8: reference the active framework's strategy file and serve config
- References to `serve_config.sh` become "the active framework's serve config"

**`memory/search_strategy.md`:**
- Renamed to `search_strategy_vllm.md`, content unchanged

### Dependencies

`pyproject.toml` uses optional dependency groups:

```toml
[project.optional-dependencies]
vllm = ["vllm>=0.19.0"]
sglang = ["sglang[all]>=0.4.0"]
```

Install per framework: `uv sync --extra vllm` or `uv sync --extra sglang` or both.

Core dependencies (`requests`, `numpy`, `pandas`, `matplotlib`) remain in the base group.

## Error Handling

**Unknown/missing framework:** `benchmark.py` fails fast with a clear error listing supported frameworks. No silent fallback.

**Framework not installed:** Adapter detects this early (before server start) and prints install instructions (e.g. `"SGLang is not installed. Run: uv sync --extra sglang"`).

**Benchmark output parsing:** If a field is missing from a framework's output, default to `0.0`. Scoring only uses `throughput_tok_per_sec` and `mean_e2el_ms`, so missing secondary metrics don't break scoring.

**Server didn't start:** Same as today — `wait_for_server()` polls health endpoint with timeout. Framework-agnostic.

**Port conflict:** Only one framework runs at a time. Rules.md notes: "Kill the previous server before switching frameworks."

## Out of Scope

- **TensorRT-LLM support** — Future. Adapter pattern enables it but not implemented now.
- **Automated cross-framework comparison workflow** — The `--unified` flag enables manual comparison. No "run all frameworks, pick winner" automation.
- **Shared results across frameworks** — Each framework's experiments are independent per run.
- **Bayesian optimization / smarter search** — Remains on roadmap.
- **Web dashboard / production tooling** — Remains on roadmap.
- **Auto-detection of installed frameworks** — Manual `FRAMEWORK=` setting only.
