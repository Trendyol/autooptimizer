# Autooptimizer

Automated LLM serving parameter optimization using LLM agents.

Inspired by Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch), which lets AI agents autonomously iterate on LLM training code to minimize validation loss. Autooptimizer applies the same autonomous experiment loop idea to a different domain: instead of optimizing training code, it optimizes **serving parameters** to maximize inference throughput and minimize latency.

## Supported Frameworks

| Framework | Status |
|-----------|--------|
| [vLLM](https://github.com/vllm-project/vllm) | Supported |
| [SGLang](https://github.com/sgl-project/sglang) | Supported |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Planned |

## Overview

An LLM agent iteratively:

1. Proposes hypotheses for improving serving performance
2. Modifies the framework's serve configuration
3. Starts the server, runs benchmarks, records results
4. Keeps improvements, reverts failures
5. Loops indefinitely until manually stopped

**Goal**: Maximize `score` (throughput / latency composite metric).

## Project Structure

```
autooptimizer/
├── memory/                              # Agent instructions and context
│   ├── overview.md                      # Target model, framework, goal, best known config
│   ├── experiment_loop.md               # Experiment workflow
│   ├── setup.md                         # How to start a new run
│   ├── rules.md                         # Allowed/prohibited changes
│   ├── search_strategy_vllm.md          # Hypothesis strategy for vLLM params
│   └── search_strategy_sglang.md        # Hypothesis strategy for SGLang params
├── project/
│   ├── edit/
│   │   ├── vllm_serve_config.sh         # Editable vLLM serve command
│   │   └── sglang_serve_config.sh       # Editable SGLang serve command
│   └── no_edit/
│       ├── benchmark.py                 # Fixed benchmark runner & scoring
│       └── adapters/                    # Framework adapters (routing logic)
│           ├── __init__.py
│           ├── vllm.py
│           └── sglang.py
├── tests/                               # Pytest suite
│   ├── test_adapters.py
│   └── test_benchmark_routing.py
├── artifacts/                           # Generated at runtime (not committed)
│   ├── hypothesis_backlog.tsv           # Experiment queue
│   └── results.tsv                      # Experiment results log
├── pyproject.toml
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
# For vLLM optimization
uv sync --extra vllm

# For SGLang optimization
uv sync --extra sglang

# Both
uv sync --extra vllm --extra sglang
```

### 2. Set the framework

In `memory/overview.md`, set the `FRAMEWORK` line:

```
FRAMEWORK=vllm
```

or

```
FRAMEWORK=sglang
```

### 3. Start the agent

```bash
cursor agent --yolo
```

> "Hi, refresh your memory with .md files in @memory folder, and let's kick off a new experiment! Let's do the setup first."

The agent will:
- Create a new experiment branch (e.g., `autooptimizer/apr11`)
- Run baseline benchmark
- Begin the experiment loop

### 4. Manual experiment

```bash
# Start server (example for vLLM)
bash project/edit/vllm_serve_config.sh > server.log 2>&1 &

# Wait + benchmark + kill
uv run project/no_edit/benchmark.py run
```

## Configuration

Set the target model and framework in `memory/overview.md`:

```
MODEL=Qwen/Qwen2.5-1.5B-Instruct
FRAMEWORK=vllm
```

If you have a known-good configuration from previous experiments, update the "Best Known Configuration" section in `memory/overview.md`.

## Constraints

- **Editable**: Only the active framework's serve config in `project/edit/`
- **Metric**: Optimize `score` only (higher is better)
- **Model**: Fixed per experiment run (set in `memory/overview.md`)
- **Framework**: Fixed per experiment run (set in `memory/overview.md`)

## Results Format

Tab-separated file (`artifacts/results.tsv`):

```
experiment    score    memory_gb    status    description
```

Status values: `keep`, `discard`, `crash`

## Roadmap

### Current
- [x] Autonomous experiment loop with LLM agent
- [x] vLLM serve parameter tuning
- [x] SGLang serve parameter tuning
- [x] Multi-framework support with adapter pattern
- [x] Deterministic benchmarking with fixed seeds
- [x] Hypothesis backlog with priority-based ordering
- [x] Automatic keep/revert based on score improvement

### Planned
- [ ] TensorRT-LLM support
- [ ] Unified cross-framework benchmark comparison (`--unified` flag)
- [ ] Smarter search — Bayesian optimization, parameter interaction detection, Pareto frontier visualization
- [ ] Hardware-aware profiles — GPU auto-detection, per-family defaults (A100, H100, L40S, ...)
- [ ] Production tooling — web dashboard, exportable configs (Docker/K8s), CI/CD integration

## License

MIT
