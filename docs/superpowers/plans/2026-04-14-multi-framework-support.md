# Multi-Framework Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SGLang support alongside vLLM by introducing a framework adapter pattern, per-framework serve scripts and strategy files, and a unified benchmark option.

**Architecture:** `benchmark.py` reads a `FRAMEWORK=` line from `memory/overview.md` and dynamically loads the matching adapter from `project/no_edit/adapters/`. Each adapter defines the serve script path, health endpoint, native benchmark command, and result parser. The agent edits framework-specific shell scripts in `project/edit/`.

**Tech Stack:** Python 3.10+, shell scripts, markdown agent playbooks. Frameworks: vLLM (>=0.19.0), SGLang (>=0.4.0) as optional deps.

**Spec:** `docs/superpowers/specs/2026-04-14-multi-framework-support-design.md`

---

## File Structure

### New files
- `project/no_edit/adapters/__init__.py` — adapter loading/routing logic
- `project/no_edit/adapters/vllm.py` — vLLM adapter (bench cmd, result parsing)
- `project/no_edit/adapters/sglang.py` — SGLang adapter (bench cmd, result parsing)
- `project/edit/sglang_serve_config.sh` — SGLang serve script (agent-editable)
- `memory/search_strategy_sglang.md` — SGLang tuning strategy
- `tests/test_adapters.py` — adapter unit tests
- `tests/test_benchmark_routing.py` — benchmark.py routing tests

### Modified files
- `project/edit/serve_config.sh` → renamed to `project/edit/vllm_serve_config.sh`
- `memory/search_strategy.md` → renamed to `memory/search_strategy_vllm.md`
- `project/no_edit/benchmark.py` — add framework routing, adapter delegation, `--unified` flag
- `memory/overview.md` — add `FRAMEWORK=` line
- `memory/rules.md` — framework-generic language
- `memory/experiment_loop.md` — framework-generic language
- `memory/setup.md` — framework-generic language
- `pyproject.toml` — optional dependency groups, add pytest
- `README.md` — update for multi-framework

---

## Task 1: Project setup — pytest and optional dependency groups

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py`

- [ ] **Step 1: Update `pyproject.toml` with optional deps and pytest**

Replace the entire `pyproject.toml` contents with:

```toml
[project]
name = "autooptimizer"
version = "0.2.0"
description = "Autonomous LLM serving parameter optimization via AI agents"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "requests>=2.32.0",
    "numpy>=2.2.0",
    "pandas>=2.2.0",
    "matplotlib>=3.9.0",
]

[project.optional-dependencies]
vllm = ["vllm>=0.19.0"]
sglang = ["sglang[all]>=0.4.0"]
dev = ["pytest>=8.0.0"]
```

Key changes: `vllm` moved from base deps to optional group, `sglang` group added, `pytest` added under `dev`, version bumped to 0.2.0, description generalized.

- [ ] **Step 2: Create empty test package**

Create `tests/__init__.py` as an empty file.

- [ ] **Step 3: Install dev deps and verify pytest works**

Run: `uv sync --extra dev`
Then: `uv run pytest --version`
Expected: pytest version prints successfully.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/__init__.py
git commit -m "feat: add optional dep groups and pytest setup"
```

---

## Task 2: Rename existing vLLM files

Rename the serve config and search strategy to framework-namespaced names before building the adapter system on top.

**Files:**
- Rename: `project/edit/serve_config.sh` → `project/edit/vllm_serve_config.sh`
- Rename: `memory/search_strategy.md` → `memory/search_strategy_vllm.md`

- [ ] **Step 1: Rename serve_config.sh**

```bash
git mv project/edit/serve_config.sh project/edit/vllm_serve_config.sh
```

- [ ] **Step 2: Update the header comment in `vllm_serve_config.sh`**

Change line 2 from:
```bash
# vLLM Serve Configuration — THIS FILE IS EDITED BY THE AGENT
```
to:
```bash
# vLLM Serve Configuration — THIS FILE IS EDITED BY THE AGENT
# Usage: bash project/edit/vllm_serve_config.sh
```

And remove the old `# Usage:` line (line 8) that references the old path.

- [ ] **Step 3: Rename search_strategy.md**

```bash
git mv memory/search_strategy.md memory/search_strategy_vllm.md
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: rename vllm files to framework-namespaced names"
```

---

## Task 3: Build the adapter loading module (`adapters/__init__.py`)

This is the routing core — it reads `FRAMEWORK` from `overview.md` and returns the right adapter module.

**Files:**
- Create: `project/no_edit/adapters/__init__.py`
- Create: `tests/test_adapters.py`

- [ ] **Step 1: Write the failing test for `load_adapter`**

Create `tests/test_adapters.py`:

```python
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "project" / "no_edit"))

from adapters import load_adapter, read_framework_from_overview, SUPPORTED_FRAMEWORKS


class TestReadFrameworkFromOverview:
    def test_reads_vllm(self, tmp_path):
        overview = tmp_path / "overview.md"
        overview.write_text("# Overview\n\n```\nMODEL=Qwen/Qwen3.5-27B-FP8\nFRAMEWORK=vllm\n```\n")
        assert read_framework_from_overview(str(overview)) == "vllm"

    def test_reads_sglang(self, tmp_path):
        overview = tmp_path / "overview.md"
        overview.write_text("# Overview\n\n```\nMODEL=Qwen/Qwen3.5-27B-FP8\nFRAMEWORK=sglang\n```\n")
        assert read_framework_from_overview(str(overview)) == "sglang"

    def test_missing_framework_raises(self, tmp_path):
        overview = tmp_path / "overview.md"
        overview.write_text("# Overview\n\n```\nMODEL=Qwen/Qwen3.5-27B-FP8\n```\n")
        with pytest.raises(SystemExit):
            read_framework_from_overview(str(overview))

    def test_unknown_framework_raises(self, tmp_path):
        overview = tmp_path / "overview.md"
        overview.write_text("# Overview\n\n```\nFRAMEWORK=tensorrt\n```\n")
        with pytest.raises(SystemExit):
            read_framework_from_overview(str(overview))


class TestLoadAdapter:
    def test_load_vllm_adapter(self):
        adapter = load_adapter("vllm")
        assert hasattr(adapter, "SERVE_SCRIPT")
        assert hasattr(adapter, "HEALTH_ENDPOINT")
        assert hasattr(adapter, "STRATEGY_FILE")
        assert hasattr(adapter, "get_native_benchmark_cmd")
        assert hasattr(adapter, "parse_native_result")
        assert "vllm" in adapter.SERVE_SCRIPT

    def test_load_sglang_adapter(self):
        adapter = load_adapter("sglang")
        assert hasattr(adapter, "SERVE_SCRIPT")
        assert "sglang" in adapter.SERVE_SCRIPT

    def test_load_unknown_raises(self):
        with pytest.raises(SystemExit):
            load_adapter("unknown_framework")

    def test_supported_frameworks_list(self):
        assert "vllm" in SUPPORTED_FRAMEWORKS
        assert "sglang" in SUPPORTED_FRAMEWORKS
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_adapters.py -v`
Expected: FAIL — `adapters` module does not exist yet.

- [ ] **Step 3: Implement `adapters/__init__.py`**

Create `project/no_edit/adapters/__init__.py`:

```python
"""
Framework adapter loader.
Reads FRAMEWORK from memory/overview.md and loads the matching adapter module.
"""

import importlib
import re
import sys
from pathlib import Path

SUPPORTED_FRAMEWORKS = ["vllm", "sglang"]

OVERVIEW_PATH = Path(__file__).resolve().parent.parent.parent.parent / "memory" / "overview.md"


def read_framework_from_overview(overview_path: str | None = None) -> str:
    path = Path(overview_path) if overview_path else OVERVIEW_PATH
    if not path.exists():
        print(f"ERROR: {path} not found.")
        sys.exit(1)

    text = path.read_text()
    match = re.search(r"^FRAMEWORK=(\S+)", text, re.MULTILINE)
    if not match:
        print(f"ERROR: No FRAMEWORK= line found in {path}.")
        print(f"Add a line like: FRAMEWORK=vllm")
        print(f"Supported frameworks: {', '.join(SUPPORTED_FRAMEWORKS)}")
        sys.exit(1)

    framework = match.group(1).strip().lower()
    if framework not in SUPPORTED_FRAMEWORKS:
        print(f"ERROR: Unknown framework '{framework}'.")
        print(f"Supported frameworks: {', '.join(SUPPORTED_FRAMEWORKS)}")
        sys.exit(1)

    return framework


def load_adapter(framework: str):
    if framework not in SUPPORTED_FRAMEWORKS:
        print(f"ERROR: Unknown framework '{framework}'.")
        print(f"Supported frameworks: {', '.join(SUPPORTED_FRAMEWORKS)}")
        sys.exit(1)

    module_name = f"adapters.{framework}"
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        print(f"ERROR: Could not load adapter for '{framework}': {e}")
        sys.exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_adapters.py -v`
Expected: All tests fail because the individual adapter modules (`adapters.vllm`, `adapters.sglang`) don't exist yet. The `TestReadFrameworkFromOverview` tests should pass. The `TestLoadAdapter` tests for vllm/sglang will fail — that's expected, we build those adapters in the next tasks.

Run just the framework-reading tests first:
```bash
uv run pytest tests/test_adapters.py::TestReadFrameworkFromOverview -v
```
Expected: All 4 pass.

- [ ] **Step 5: Commit**

```bash
git add project/no_edit/adapters/__init__.py tests/test_adapters.py
git commit -m "feat: add adapter loader with framework detection from overview.md"
```

---

## Task 4: Build the vLLM adapter (`adapters/vllm.py`)

Extract the vLLM-specific benchmark command and result parsing from the current `benchmark.py` into a standalone adapter module.

**Files:**
- Create: `project/no_edit/adapters/vllm.py`

- [ ] **Step 1: Write the vLLM adapter**

Create `project/no_edit/adapters/vllm.py`:

```python
"""
vLLM framework adapter.
Defines serve script path, health endpoint, and native benchmark command/parsing.
"""

import json
import os
import sys
from pathlib import Path

SERVE_SCRIPT = "project/edit/vllm_serve_config.sh"
HEALTH_ENDPOINT = "/health"
STRATEGY_FILE = "memory/search_strategy_vllm.md"
RESULT_FILE = "bench_result.json"


def get_native_benchmark_cmd(
    model: str,
    base_url: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int,
    request_rate: str,
    result_filename: str,
) -> list[str]:
    return [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "serve",
        "--backend", "openai",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", "random",
        "--num-prompts", str(num_prompts),
        "--input-len", str(input_len),
        "--output-len", str(output_len),
        "--seed", str(seed),
        "--random-range-ratio", "0.0",
        "--disable-shuffle",
        "--request-rate", request_rate,
        "--save-result",
        "--result-dir", ".",
        "--result-filename", result_filename,
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,99",
    ]


def parse_native_result(result_file: str, duration: float) -> dict | None:
    if not os.path.exists(result_file):
        print(f"Result file {result_file} not found")
        return None

    try:
        with open(result_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Failed to parse result: {e}")
        return None

    return {
        "benchmark_duration_sec": data.get("duration", duration),
        "completed_requests": data.get("completed", 0),
        "total_input_tokens": data.get("total_input_tokens", 0),
        "total_output_tokens": data.get("total_output_tokens", 0),
        "throughput_req_per_sec": data.get("request_throughput", 0.0),
        "throughput_tok_per_sec": data.get("output_throughput", 0.0),
        "mean_ttft_ms": data.get("mean_ttft_ms", 0.0),
        "median_ttft_ms": data.get("median_ttft_ms", 0.0),
        "p99_ttft_ms": data.get("p99_ttft_ms", 0.0),
        "mean_tpot_ms": data.get("mean_tpot_ms", 0.0),
        "median_tpot_ms": data.get("median_tpot_ms", 0.0),
        "p99_tpot_ms": data.get("p99_tpot_ms", 0.0),
        "mean_itl_ms": data.get("mean_itl_ms", 0.0),
        "median_itl_ms": data.get("median_itl_ms", 0.0),
        "p99_itl_ms": data.get("p99_itl_ms", 0.0),
        "mean_e2el_ms": data.get("mean_e2el_ms", 0.0),
        "median_e2el_ms": data.get("median_e2el_ms", 0.0),
        "p99_e2el_ms": data.get("p99_e2el_ms", 0.0),
    }
```

- [ ] **Step 2: Run the adapter loading test for vLLM**

```bash
uv run pytest tests/test_adapters.py::TestLoadAdapter::test_load_vllm_adapter -v
```
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add project/no_edit/adapters/vllm.py
git commit -m "feat: add vllm adapter with native benchmark cmd and result parsing"
```

---

## Task 5: Build the SGLang adapter (`adapters/sglang.py`)

**Files:**
- Create: `project/no_edit/adapters/sglang.py`

- [ ] **Step 1: Write the SGLang adapter**

Create `project/no_edit/adapters/sglang.py`:

```python
"""
SGLang framework adapter.
Defines serve script path, health endpoint, and native benchmark command/parsing.
"""

import json
import os
import sys
from pathlib import Path

SERVE_SCRIPT = "project/edit/sglang_serve_config.sh"
HEALTH_ENDPOINT = "/health"
STRATEGY_FILE = "memory/search_strategy_sglang.md"
RESULT_FILE = "sglang_bench_result.json"


def get_native_benchmark_cmd(
    model: str,
    base_url: str,
    num_prompts: int,
    input_len: int,
    output_len: int,
    seed: int,
    request_rate: str,
    result_filename: str,
) -> list[str]:
    return [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "openai",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", "random",
        "--num-prompts", str(num_prompts),
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--seed", str(seed),
        "--random-range-ratio", "0.0",
        "--disable-shuffle",
        "--request-rate", str(request_rate),
        "--save-result",
        "--result-filename", result_filename,
    ]


def parse_native_result(result_file: str, duration: float) -> dict | None:
    if not os.path.exists(result_file):
        print(f"Result file {result_file} not found")
        return None

    try:
        with open(result_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Failed to parse result: {e}")
        return None

    return {
        "benchmark_duration_sec": data.get("duration", duration),
        "completed_requests": data.get("completed", 0),
        "total_input_tokens": data.get("total_input_tokens", 0),
        "total_output_tokens": data.get("total_output_tokens", 0),
        "throughput_req_per_sec": data.get("request_throughput", 0.0),
        "throughput_tok_per_sec": data.get("output_throughput", 0.0),
        "mean_ttft_ms": data.get("mean_ttft_ms", 0.0),
        "median_ttft_ms": data.get("median_ttft_ms", 0.0),
        "p99_ttft_ms": data.get("p99_ttft_ms", 0.0),
        "mean_tpot_ms": data.get("mean_tpot_ms", 0.0),
        "median_tpot_ms": data.get("median_tpot_ms", 0.0),
        "p99_tpot_ms": data.get("p99_tpot_ms", 0.0),
        "mean_itl_ms": data.get("mean_itl_ms", 0.0),
        "median_itl_ms": data.get("median_itl_ms", 0.0),
        "p99_itl_ms": data.get("p99_itl_ms", 0.0),
        "mean_e2el_ms": data.get("mean_e2el_ms", 0.0),
        "median_e2el_ms": data.get("median_e2el_ms", 0.0),
        "p99_e2el_ms": data.get("p99_e2el_ms", 0.0),
    }
```

- [ ] **Step 2: Run the adapter loading test for SGLang**

```bash
uv run pytest tests/test_adapters.py::TestLoadAdapter::test_load_sglang_adapter -v
```
Expected: PASS

- [ ] **Step 3: Run all adapter tests**

```bash
uv run pytest tests/test_adapters.py -v
```
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add project/no_edit/adapters/sglang.py
git commit -m "feat: add sglang adapter with native benchmark cmd and result parsing"
```

---

## Task 6: Refactor `benchmark.py` to use adapters

This is the main integration task. `benchmark.py` stops hardcoding vLLM and delegates to the loaded adapter.

**Files:**
- Modify: `project/no_edit/benchmark.py`
- Create: `tests/test_benchmark_routing.py`

- [ ] **Step 1: Write tests for benchmark routing**

Create `tests/test_benchmark_routing.py`:

```python
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "project" / "no_edit"))

from adapters import read_framework_from_overview


class TestBenchmarkFrameworkRouting:
    """Verify that benchmark.py correctly reads FRAMEWORK and picks the right adapter."""

    def test_vllm_adapter_serve_script(self, tmp_path):
        overview = tmp_path / "overview.md"
        overview.write_text("```\nMODEL=test\nFRAMEWORK=vllm\n```\n")
        fw = read_framework_from_overview(str(overview))
        from adapters import load_adapter
        adapter = load_adapter(fw)
        assert adapter.SERVE_SCRIPT == "project/edit/vllm_serve_config.sh"
        assert adapter.HEALTH_ENDPOINT == "/health"

    def test_sglang_adapter_serve_script(self, tmp_path):
        overview = tmp_path / "overview.md"
        overview.write_text("```\nMODEL=test\nFRAMEWORK=sglang\n```\n")
        fw = read_framework_from_overview(str(overview))
        from adapters import load_adapter
        adapter = load_adapter(fw)
        assert adapter.SERVE_SCRIPT == "project/edit/sglang_serve_config.sh"
        assert adapter.HEALTH_ENDPOINT == "/health"


class TestVllmNativeBenchmarkCmd:
    def test_cmd_structure(self):
        from adapters.vllm import get_native_benchmark_cmd
        cmd = get_native_benchmark_cmd(
            model="test-model",
            base_url="http://127.0.0.1:8000",
            num_prompts=500,
            input_len=512,
            output_len=128,
            seed=42,
            request_rate="inf",
            result_filename="bench_result.json",
        )
        assert "bench" in cmd
        assert "serve" in cmd
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "test-model"
        assert "--num-prompts" in cmd
        idx = cmd.index("--num-prompts")
        assert cmd[idx + 1] == "500"


class TestSglangNativeBenchmarkCmd:
    def test_cmd_structure(self):
        from adapters.sglang import get_native_benchmark_cmd
        cmd = get_native_benchmark_cmd(
            model="test-model",
            base_url="http://127.0.0.1:8000",
            num_prompts=500,
            input_len=512,
            output_len=128,
            seed=42,
            request_rate="inf",
            result_filename="sglang_bench_result.json",
        )
        assert "sglang.bench_serving" in " ".join(cmd)
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "test-model"


class TestVllmParseResult:
    def test_parse_valid_result(self, tmp_path):
        import json
        result_file = tmp_path / "bench_result.json"
        result_file.write_text(json.dumps({
            "duration": 10.5,
            "completed": 500,
            "total_input_tokens": 256000,
            "total_output_tokens": 64000,
            "request_throughput": 47.6,
            "output_throughput": 6095.2,
            "mean_ttft_ms": 12.3,
            "median_ttft_ms": 10.1,
            "p99_ttft_ms": 45.6,
            "mean_tpot_ms": 3.4,
            "median_tpot_ms": 3.1,
            "p99_tpot_ms": 8.9,
            "mean_itl_ms": 3.2,
            "median_itl_ms": 2.9,
            "p99_itl_ms": 7.8,
            "mean_e2el_ms": 456.7,
            "median_e2el_ms": 420.1,
            "p99_e2el_ms": 890.1,
        }))
        from adapters.vllm import parse_native_result
        result = parse_native_result(str(result_file), 10.5)
        assert result is not None
        assert result["throughput_tok_per_sec"] == 6095.2
        assert result["mean_e2el_ms"] == 456.7
        assert result["completed_requests"] == 500

    def test_parse_missing_file(self, tmp_path):
        from adapters.vllm import parse_native_result
        result = parse_native_result(str(tmp_path / "nonexistent.json"), 0.0)
        assert result is None


class TestSglangParseResult:
    def test_parse_valid_result(self, tmp_path):
        import json
        result_file = tmp_path / "sglang_bench_result.json"
        result_file.write_text(json.dumps({
            "duration": 11.2,
            "completed": 500,
            "total_input_tokens": 256000,
            "total_output_tokens": 64000,
            "request_throughput": 44.6,
            "output_throughput": 5714.3,
            "mean_ttft_ms": 14.5,
            "mean_e2el_ms": 502.3,
        }))
        from adapters.sglang import parse_native_result
        result = parse_native_result(str(result_file), 11.2)
        assert result is not None
        assert result["throughput_tok_per_sec"] == 5714.3
        assert result["mean_e2el_ms"] == 502.3

    def test_missing_fields_default_to_zero(self, tmp_path):
        import json
        result_file = tmp_path / "sglang_bench_result.json"
        result_file.write_text(json.dumps({
            "output_throughput": 5000.0,
            "mean_e2el_ms": 400.0,
        }))
        from adapters.sglang import parse_native_result
        result = parse_native_result(str(result_file), 5.0)
        assert result is not None
        assert result["mean_ttft_ms"] == 0.0
        assert result["p99_tpot_ms"] == 0.0
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
uv run pytest tests/test_benchmark_routing.py -v
```
Expected: All tests PASS (these test adapters directly, not benchmark.py integration yet).

- [ ] **Step 3: Rewrite `benchmark.py` to use adapter routing**

Replace the full contents of `project/no_edit/benchmark.py` with:

```python
"""
Autooptimizer benchmark tool.
Waits for a running server, runs benchmarks, reports results.

DO NOT MODIFY THIS FILE.

Commands:
    python benchmark.py wait                — wait until server is healthy
    python benchmark.py benchmark           — run benchmark against running server
    python benchmark.py benchmark --model X — run benchmark with explicit model name
    python benchmark.py kill                — kill server on port 8000
    python benchmark.py run                 — wait + benchmark + kill (full pipeline)
    python benchmark.py health              — check if server is running (exit 0/1)
"""

import os
import sys
import time
import signal
import subprocess
import argparse
from dataclasses import dataclass
from typing import Optional

import requests as http_requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapters import read_framework_from_overview, load_adapter

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

SERVER_PORT = 8000
SERVER_HOST = "127.0.0.1"
BASE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
SERVER_STARTUP_TIMEOUT = 300

BENCHMARK_TIMEOUT = 240
NUM_PROMPTS = 500
BENCHMARK_INPUT_LEN = 512
BENCHMARK_OUTPUT_LEN = 128
BENCHMARK_SEED = 42
REQUEST_RATE = "inf"
RESULT_FILE = "bench_result.json"

# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    throughput_req_per_sec: float = 0.0
    throughput_tok_per_sec: float = 0.0
    mean_ttft_ms: float = 0.0
    median_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    median_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0
    mean_itl_ms: float = 0.0
    median_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    mean_e2el_ms: float = 0.0
    median_e2el_ms: float = 0.0
    p99_e2el_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    completed_requests: int = 0
    benchmark_duration_sec: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    score: float = 0.0

# ---------------------------------------------------------------------------
# Scoring function (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def compute_score(result: BenchmarkResult) -> float:
    """
    Composite score: higher is better.
    Score = throughput_tok_per_sec / (1 + mean_e2el_ms / 1000)
    """
    if result.throughput_tok_per_sec <= 0:
        return 0.0
    e2el_penalty = 1.0 + result.mean_e2el_ms / 1000.0
    return result.throughput_tok_per_sec / e2el_penalty

# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------

def check_server_health(adapter) -> bool:
    url = f"{BASE_URL}{adapter.HEALTH_ENDPOINT}"
    try:
        resp = http_requests.get(url, timeout=5)
        return resp.status_code == 200
    except (http_requests.ConnectionError, http_requests.Timeout):
        return False


def wait_for_server(adapter, timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    url = f"{BASE_URL}{adapter.HEALTH_ENDPOINT}"
    print(f"Waiting for server at {url} ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if check_server_health(adapter):
            elapsed = time.time() - t0
            print(f"Server is healthy! (took {elapsed:.1f}s)")
            return True
        time.sleep(2)
    print(f"Server did not become healthy within {timeout}s")
    return False


def find_server_pids() -> list[int]:
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{SERVER_PORT}"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
    except (subprocess.TimeoutExpired, ValueError):
        pass
    return []


def kill_server():
    pids = find_server_pids()
    if not pids:
        print(f"No server found on port {SERVER_PORT}")
        return
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"Sent SIGTERM to PID {pid}")
        except ProcessLookupError:
            pass
    time.sleep(3)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
            print(f"Sent SIGKILL to PID {pid}")
        except ProcessLookupError:
            pass
    print("Server killed.")

# ---------------------------------------------------------------------------
# GPU memory
# ---------------------------------------------------------------------------

def get_gpu_memory_mb() -> float:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            values = [float(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]
            return max(values) if values else 0.0
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0.0

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def detect_model_name() -> Optional[str]:
    try:
        resp = http_requests.get(f"{BASE_URL}/v1/models", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id")
    except Exception:
        pass
    return None


def run_benchmark(adapter, model_name: str) -> Optional[BenchmarkResult]:
    result_file = getattr(adapter, "RESULT_FILE", RESULT_FILE)
    if os.path.exists(result_file):
        os.remove(result_file)

    bench_cmd = adapter.get_native_benchmark_cmd(
        model=model_name,
        base_url=BASE_URL,
        num_prompts=NUM_PROMPTS,
        input_len=BENCHMARK_INPUT_LEN,
        output_len=BENCHMARK_OUTPUT_LEN,
        seed=BENCHMARK_SEED,
        request_rate=REQUEST_RATE,
        result_filename=result_file,
    )

    print(f"Running benchmark ({NUM_PROMPTS} prompts, input={BENCHMARK_INPUT_LEN}, output={BENCHMARK_OUTPUT_LEN})...")
    t0 = time.time()
    try:
        result = subprocess.run(bench_cmd, capture_output=True, text=True, timeout=BENCHMARK_TIMEOUT)
        duration = time.time() - t0

        if result.stdout:
            for line in result.stdout.strip().split("\n")[-25:]:
                print(f"  {line}")

        if result.returncode != 0:
            print(f"Benchmark failed (exit code {result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[-10:]:
                    print(f"  {line}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Benchmark timed out after {BENCHMARK_TIMEOUT}s")
        return None

    return parse_result(adapter, result_file, duration)


def parse_result(adapter, result_file: str, duration: float) -> Optional[BenchmarkResult]:
    raw = adapter.parse_native_result(result_file, duration)
    if raw is None:
        return None

    br = BenchmarkResult()
    br.benchmark_duration_sec = raw.get("benchmark_duration_sec", duration)
    br.completed_requests = raw.get("completed_requests", 0)
    br.total_input_tokens = raw.get("total_input_tokens", 0)
    br.total_output_tokens = raw.get("total_output_tokens", 0)
    br.throughput_req_per_sec = raw.get("throughput_req_per_sec", 0.0)
    br.throughput_tok_per_sec = raw.get("throughput_tok_per_sec", 0.0)
    br.mean_ttft_ms = raw.get("mean_ttft_ms", 0.0)
    br.median_ttft_ms = raw.get("median_ttft_ms", 0.0)
    br.p99_ttft_ms = raw.get("p99_ttft_ms", 0.0)
    br.mean_tpot_ms = raw.get("mean_tpot_ms", 0.0)
    br.median_tpot_ms = raw.get("median_tpot_ms", 0.0)
    br.p99_tpot_ms = raw.get("p99_tpot_ms", 0.0)
    br.mean_itl_ms = raw.get("mean_itl_ms", 0.0)
    br.median_itl_ms = raw.get("median_itl_ms", 0.0)
    br.p99_itl_ms = raw.get("p99_itl_ms", 0.0)
    br.mean_e2el_ms = raw.get("mean_e2el_ms", 0.0)
    br.median_e2el_ms = raw.get("median_e2el_ms", 0.0)
    br.p99_e2el_ms = raw.get("p99_e2el_ms", 0.0)
    br.peak_gpu_memory_mb = get_gpu_memory_mb()
    br.score = compute_score(br)
    return br


def print_results(result: BenchmarkResult):
    print("---")
    print(f"score:                {result.score:.2f}")
    print(f"throughput_req/s:     {result.throughput_req_per_sec:.2f}")
    print(f"throughput_tok/s:     {result.throughput_tok_per_sec:.2f}")
    print(f"mean_ttft_ms:         {result.mean_ttft_ms:.2f}")
    print(f"median_ttft_ms:       {result.median_ttft_ms:.2f}")
    print(f"p99_ttft_ms:          {result.p99_ttft_ms:.2f}")
    print(f"mean_tpot_ms:         {result.mean_tpot_ms:.2f}")
    print(f"median_tpot_ms:       {result.median_tpot_ms:.2f}")
    print(f"p99_tpot_ms:          {result.p99_tpot_ms:.2f}")
    print(f"mean_itl_ms:          {result.mean_itl_ms:.2f}")
    print(f"mean_e2el_ms:         {result.mean_e2el_ms:.2f}")
    print(f"p99_e2el_ms:          {result.p99_e2el_ms:.2f}")
    print(f"completed_requests:   {result.completed_requests}")
    print(f"total_output_tokens:  {result.total_output_tokens}")
    print(f"benchmark_duration_s: {result.benchmark_duration_sec:.1f}")
    print(f"peak_gpu_memory_mb:   {result.peak_gpu_memory_mb:.1f}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="autooptimizer — benchmark tool for LLM serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server in background, then:
  python benchmark.py wait
  python benchmark.py benchmark
  python benchmark.py kill

  # Or all-in-one:
  python benchmark.py run
        """,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("wait", help="Wait until server is healthy")
    sub.add_parser("kill", help="Kill server on port 8000")
    sub.add_parser("health", help="Check if server is running (exit 0/1)")

    bench_p = sub.add_parser("benchmark", help="Run benchmark against running server")
    bench_p.add_argument("--model", type=str, default=None,
                         help="Model name (auto-detected from server if omitted)")

    run_p = sub.add_parser("run", help="Wait for server + benchmark + kill (full pipeline)")
    run_p.add_argument("--model", type=str, default=None,
                       help="Model name (auto-detected from server if omitted)")
    run_p.add_argument("--no-kill", action="store_true",
                       help="Don't kill the server after benchmark")

    args = parser.parse_args()

    adapter = load_adapter(read_framework_from_overview())

    if args.command == "wait":
        ok = wait_for_server(adapter)
        sys.exit(0 if ok else 1)

    elif args.command == "kill":
        kill_server()

    elif args.command == "health":
        healthy = check_server_health(adapter)
        print("healthy" if healthy else "not running")
        sys.exit(0 if healthy else 1)

    elif args.command == "benchmark":
        model = args.model or detect_model_name()
        if not model:
            print("Could not detect model name. Is the server running? Use --model to specify.")
            sys.exit(1)
        print(f"Model: {model}")
        result = run_benchmark(adapter, model)
        if result is None:
            sys.exit(1)
        print()
        print_results(result)

    elif args.command == "run":
        ok = wait_for_server(adapter)
        if not ok:
            sys.exit(1)
        model = args.model or detect_model_name()
        if not model:
            print("Could not detect model name. Use --model to specify.")
            sys.exit(1)
        print(f"Model: {model}")
        result = run_benchmark(adapter, model)
        if not args.no_kill:
            print()
            kill_server()
        if result is None:
            sys.exit(1)
        print()
        print_results(result)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests**

```bash
uv run pytest tests/ -v
```
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add project/no_edit/benchmark.py tests/test_benchmark_routing.py
git commit -m "refactor: benchmark.py uses adapter routing based on FRAMEWORK"
```

---

## Task 7: Create SGLang serve script

**Files:**
- Create: `project/edit/sglang_serve_config.sh`

- [ ] **Step 1: Create the SGLang serve config**

Create `project/edit/sglang_serve_config.sh`:

```bash
#!/bin/bash
# ==========================================================================
# SGLang Serve Configuration — THIS FILE IS EDITED BY THE AGENT
#
# This script starts the SGLang server. The agent modifies the flags below
# to optimize serving performance.
#
# Usage: bash project/edit/sglang_serve_config.sh
# ==========================================================================

# Model (set by user in memory/overview.md — do NOT change)
MODEL="Qwen/Qwen3.5-27B-FP8"

# --- SGLang serve parameters (agent edits everything below) ---

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --mem-fraction-static 0.90 \
    --chunked-prefill-size 8192
```

- [ ] **Step 2: Make it executable**

```bash
chmod +x project/edit/sglang_serve_config.sh
```

- [ ] **Step 3: Commit**

```bash
git add project/edit/sglang_serve_config.sh
git commit -m "feat: add sglang serve config script"
```

---

## Task 8: Create SGLang search strategy

**Files:**
- Create: `memory/search_strategy_sglang.md`

- [ ] **Step 1: Write the SGLang strategy file**

Create `memory/search_strategy_sglang.md`:

```markdown
# Search Strategy (SGLang)

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
```

- [ ] **Step 2: Commit**

```bash
git add memory/search_strategy_sglang.md
git commit -m "feat: add sglang search strategy for agent"
```

---

## Task 9: Update memory files to be framework-generic

**Files:**
- Modify: `memory/overview.md`
- Modify: `memory/rules.md`
- Modify: `memory/experiment_loop.md`
- Modify: `memory/setup.md`

- [ ] **Step 1: Update `memory/overview.md`**

Replace the full contents with:

```markdown
# Autooptimizer Overview

Automated LLM serving parameter optimization using LLM agents.

## Target Model

```
MODEL=Qwen/Qwen3.5-27B-FP8
FRAMEWORK=vllm
```

**Do NOT change the model.** All experiments must use this exact model. The goal is to find the best serving parameters for this specific model on this specific hardware.

**FRAMEWORK** determines which serving framework to optimize. Supported values: `vllm`, `sglang`.

## Best Known Configuration

### vLLM

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

### SGLang

```bash
python -m sglang.launch_server --model-path Qwen/Qwen3.5-27B-FP8 \
                               --mem-fraction-static 0.90 \
                               --chunked-prefill-size 8192
```

> When a better configuration is found, update the section for the active framework.

## Goal

**Get the highest `score`** — higher is better.

```
score = throughput_tok_per_sec / (1 + mean_e2el_ms / 1000)
```

This rewards high token throughput while penalizing high end-to-end response latency (total time from request to completion).

## Files

| File | Purpose | Editable |
|------|---------|----------|
| `project/edit/vllm_serve_config.sh` | vLLM serve command with parameters | **Yes** (when FRAMEWORK=vllm) |
| `project/edit/sglang_serve_config.sh` | SGLang serve command with parameters | **Yes** (when FRAMEWORK=sglang) |
| `project/no_edit/benchmark.py` | Benchmark runner, scoring, server management | No |
| `project/no_edit/adapters/` | Framework adapters (routing logic) | No |
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
```

- [ ] **Step 2: Update `memory/rules.md`**

Replace the full contents with:

```markdown
# Experimentation Rules

Each experiment starts a server, runs a fixed benchmark, then kills the server.

## Active Framework

Read `FRAMEWORK` from `memory/overview.md` to determine which framework you are optimizing. Only edit the serve config for the active framework.

| Framework | Serve Config | Strategy |
|-----------|-------------|----------|
| vllm | `project/edit/vllm_serve_config.sh` | `memory/search_strategy_vllm.md` |
| sglang | `project/edit/sglang_serve_config.sh` | `memory/search_strategy_sglang.md` |

## What You CAN Do

- Modify the active framework's serve config in `project/edit/`
- Use **ANY** flag supported by the active framework — no restrictions on parameters
- Discover available flags by running the framework's serve command with `--help`:
  - vLLM: `vllm serve --help`
  - SGLang: `python -m sglang.launch_server --help`

## What You CANNOT Do

- **Change the model** — always use the model specified in `memory/overview.md`
- **Modify `project/no_edit/benchmark.py`** or anything in `project/no_edit/` — read-only
- **Install new packages** or add dependencies
- **Change the port** — always use `--port 8000` (benchmark.py expects this)
- **Change the host** — always use `--host 127.0.0.1`
- **Edit another framework's serve config** — only edit the active framework's script

## Scoring

```
score = throughput_tok_per_sec / (1 + mean_e2el_ms / 1000)
```

Higher is better. This is the ONLY metric that matters for keep/discard decisions.

## GPU Memory

Soft constraint. The server must not OOM. Some increase in memory usage is acceptable for meaningful score gains.
```

- [ ] **Step 3: Update `memory/experiment_loop.md`**

Replace the full contents with:

```markdown
# The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autooptimizer/apr11`).

## LOOP FOREVER

1. **Read hypothesis backlog**: Open `artifacts/hypothesis_backlog.tsv` and pick the **FIRST (topmost) row** with an empty `result` column. Do NOT skip rows.

2. **Implement the hypothesis**: Modify the active framework's serve config (see `memory/rules.md` for which file to edit based on FRAMEWORK).

3. **Git commit your changes** (REQUIRED before running):
   ```bash
   git add project/edit/
   git commit -m "experiment: <short description>"
   ```
   Save the commit hash: `COMMIT=$(git rev-parse --short HEAD)`

4. **Kill any existing server**:
   ```bash
   uv run project/no_edit/benchmark.py kill
   ```

5. **Start the server** (background, logs saved):
   ```bash
   bash project/edit/<framework>_serve_config.sh > logs/$COMMIT-server.log 2>&1 &
   ```
   Use the correct serve config for the active FRAMEWORK (e.g. `vllm_serve_config.sh` or `sglang_serve_config.sh`).

6. **Wait + benchmark + kill**:
   ```bash
   uv run project/no_edit/benchmark.py run > logs/$COMMIT.log 2>&1
   ```

7. **Read results**: `grep "^score:\|^throughput_tok/s:\|^mean_ttft_ms:\|^peak_gpu_memory_mb:" logs/$COMMIT.log`

8. **If grep is empty** → run crashed. Check `tail -n 50 logs/$COMMIT-server.log` and `tail -n 50 logs/$COMMIT.log` to debug. Give up after 2-3 attempts.

9. **Record results** in `artifacts/results.tsv` (do NOT commit results.tsv)

10. **If score IMPROVED** (higher than previous best):
    - Keep the commit
    - Update your best score reference
    - Log status as `keep`

11. **If score is EQUAL or WORSE**:
    - **MUST revert**: `git reset --hard HEAD~1`
    - Log status as `discard`
    - Verify: `git log --oneline -1`

12. **Update hypothesis backlog**:
    - Fill in the `result` column for the experimented hypothesis (e.g., "keep - score: 1456.78" or "discard - score: 1100.00" or "crash")
    - Continue with the next hypothesis that has an empty `result` column
    - Do NOT commit hypothesis_backlog.tsv

13. **MAY append new hypotheses** (1-3 ideas) at end of backlog.

14. **Loop back to step 1**.

## Results TSV Format

Tab-separated, 5 columns:

```
experiment	score	memory_gb	status	description
```

- `experiment`: sequential number (1, 2, 3, ...)
- `score`: e.g. 1234.56 (use 0.00 for crashes)
- `memory_gb`: peak_gpu_memory_mb / 1024, round to .1f (use 0.0 for crashes)
- `status`: `keep`, `discard`, or `crash`
- `description`: the flags used + what changed

Example:
```
experiment	score	memory_gb	status	description
1	1234.56	44.0	keep	baseline (no extra flags)
2	1456.78	44.2	keep	--max-num-seqs 512
3	0.00	0.0	crash	--gpu-memory-utilization 0.99 (OOM)
4	1800.00	42.0	keep	--quantization fp8 --max-num-seqs 512
```

## Important Notes

**Timeout**: If a server doesn't come up within 5 minutes, or the benchmark hangs, kill it and treat as failure.

**Crashes**: If easy fix (typo, invalid flag), fix and re-run. If fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Continue *indefinitely* until manually stopped. If out of ideas, think harder — run the framework's `--help`, re-read files, combine near-misses, try radical changes.
```

- [ ] **Step 4: Update `memory/setup.md`**

Replace the full contents with:

```markdown
# Experiment Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr11`). The branch `autooptimizer/<tag>` must not already exist — this is a fresh run.

2. **Create the branch**: `git checkout -b autooptimizer/<tag>` from current master.

3. **Read the memory md files for context**:
   - `memory/overview.md` — target model, framework, goal, best known config
   - `memory/experiment_loop.md` — experiment workflow
   - `memory/rules.md` — allowed and prohibited changes
   - Read the search strategy for the active FRAMEWORK:
     - vLLM: `memory/search_strategy_vllm.md`
     - SGLang: `memory/search_strategy_sglang.md`

   Also examine the fixed benchmark tool:
   - `project/no_edit/benchmark.py` — benchmark runner, scoring (read-only)

   And the editable config for the active FRAMEWORK:
   - vLLM: `project/edit/vllm_serve_config.sh`
   - SGLang: `project/edit/sglang_serve_config.sh`

4. **Check GPU**: Run `nvidia-smi` to see what GPU(s) you have and how much VRAM is available.

5. **Check framework flags**: Run the active framework's serve command with `--help` to see ALL available parameters:
   - vLLM: `vllm serve --help`
   - SGLang: `python -m sglang.launch_server --help`

6. **Initialize artifacts and logs**:
   - Create `logs/` folder: `mkdir -p logs`
   - Create `artifacts/results.tsv` with just the header row (if not exists).

7. **Initialize hypothesis_backlog.tsv**: If not exists, create `artifacts/hypothesis_backlog.tsv` with the header row:
   ```
   hypothesis	uplift_expectation	effort	priority	rationale	result
   ```

8. **Populate hypothesis backlog with high-impact experiments**:
   - Run the framework's `--help` to discover all available parameters
   - Examine the active framework's serve config to see the baseline
   - Check `memory/overview.md` for the best known configuration
   - Read the active framework's search strategy file
   - Identify high-impact parameter changes (quantization, batching, memory, scheduling)
   - Suggest **bold changes first** (quantization, speculative decoding, large batch sizes)
   - Add **quick wins with high priority** (simple flag toggles)
   - Order by priority: `uplift_expectation - 0.5 * effort`
   - Include clear rationale for each hypothesis

9. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.
```

- [ ] **Step 5: Commit**

```bash
git add memory/overview.md memory/rules.md memory/experiment_loop.md memory/setup.md
git commit -m "docs: update memory files for multi-framework support"
```

---

## Task 10: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README.md**

Replace the full contents with:

```markdown
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
├── artifacts/
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
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README for multi-framework support"
```

---

## Deferred: Unified HTTP Benchmark (`--unified` flag)

The spec describes a framework-agnostic HTTP load generator invoked via `--unified` that sends requests to `/v1/completions` and measures TTFT/TPOT/ITL/E2EL from streaming responses. This is a self-contained feature that doesn't block the core multi-framework support. It should be implemented as a follow-up once the native-per-framework benchmarks are working. The README roadmap lists it as "Planned."

---

## Task 11: Run full test suite and final verification

- [ ] **Step 1: Run all tests**

```bash
uv run pytest tests/ -v
```
Expected: All tests PASS.

- [ ] **Step 2: Verify benchmark.py help works**

```bash
uv run project/no_edit/benchmark.py --help
```
Expected: Help text prints without errors.

- [ ] **Step 3: Verify framework detection**

Check that `benchmark.py` reads the framework correctly by running a command that triggers adapter loading (health check won't connect but should show the right error):

```bash
uv run project/no_edit/benchmark.py health
```
Expected: Prints "not running" (which means it loaded the adapter and tried to health-check — framework routing works).

- [ ] **Step 4: Final commit if any fixes were needed**

```bash
git status
# If clean, nothing to commit. If fixes were made:
git add -A
git commit -m "fix: address issues found during final verification"
```
