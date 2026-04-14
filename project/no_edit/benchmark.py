"""
Autooptimizer benchmark tool.
Waits for a running vLLM server, runs benchmarks, reports results.

DO NOT MODIFY THIS FILE.

Commands:
    python benchmark.py wait                — wait until vLLM server is healthy
    python benchmark.py benchmark           — run benchmark against running server
    python benchmark.py benchmark --model X — run benchmark with explicit model name
    python benchmark.py kill                — kill vLLM server on port 8000
    python benchmark.py run                 — wait + benchmark + kill (full pipeline)
    python benchmark.py health              — check if server is running (exit 0/1)
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
from dataclasses import dataclass
from typing import Optional

import requests as http_requests

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

SERVER_PORT = 8000
SERVER_HOST = "127.0.0.1"
HEALTH_ENDPOINT = f"http://{SERVER_HOST}:{SERVER_PORT}/health"
SERVER_STARTUP_TIMEOUT = 300

BENCHMARK_TIMEOUT = 240
NUM_PROMPTS = 500
BENCHMARK_DATASET = "random"
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

def check_server_health() -> bool:
    try:
        resp = http_requests.get(HEALTH_ENDPOINT, timeout=5)
        return resp.status_code == 200
    except (http_requests.ConnectionError, http_requests.Timeout):
        return False


def wait_for_server(timeout: int = SERVER_STARTUP_TIMEOUT) -> bool:
    print(f"Waiting for vLLM server at {HEALTH_ENDPOINT} ...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if check_server_health():
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
        resp = http_requests.get(f"http://{SERVER_HOST}:{SERVER_PORT}/v1/models", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id")
    except Exception:
        pass
    return None


def run_benchmark(model_name: str) -> Optional[BenchmarkResult]:
    if os.path.exists(RESULT_FILE):
        os.remove(RESULT_FILE)

    bench_cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main",
        "bench", "serve",
        "--backend", "openai",
        "--base-url", f"http://{SERVER_HOST}:{SERVER_PORT}",
        "--model", model_name,
        "--dataset-name", BENCHMARK_DATASET,
        "--num-prompts", str(NUM_PROMPTS),
        "--input-len", str(BENCHMARK_INPUT_LEN),
        "--output-len", str(BENCHMARK_OUTPUT_LEN),
        "--seed", str(BENCHMARK_SEED),
        "--random-range-ratio", "0.0",
        "--disable-shuffle",
        "--request-rate", REQUEST_RATE,
        "--save-result",
        "--result-dir", ".",
        "--result-filename", RESULT_FILE,
        "--percentile-metrics", "ttft,tpot,itl,e2el",
        "--metric-percentiles", "50,99",
    ]

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

    return parse_result(duration)


def parse_result(duration: float) -> Optional[BenchmarkResult]:
    if not os.path.exists(RESULT_FILE):
        print(f"Result file {RESULT_FILE} not found")
        return None

    try:
        with open(RESULT_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Failed to parse result: {e}")
        return None

    br = BenchmarkResult()
    br.benchmark_duration_sec = data.get("duration", duration)
    br.completed_requests = data.get("completed", 0)
    br.total_input_tokens = data.get("total_input_tokens", 0)
    br.total_output_tokens = data.get("total_output_tokens", 0)
    br.throughput_req_per_sec = data.get("request_throughput", 0.0)
    br.throughput_tok_per_sec = data.get("output_throughput", 0.0)
    br.mean_ttft_ms = data.get("mean_ttft_ms", 0.0)
    br.median_ttft_ms = data.get("median_ttft_ms", 0.0)
    br.p99_ttft_ms = data.get("p99_ttft_ms", 0.0)
    br.mean_tpot_ms = data.get("mean_tpot_ms", 0.0)
    br.median_tpot_ms = data.get("median_tpot_ms", 0.0)
    br.p99_tpot_ms = data.get("p99_tpot_ms", 0.0)
    br.mean_itl_ms = data.get("mean_itl_ms", 0.0)
    br.median_itl_ms = data.get("median_itl_ms", 0.0)
    br.p99_itl_ms = data.get("p99_itl_ms", 0.0)
    br.mean_e2el_ms = data.get("mean_e2el_ms", 0.0)
    br.median_e2el_ms = data.get("median_e2el_ms", 0.0)
    br.p99_e2el_ms = data.get("p99_e2el_ms", 0.0)
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
        description="autooptimizer — benchmark tool for vLLM serving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start vLLM server in background, then:
  python benchmark.py wait
  python benchmark.py benchmark
  python benchmark.py kill

  # Or all-in-one:
  python benchmark.py run
        """,
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("wait", help="Wait until vLLM server is healthy")
    sub.add_parser("kill", help="Kill vLLM server on port 8000")
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

    if args.command == "wait":
        ok = wait_for_server()
        sys.exit(0 if ok else 1)

    elif args.command == "kill":
        kill_server()

    elif args.command == "health":
        healthy = check_server_health()
        print("healthy" if healthy else "not running")
        sys.exit(0 if healthy else 1)

    elif args.command == "benchmark":
        model = args.model or detect_model_name()
        if not model:
            print("Could not detect model name. Is the server running? Use --model to specify.")
            sys.exit(1)
        print(f"Model: {model}")
        result = run_benchmark(model)
        if result is None:
            sys.exit(1)
        print()
        print_results(result)

    elif args.command == "run":
        ok = wait_for_server()
        if not ok:
            sys.exit(1)
        model = args.model or detect_model_name()
        if not model:
            print("Could not detect model name. Use --model to specify.")
            sys.exit(1)
        print(f"Model: {model}")
        result = run_benchmark(model)
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
