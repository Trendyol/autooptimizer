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
    dataset_name: str = "random",
    dataset_path: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "sglang.bench_serving",
        "--backend", "openai",
        "--base-url", base_url,
        "--model", model,
        "--dataset-name", dataset_name,
        "--num-prompts", str(num_prompts),
        "--seed", str(seed),
        "--request-rate", str(request_rate),
        "--save-result",
        "--result-filename", result_filename,
    ]

    if dataset_name == "random":
        cmd += [
            "--random-input-len", str(input_len),
            "--random-output-len", str(output_len),
            "--random-range-ratio", "0.0",
            "--disable-shuffle",
        ]
    elif dataset_name == "sharegpt":
        cmd += ["--sharegpt-output-len", str(output_len)]

    if dataset_path:
        cmd += ["--dataset-path", dataset_path]

    return cmd


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
