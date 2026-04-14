import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "project" / "no_edit"))

from adapters import read_framework_from_overview


class TestBenchmarkFrameworkRouting:
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
