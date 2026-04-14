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
