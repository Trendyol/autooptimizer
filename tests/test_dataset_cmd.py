import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "project" / "no_edit"))

from adapters.sglang import get_native_benchmark_cmd as get_sglang_cmd
from adapters.vllm import get_native_benchmark_cmd as get_vllm_cmd


def _value_after(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def _vllm_cmd(dataset_name: str = "random", dataset_path: str | None = None) -> list[str]:
    return get_vllm_cmd(
        model="test-model",
        base_url="http://127.0.0.1:8000",
        num_prompts=500,
        input_len=512,
        output_len=128,
        seed=42,
        request_rate="inf",
        result_filename="bench_result.json",
        dataset_name=dataset_name,
        dataset_path=dataset_path,
    )


def _sglang_cmd(dataset_name: str = "random", dataset_path: str | None = None) -> list[str]:
    return get_sglang_cmd(
        model="test-model",
        base_url="http://127.0.0.1:8000",
        num_prompts=500,
        input_len=512,
        output_len=128,
        seed=42,
        request_rate="inf",
        result_filename="sglang_bench_result.json",
        dataset_name=dataset_name,
        dataset_path=dataset_path,
    )


def test_vllm_random_dataset_flags():
    cmd = _vllm_cmd(dataset_name="random")

    assert _value_after(cmd, "--input-len") == "512"
    assert _value_after(cmd, "--output-len") == "128"
    assert _value_after(cmd, "--random-range-ratio") == "0.0"
    assert "--disable-shuffle" in cmd


def test_vllm_custom_dataset_uses_custom_output_len():
    cmd = _vllm_cmd(dataset_name="custom")

    assert _value_after(cmd, "--custom-output-len") == "128"
    assert "--input-len" not in cmd


def test_vllm_dataset_path_flag():
    cmd = _vllm_cmd(dataset_path="/d.jsonl")

    assert _value_after(cmd, "--dataset-path") == "/d.jsonl"


def test_sglang_random_dataset_flags():
    cmd = _sglang_cmd(dataset_name="random")

    assert _value_after(cmd, "--random-input-len") == "512"
    assert _value_after(cmd, "--random-output-len") == "128"
    assert _value_after(cmd, "--random-range-ratio") == "0.0"
    assert "--disable-shuffle" in cmd


def test_sglang_sharegpt_dataset_flags():
    cmd = _sglang_cmd(dataset_name="sharegpt")

    assert _value_after(cmd, "--sharegpt-output-len") == "128"


def test_sglang_custom_dataset_adds_no_extra_length_flags():
    cmd = _sglang_cmd(dataset_name="custom")

    assert "--random-input-len" not in cmd
    assert "--random-output-len" not in cmd
    assert "--sharegpt-output-len" not in cmd


def test_sglang_dataset_path_flag():
    cmd = _sglang_cmd(dataset_path="/d.jsonl")

    assert _value_after(cmd, "--dataset-path") == "/d.jsonl"
