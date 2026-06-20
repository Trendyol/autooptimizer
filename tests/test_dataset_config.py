import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "project" / "no_edit"))

from adapters import read_dataset_from_overview


def test_defaults_to_random_dataset_when_missing(tmp_path):
    overview = tmp_path / "overview.md"
    overview.write_text("```\nMODEL=test\nFRAMEWORK=vllm\n```\n")

    assert read_dataset_from_overview(str(overview)) == ("random", None)


def test_reads_custom_dataset_path(tmp_path):
    overview = tmp_path / "overview.md"
    overview.write_text(
        "```\n"
        "MODEL=test\n"
        "FRAMEWORK=vllm\n"
        "DATASET=custom\n"
        "DATASET_PATH=/data/x.jsonl\n"
        "```\n"
    )

    assert read_dataset_from_overview(str(overview)) == ("custom", "/data/x.jsonl")


def test_empty_dataset_path_is_none(tmp_path):
    overview = tmp_path / "overview.md"
    overview.write_text(
        "```\n"
        "MODEL=test\n"
        "FRAMEWORK=sglang\n"
        "DATASET=sharegpt\n"
        "DATASET_PATH=\n"
        "```\n"
    )

    assert read_dataset_from_overview(str(overview)) == ("sharegpt", None)


def test_dataset_name_is_lowercased(tmp_path):
    overview = tmp_path / "overview.md"
    overview.write_text("```\nMODEL=test\nFRAMEWORK=vllm\nDATASET=Custom\n```\n")

    assert read_dataset_from_overview(str(overview)) == ("custom", None)
