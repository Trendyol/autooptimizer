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
