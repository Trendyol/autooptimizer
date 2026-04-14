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
