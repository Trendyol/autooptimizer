#!/bin/bash
# ==========================================================================
# vLLM Serve Configuration — THIS FILE IS EDITED BY THE AGENT
#
# This script starts the vLLM server. The agent modifies the flags below
# to optimize serving performance.
#
# Usage: bash project/edit/vllm_serve_config.sh
# ==========================================================================

# Model (set by user in memory/overview.md — do NOT change)
MODEL="Qwen/Qwen3.5-27B-FP8"

# --- vLLM serve parameters (agent edits everything below) ---
# Baseline: best-known config from memory/overview.md (+ required host/port)

vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --speculative-config '{""method"":""mtp"",""num_speculative_tokens"":1}' \
    --enable-chunked-prefill \
    --max-num-batched-tokens 16384 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90
