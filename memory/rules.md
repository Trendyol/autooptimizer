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
