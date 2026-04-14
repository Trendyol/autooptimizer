# The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autooptimizer/apr11`).

## LOOP FOREVER

1. **Read hypothesis backlog**: Open `artifacts/hypothesis_backlog.tsv` and pick the **FIRST (topmost) row** with an empty `result` column. Do NOT skip rows.

2. **Implement the hypothesis**: Modify `project/edit/serve_config.sh` with the new vLLM parameters.

3. **Git commit your changes** (REQUIRED before running):
   ```bash
   git add project/edit/serve_config.sh
   git commit -m "experiment: <short description>"
   ```
   Save the commit hash: `COMMIT=$(git rev-parse --short HEAD)`

4. **Kill any existing server**:
   ```bash
   uv run project/no_edit/benchmark.py kill
   ```

5. **Start the server** (background, logs saved):
   ```bash
   bash project/edit/serve_config.sh > logs/$COMMIT-server.log 2>&1 &
   ```

6. **Wait + benchmark + kill**:
   ```bash
   uv run project/no_edit/benchmark.py run > logs/$COMMIT.log 2>&1
   ```

7. **Read results**: `grep "^score:\|^throughput_tok/s:\|^mean_ttft_ms:\|^peak_gpu_memory_mb:" logs/$COMMIT.log`

8. **If grep is empty** → run crashed. Check `tail -n 50 logs/$COMMIT-server.log` and `tail -n 50 logs/$COMMIT.log` to debug. Give up after 2-3 attempts.

9. **Record results** in `artifacts/results.tsv` (do NOT commit results.tsv)

10. **If score IMPROVED** (higher than previous best):
    - Keep the commit
    - Update your best score reference
    - Log status as `keep`

11. **If score is EQUAL or WORSE**:
    - **MUST revert**: `git reset --hard HEAD~1`
    - Log status as `discard`
    - Verify: `git log --oneline -1`

12. **Update hypothesis backlog**:
    - Fill in the `result` column for the experimented hypothesis (e.g., "keep - score: 1456.78" or "discard - score: 1100.00" or "crash")
    - Continue with the next hypothesis that has an empty `result` column
    - Do NOT commit hypothesis_backlog.tsv

13. **MAY append new hypotheses** (1-3 ideas) at end of backlog.

14. **Loop back to step 1**.

## Results TSV Format

Tab-separated, 5 columns:

```
experiment	score	memory_gb	status	description
```

- `experiment`: sequential number (1, 2, 3, ...)
- `score`: e.g. 1234.56 (use 0.00 for crashes)
- `memory_gb`: peak_gpu_memory_mb / 1024, round to .1f (use 0.0 for crashes)
- `status`: `keep`, `discard`, or `crash`
- `description`: the vLLM flags used + what changed

Example:
```
experiment	score	memory_gb	status	description
1	1234.56	44.0	keep	baseline (no extra flags)
2	1456.78	44.2	keep	--max-num-seqs 512
3	0.00	0.0	crash	--gpu-memory-utilization 0.99 (OOM)
4	1800.00	42.0	keep	--quantization fp8 --max-num-seqs 512
```

## Important Notes

**Timeout**: If a server doesn't come up within 5 minutes, or the benchmark hangs, kill it and treat as failure.

**Crashes**: If easy fix (typo, invalid flag), fix and re-run. If fundamentally broken, log "crash" and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. Continue *indefinitely* until manually stopped. If out of ideas, think harder — run `vllm serve --help`, re-read files, combine near-misses, try radical changes.
