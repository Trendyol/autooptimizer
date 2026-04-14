# Experiment Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr11`). The branch `autooptimizer/<tag>` must not already exist — this is a fresh run.

2. **Create the branch**: `git checkout -b autooptimizer/<tag>` from current master.

3. **Read the memory md files for context**:
   - `memory/overview.md` — target model, goal, best known config
   - `memory/experiment_loop.md` — experiment workflow
   - `memory/rules.md` — allowed and prohibited changes
   - `memory/search_strategy.md` — hypothesis strategy

   Also examine the fixed benchmark tool:
   - `project/no_edit/benchmark.py` — benchmark runner, scoring (read-only)

   And the editable config:
   - `project/edit/serve_config.sh` — the vLLM serve command you modify

4. **Check GPU**: Run `nvidia-smi` to see what GPU(s) you have and how much VRAM is available.

5. **Check vLLM flags**: Run `vllm serve --help` to see ALL available parameters. This is your playground.

6. **Initialize artifacts and logs**:
   - Create `logs/` folder: `mkdir -p logs`
   - Create `artifacts/results.tsv` with just the header row (if not exists).

7. **Initialize hypothesis_backlog.tsv**: If not exists, create `artifacts/hypothesis_backlog.tsv` with the header row:
   ```
   hypothesis	uplift_expectation	effort	priority	rationale	result
   ```

8. **Populate hypothesis backlog with high-impact experiments**:
   - Run `vllm serve --help` to discover all available parameters
   - Examine the current `project/edit/serve_config.sh` to see the baseline config
   - Check `memory/overview.md` for the best known configuration
   - Identify high-impact parameter changes (quantization, batching, memory, scheduling)
   - Suggest **bold changes first** (quantization, speculative decoding, large batch sizes)
   - Add **quick wins with high priority** (simple flag toggles)
   - Order by priority: `uplift_expectation - 0.5 * effort`
   - Include clear rationale for each hypothesis

9. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.
