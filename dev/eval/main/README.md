# dev/eval/main — paper §sec:eval-main runners

Per-cell scripts produce one `bench.json` (compatible with `aggregate.py`).
Run the same cell with different `INTRA` / `INTER` flags to fill the
4-cell ablation; loop trials externally to get $n{=}5$.

The 8-GPU H200 host is assumed; passwordless `sudo` is required for the
pre-boot port + GPU cleanup. Run dirs default to `dev/eval/runs/<name>/`.

## Single SGLang cell

```bash
# regime ∈ {m1, m2, m3, cc_traj}; cell = (INTRA, INTER) ∈ {0,1}²
MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=0 \
    INTRA=0 INTER=0 \
    PORT=33000 OUT_DIR=dev/eval/runs/q35-m1-stock \
    bash dev/eval/main/run_m1.sh
```

Swap the runner for the regime: `run_m2.sh`, `run_m3.sh`, `run_cc_traj.sh`.

`cc_traj` defaults to `dev/eval/datasets/cc_long_traces.jsonl`
(106 unique ≥100K-token public Claude Code sessions); override with
`TRACES_FILE=`.

## Single vLLM cell

```bash
# regime ∈ {m1, m2, m3} via run_vllm.sh; cc_traj via run_cc_traj_vllm.sh
MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=0 REGIME=m1 \
    PORT=33000 OUT_DIR=dev/eval/runs/q35-m1-vllm \
    bash dev/eval/main/run_vllm.sh

MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=0 \
    PORT=33000 OUT_DIR=dev/eval/runs/q35-cc-vllm \
    bash dev/eval/main/run_cc_traj_vllm.sh
```

Per-model native TP: Qwen3.5 = 1, Qwen3-Next 80B = 2 (single H200 too small),
Kimi 48B = 1 in SGLang / **2 in vLLM** (vLLM v0.20 OOMs at TP=1 with default
`--gpu-memory-utilization 0.85`).

## Full vLLM cross-engine baseline (3 models × 3 regimes × n trials)

```bash
bash dev/eval/main/vllm_baseline.sh        # n=5 default
bash dev/eval/main/vllm_baseline.sh 1      # n=1 sanity first
```

Three sequential phases, each filling all 8 GPUs at the model's native TP
(Kimi/Qwen3-Next at TP=2 → 4 workers on GPU pairs; Qwen3.5 at TP=1 → 8
workers on single GPUs). Per-cell logs validate `bench.json` (catches
0-reqs invalid runs that get `rc=0`); main log + `_summary.tsv` in run dir.

## Static-best sweep

```bash
for r in 0.3 0.5 0.7 0.9; do
    MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=0 \
        INTRA=0 INTER=0 \
        PORT=33000 OUT_DIR=dev/eval/runs/q35-m1-sb-r${r} \
        EXTRA_LAUNCH_FLAGS="--mamba-full-memory-ratio $r" \
        CELL_LABEL_OVERRIDE="static_best_r${r}" \
        bash dev/eval/main/run_m1.sh
done
```

Pick the best ratio per regime; rerun $n{=}5$ at the chosen ratio for
mean ± std.

## Aggregate

```bash
python3 dev/eval/main/aggregate.py dev/eval/runs/<run_dir>
```

Reads every `bench.json` under `<run_dir>/<model>/<regime>/<cell>/`, emits
`main_table.csv` mean ± std per cell.

## Files

- `_common.sh` — `apply_cell_env`, `boot_sglang`, `cleanup_before_boot`
- `run_m1.sh` / `run_m2.sh` / `run_m3.sh` — single SGLang cell, one regime each
- `run_cc_traj.sh` — single SGLang Claude Code trajectory replay cell
- `run_vllm.sh` — single vLLM cell (m1 / m2 / m3 via `REGIME` env)
- `run_cc_traj_vllm.sh` — single vLLM Claude Code trajectory replay cell
- `cc_trace_replay.py` — async OpenAI-compatible replay client
- `vllm_baseline.sh` — orchestrates the full vLLM cross-engine baseline
- `aggregate.py` — bench.json → CSV
