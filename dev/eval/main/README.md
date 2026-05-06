# dev/eval/main — runners for paper §sec:eval-main

Each section below maps to one paper table. Run dirs default to
`dev/eval/runs/<name>/`. Passwordless `sudo` required (pre-boot port +
GPU cleanup).

---

## Table 1 — `tab:main-cross-model` (3 models × 3 regimes × 4 baselines)

Cells: SGLang stock, SGLang static-best, vLLM, **\sys{}** (full system $L_\text{intra}{=}1, L_\text{inter}{=}1$).

### SGLang stock (cell `intra=0, inter=0`) — n=5

```bash
# Per regime: regime ∈ {m1, m2, m3} → run_m{1,2,3}.sh
for trial in 1 2 3 4 5; do
    MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=0 \
        INTRA=0 INTER=0 \
        PORT=33000 OUT_DIR=dev/eval/runs/q35-m1-stock-trial${trial} \
        bash dev/eval/main/run_m1.sh
done
```

Per-model native TP: Qwen3.5 = 1, Qwen3-Next 80B = 2, Kimi 48B = 1.

### SGLang static-best (4-ratio sweep + n=5 at the winner)

```bash
# 1) Sweep ratios (single trial per ratio)
for r in 0.3 0.5 0.7 0.9; do
    MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=0 \
        INTRA=0 INTER=0 \
        PORT=33000 OUT_DIR=dev/eval/runs/q35-m1-sb-r${r} \
        EXTRA_LAUNCH_FLAGS="--mamba-full-memory-ratio $r" \
        CELL_LABEL_OVERRIDE="static_best_r${r}" \
        bash dev/eval/main/run_m1.sh
done
# 2) Pick the best ratio per regime; rerun n=5 at that ratio
```

### vLLM cross-engine baseline (full Table 1 column, all 3 models × 3 regimes × n=5)

```bash
bash dev/eval/main/vllm_baseline.sh        # n=5 default
bash dev/eval/main/vllm_baseline.sh 1      # n=1 sanity first
```

vLLM TP convention: Qwen3.5 = 1, Qwen3-Next = 2, Kimi = **2** (vLLM v0.20
OOMs at TP=1 with default `--gpu-memory-utilization 0.85`).

### \sys{} cell `intra=1, inter=1` — n=5

```bash
# Same regime-runner as stock; toggle the cell flags
for trial in 1 2 3 4 5; do
    MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=0 \
        INTRA=1 INTER=1 \
        PORT=33000 OUT_DIR=dev/eval/runs/q35-m1-fulcrum-trial${trial} \
        bash dev/eval/main/run_m1.sh
done
```

### Per-regime 4-cell ablation (`tab:multiturn-headline`, `tab:swarm-headline`, `tab:phase-shift-headline`)

Same runner, all four `(INTRA, INTER) ∈ {0,1}²` combinations at n=5
trials each on Qwen3.5-35B-A3B:

```bash
for cell in "0 0" "1 0" "0 1" "1 1"; do
    read intra inter <<< "$cell"
    for trial in 1 2 3 4 5; do
        MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=0 \
            INTRA=$intra INTER=$inter \
            PORT=33000 OUT_DIR=dev/eval/runs/q35-m1-i${intra}j${inter}-t${trial} \
            bash dev/eval/main/run_m1.sh
    done
done
```

---

## Table 2 — `tab:real-workload` (SWE-Bench-Pro + Claude Code trajectory replay)

(i) SWE-Bench-Pro is run externally through Harbor + Claude Code; not in
this dir. (ii) Claude Code trajectory replay is what the runners below
produce.

### SGLang Claude Code trajectory replay — n=5 per model

Default trace dataset: `dev/eval/datasets/cc_long_traces.jsonl` (n=106 unique
≥100K-token public Claude Code sessions). Override with `TRACES_FILE=`.
For \sys{} flip `INTRA=1 INTER=1` on any of the commands below.

Qwen3.5-35B-A3B (TP=1, 5 GPUs in parallel, one per trial):

```bash
for trial in 1 2 3 4 5; do
    MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=$((trial - 1)) \
        INTRA=0 INTER=0 \
        PORT=$((33000 + trial)) \
        OUT_DIR=dev/eval/runs/q35-cc-stock/trial${trial} \
        bash dev/eval/main/run_cc_traj.sh &
done
wait
```

Qwen3-Next 80B-A3B (TP=2, 4 GPU pairs in parallel + trial 5 sequential after one pair frees):

```bash
for trial in 1 2 3 4; do
    pair=$(( (trial - 1) * 2 )); pair_str="${pair},$((pair + 1))"
    MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct TP=2 GPU_LIST="$pair_str" \
        INTRA=0 INTER=0 \
        PORT=$((33100 + trial)) \
        OUT_DIR=dev/eval/runs/q3n-cc-stock/trial${trial} \
        bash dev/eval/main/run_cc_traj.sh &
done
wait -n; sleep 30
MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct TP=2 GPU_LIST="0,1" \
    INTRA=0 INTER=0 \
    PORT=33105 OUT_DIR=dev/eval/runs/q3n-cc-stock/trial5 \
    bash dev/eval/main/run_cc_traj.sh
wait
```

Kimi-Linear 48B-A3B — same 4-parallel + 1-sequential layout as Qwen3-Next,
TP=2 (TP=1 KV-cache headroom too tight for 14 conc × ≥100K context):

```bash
# Same shape as Qwen3-Next above; swap MODEL=moonshotai/Kimi-Linear-48B-A3B-Instruct
```

### vLLM Claude Code trajectory replay — n=5 per model

vLLM TP convention: Qwen3.5 = 1, Qwen3-Next = 2, Kimi = **2** (vLLM v0.20
OOMs at TP=1 with default `--gpu-memory-utilization 0.85`).

Qwen3.5-35B-A3B (TP=1, 5 GPUs in parallel):

```bash
for trial in 1 2 3 4 5; do
    MODEL=Qwen/Qwen3.5-35B-A3B TP=1 GPU_LIST=$((trial - 1)) \
        PORT=$((34100 + trial)) \
        OUT_DIR=dev/eval/runs/q35-cc-vllm/trial${trial} \
        bash dev/eval/main/run_cc_traj_vllm.sh &
done
wait
```

Qwen3-Next 80B-A3B / Kimi-Linear 48B-A3B (TP=2, 4 pairs parallel + 1 sequential):

```bash
for trial in 1 2 3 4; do
    pair=$(( (trial - 1) * 2 )); pair_str="${pair},$((pair + 1))"
    MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct TP=2 GPU_LIST="$pair_str" \
        PORT=$((34200 + trial)) \
        OUT_DIR=dev/eval/runs/q3n-cc-vllm/trial${trial} \
        bash dev/eval/main/run_cc_traj_vllm.sh &
done
wait -n; sleep 30
MODEL=Qwen/Qwen3-Next-80B-A3B-Instruct TP=2 GPU_LIST="0,1" \
    PORT=34205 OUT_DIR=dev/eval/runs/q3n-cc-vllm/trial5 \
    bash dev/eval/main/run_cc_traj_vllm.sh
wait

# Kimi: same shape, swap MODEL=moonshotai/Kimi-Linear-48B-A3B-Instruct
```

---

## Aggregate

```bash
python3 dev/eval/main/aggregate.py dev/eval/runs/<run_dir>
```

Reads every `bench.json` under `<run_dir>/<model>/<regime>/<cell>/`,
emits `main_table.csv` mean ± std per cell.
