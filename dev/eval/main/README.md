# Main experiments — paper §sec:eval-main

Scripts that produce the data behind `tab:main-cross-model` and the per-regime
4-cell ablation tables (`tab:multiturn-headline`, `tab:swarm-headline`,
`tab:headline-v9`).

## What goes into the paper

| Paper artifact | Cells produced here |
|---|---|
| `tab:main-cross-model` | stock SGLang, SGLang static-best, vLLM, Fulcrum (1,1) — for {Qwen3.5-35B-A3B, Qwen3-Next 80B-A3B, Kimi Linear} × {M1, M2, M3} |
| `tab:multiturn-headline` (M1 ablation) | 4-cell × 5-trial on Qwen3.5-35B-A3B |
| `tab:swarm-headline` (M2 ablation) | 4-cell × 5-trial on Qwen3.5-35B-A3B |
| `tab:headline-v9` (M3 ablation) | 4-cell × 3-trial on Qwen3.5-35B-A3B |
| `tab:static-best` | 4-ratio sweep × Qwen3.5-35B-A3B × M3 phases |

Workload definitions are in paper §sec:eval-setup. Cell ordering is
`(L_intra, L_inter) ∈ {0,1}^2`.

## Layout

```
dev/eval/main/
├── README.md                    # this file
├── run_m1.sh                    # one cell of M1 long-horizon
├── run_m2.sh                    # one cell of M2 swarm
├── run_m3.sh                    # one cell of M3 phase-shift trace
├── run_static_best.sh           # mamba_full_memory_ratio sweep (Qwen3.5)
├── run_vllm.sh                  # vLLM v0.20.0 baseline cell
├── orchestrator.sh              # schedules everything across 8× H200
├── aggregate.py                 # walks runs/, writes CSV in paper format
└── runs/                        # output (created by run scripts)
    └── <run-name>/
        └── <model>/
            └── <regime>/
                └── trial<N>_intra<I>_inter<J>/
                    ├── server.log
                    ├── bench.json
                    ├── budgeter.jsonl    # only when L_inter=1
                    └── client.log
```

## Per-cell scripts (`run_m1.sh` / `run_m2.sh` / `run_m3.sh` / `run_static_best.sh` / `run_vllm.sh`)

Required env:

| var | meaning |
|---|---|
| `MODEL` | HuggingFace id, e.g. `Qwen/Qwen3.5-35B-A3B` |
| `TP` | tensor-parallel size (1 for ≤80B on H200, 2 for Qwen3-Next 80B) |
| `GPU_LIST` | csv like `0` or `0,1` — must match `TP` count |
| `INTRA` | 0 or 1 — paper L_intra (HPB-LRU) |
| `INTER` | 0 or 1 — paper L_inter (cross-pool transfer) |
| `PORT` | listening port |
| `OUT_DIR` | absolute path; output written here |
| `MEM_FRAC` | optional, defaults to 0.8 (0.7 on inter-on cells) |

Each script:

1. Boots one SGLang (or vLLM) server with the right env flags
2. Drives the workload-specific client (long-horizon / swarm / phase-shift)
3. Tears down the server
4. Leaves `bench.json` + `server.log` (+ `budgeter.jsonl` if `INTER=1`) in `OUT_DIR`

## Top-level orchestrator (`orchestrator.sh`)

```bash
bash dev/eval/main/orchestrator.sh \
    [SMOKE=1] \                          # 1-trial sanity check (default 0 = full)
    [MODELS="qwen3.5"] \                 # subset; default = all 3
    [REGIMES="m1 m2 m3"] \               # subset; default = all 3
    [N_TRIALS=5] \                       # M1/M2 trial count; M3 is forced to 3
    [INCLUDE_VLLM=1] \                   # default 1
    [INCLUDE_STATIC_BEST=1] \            # default 1 (only Qwen3.5)
    [PORT_BASE=33000]
```

Schedules `(model × regime × cell × trial)` across the 8× H200 respecting
TP requirements:

- Qwen3.5-35B-A3B (TP=1): 4 cells × 1 GPU each → 4-way parallel per regime
- Qwen3-Next 80B-A3B (TP=2): 4 cells × 2 GPUs each → all 8 GPUs busy
- Kimi Linear (TP=1 unless oversize): 4-way parallel like Qwen3.5

Within a model run all four cells are kicked off in parallel; cells of
different models run sequentially (one model occupies all 8 GPUs at most).

## Sanity check

```bash
SMOKE=1 MODELS=qwen3.5 N_TRIALS=1 bash dev/eval/main/orchestrator.sh
```

This runs:
- 1 trial × 4 cells × 3 regimes = 12 server bringups on Qwen3.5-35B-A3B
- ~10 minutes total (each cell ~50s warmup + ~60s bench)
- Verifies wiring; does NOT produce paper-quality numbers

## Full main-table run

```bash
N_TRIALS=5 bash dev/eval/main/orchestrator.sh
```

Estimated wall-clock with 8 H200s saturated:
- Qwen3.5-35B-A3B: ~3-4h
- Qwen3-Next 80B-A3B: ~3-4h (TP=2, fewer parallel cells)
- Kimi Linear: ~3-4h
- Total: ~10-12h. First Qwen3-Next run pays a one-time ~30min model
  download.

## Aggregating

```bash
python3 dev/eval/main/aggregate.py dev/eval/runs/<run-name>/ \
    > dev/eval/runs/<run-name>/main_table.csv
```

Output schema matches paper `tab:main-cross-model` (out-TPS, P99 TTFT,
mean TTFT, mean E2E, requests-completed, transfers-with-movement).

## vLLM venv

vLLM v0.20.0 is installed at `/data/yuzhou/projects/vllm/.venv/`. The
`run_vllm.sh` script picks this up automatically. Engine command and
benchmark client mirror SGLang's where possible; cells that are
non-applicable (vLLM has no L_intra/L_inter) are recorded under a
single `intra0_inter0_vllm` cell label.

## Cell-name conventions

- `intra0_inter0` — stock engine
- `intra1_inter0` — HPB cache only (paper §sec:design-l1)
- `intra0_inter1` — cross-pool transfer only (paper §sec:design-l2)
- `intra1_inter1` — full Fulcrum
- `static_best_r<R>` — SGLang stock with `mamba_full_memory_ratio=R`
- `vllm_v0_20_0` — vLLM v0.20.0 baseline
