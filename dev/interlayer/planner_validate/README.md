# planner_validate — end-to-end engine sweeps

A/B and 4-cell driver scripts that boot a real sglang server, serve
a workload, and report whether the planner + actuator delivered the
expected effect. Maps to [`../design.md`](../design.md) D6m / D7–D11
conjectures.

## Cell labels (paper 2×2 ablation)

- `off` — default sglang. No budgeter, no planner, no actuator.
- `intra` — intralayer (LPB) only.
- `inter` — interlayer (pressure planner `xpool_planner` + transfer actuator `xpool_actuator`) only.
- `both` — both layers (`full` cell, paper's headline).

## Driver catalog

| driver | workload | design.md ref |
|---|---|---|
| `_r1.sh` | R1 random RPS=32, 1024-in/256-out | §D8 — saturated single-pool bubble harvest |
| `_r1_idle.sh` | R1 random RPS=4 | §D8b — idle workload no-regression |
| `_m3.sh` | M3 phase-shift (3 phases) | §D7 — real byte transfer + working-set invariant; also instrumented for §D6m-cov / §D6m-disc |
| `_skewed.sh` | skewed-popularity Zipf | §D9 — Budgeter convergence (no oscillation) |
| `_cc.sh` | CC traces (real Claude Code) | §D10 — real-world headline |
| `_burst.sh` | quiet-then-burst R1 phase shift | §D11 — Admitter handles burst synchronously |
| `_drift.sh` | linear mix drift over 10 min | §D6d — Budgeter smooth tracking |
| `_alternating.sh` | 2 s KV/mamba alternating phases | §D8c — adversarial alternating no-regression |
| `_double_sat.sh` | both pools ~95 % saturated | §D8d — Migration earns its keep |
| `_subtick.sh` | 200 ms mix oscillation, 1 Hz Budgeter | §D9c — sub-tick burst doesn't thrash Budgeter |
| `_super_burst.sh` | 2× max_running_requests in 1 s | §D11n — super-capacity defer |

## Reproduce

All drivers take the same shape:

```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=<gpu> GPU_LIST=<gpu> \
    bash dev/interlayer/planner_validate/_<workload>.sh
```

Each driver:
1. Boots the engine per-cell (`--radix-eviction-policy lpb` for
   `intra`/`both`, `SGLANG_HIMA=1` for `inter`/`both`).
2. Runs the workload via `sglang.bench_serving` or workload-specific
   driver.
3. Tears down.
4. Emits a per-cell summary table.

Per-cell run artifacts land under each driver's `runs/<driver>/<cell>/`
directory (transient — rerun to regenerate; the absolute path is
deployment-specific via `RUN_ROOT` env var, defaults to the
adjacent `runs/` next to each driver script).
