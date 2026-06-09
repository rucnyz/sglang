# hbm_pressure — cap completion + shrink pool to force HBM pressure

## Workload spec

| param | value | delta vs swebench_default |
|---|---|---|
| harbor profile | `swebenchpro` | same |
| harbor agent | `terminus-2` | same |
| trials (`-n`) | 32 | same |
| concurrent (`-l`) | 32 | same |
| `--max-turns` | 200 | same |
| `temperature` | 0.0 | same |
| `seed` | 42 | same |
| **`--max-completion-tokens`** | **4096** | **NEW — caps runaway tail** |
| **`--max-total-tokens`** | **262144** (256 K) | **NEW — shrinks KV pool from ~10 M** |
| sglang TP / EP | 2 / 2 | same |
| HiCache | `--enable-hierarchical-cache --hicache-ratio 1.5` | same |
| GPUs | 5,6 (default) | same |

## Why this workload

`swebench_default` has the daemon structurally inactive:
* HBM pool peak: 0.02 — pool too big, no eviction pressure
* daemon dispatched migrations: 0
* admission pauses: 0
* runaway 60 k generations dominate (80 % of wall) — daemon can't help decode

This scenario flips both knobs:
* `max-completion-tokens 4096` cuts the runaway tail → trial-time
  variance falls, mean drops
* `max-total-tokens 262144` shrinks the device KV pool 40× → HBM
  peaks at 0.62-0.97, admission threshold (0.85) crossed for real

→ This is the regime where paper §3's multi-tier daemon
**actually fires end-to-end** (DRAM→HBM promotes, V_u-based
decisions).  Provides direct evidence for paper §3.

See [`PLAN.md`](PLAN.md) for the design rationale.

## Arms

| arm | status | what it tests |
|---|---|---|
| [`lru/`](arms/lru/) | **TODO** | baseline under HBM pressure |
| [`ta/`](arms/ta/) | **TODO** | TA program-pause under HBM pressure |
| [`ours_inline/`](arms/ours_inline/) | **TODO** | inline scorer alone under HBM pressure |
| [`ours_full/`](arms/ours_full/) | **DONE** v3 → v9 (see [RESULTS.md](arms/ours_full/RESULTS.md)) | full daemon — promote actually fires |

## Headline so far

A3 ours_full (N=4) vs swebench_default ours_full (N=3):
* Δ wall = **−165.8 s** (−12.3 %)
* z = **−4.12**, p ≈ 0.00002

Most of that delta is the workload regime change itself (runaway
cap), but the daemon contributes an isolatable ~ 80 s on top.

Once the three TODO arms run, this scenario will support a clean
LRU vs TA vs OURS comparison **under the workload where multi-tier
scheduling has work to do**.

## Reproduce

ours_full N≥3 (currently the only complete arm):
```bash
bash repro.sh                 # 2 cycles, ≈ 1 h 40 m GPU
```

Single ours_full cycle:
```bash
bash ../_shared/run_k.sh a3
```

LRU / TA / ours_inline arm runners: pending — to add once the 4-arm
matrix under hbm_pressure (task #114) is fired.

## Cycle evolution (ours_full)

| cycle | what changed | applied | per-trial mean |
|---|---|---|---|
| v3_initial | G11 promote first wired | 2 130 | 1181 s |
| v4_assert_capture | added exc.msg in skip reason | 1 300 | 1107 s |
| v5_repeat / v6_repeat | N≥3 replication | 983 / 1473 | 1219 / 1206 s |
| v7_swa_fix | swa_component soft-skip | 1 510 | 1231 s |
| v8_finegrain_decline | per-decline category tagging | 1 510 | 1150 s |
| v9_controller_root | controller sub-bucketing | 1 510 | (TBD) |

Detail: [`arms/ours_full/RESULTS.md`](arms/ours_full/RESULTS.md).
