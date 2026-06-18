# idle_no_regression — light-workload no-regression

What it tests: under a light workload (RPS=4 random 1024-in/128-out,
180s), interlayer should not regress throughput vs baseline, and TTFT
either improves or degrades by at most 5%.

## Driver + validator

- `run_idle.sh` — boots sglang twice (off + inter) on the
  same model/workload, captures `{off,inter}.bench.json` +
  `inter.budgeter.jsonl`
- `validate_idle.py` — parses both, checks throughput ≤1% delta and
  TTFT bound

## Reproduce — 3-model sweep

Each row of the 3-model status table came from one invocation of the
driver with these env-vars. **Run sequentially, not in parallel** —
earlier we saw GPU 4-5 boot competing with GPU 3 boot for CPU/PCIe
inflated 122B's TTFT_off by ~50%. Total wall ~50 min for all 3.

```bash
cd /scratch/yuzhou/projects/sglang

# Row 1: Qwen3.5-9B (TP=1, ~8 min)
rm -rf /tmp/d8b_9b && \
    MODEL_NAME=Qwen3.5-9B GPU=3 TP=1 PORT=30077 OUT_DIR=/tmp/d8b_9b \
    bash dev/interlayer/4_e2e/idle_no_regression/run_idle.sh

# Row 2: Qwen3.5-35B-A3B (TP=1, ~13 min)
rm -rf /tmp/d8b_35b && \
    MODEL_NAME=Qwen3.5-35B-A3B GPU=3 TP=1 PORT=30077 OUT_DIR=/tmp/d8b_35b \
    bash dev/interlayer/4_e2e/idle_no_regression/run_idle.sh

# Row 3: Qwen3.5-122B-A10B (TP=2, ~25 min)
# Weights ~244GB / 2 = 122GB per GPU on H200 143GB → MEM_FRAC=0.92
# needed for non-trivial pool budget (the default 0.7 doesn't fit).
rm -rf /tmp/d8b_122b && \
    MODEL_NAME=Qwen3.5-122B-A10B GPU=4,5 TP=2 PORT=30078 \
    OUT_DIR=/tmp/d8b_122b MEM_FRAC=0.92 \
    bash dev/interlayer/4_e2e/idle_no_regression/run_idle.sh
```

Re-validate any single cell at any time:

```bash
.venv/bin/python dev/interlayer/4_e2e/idle_no_regression/validate_idle.py --out-dir /tmp/d8b_9b
```

## Result tables (link back to commits)

| Run | Model | TTFT_off | TTFT_inter | Δ% | thrpt Δ% | commit |
|---|---|---|---|---|---|---|
| sync fire (legacy) | 9B    | 26.28 | 27.24 | +3.6%   | +0.004% | `f4c772cb4d` |
| sync fire          | 35B-A3B | 41.30 | 42.47 | +2.8%   | +0.022% | `f4c772cb4d` |
| sync fire          | 122B   | 137.33 | 117.61 | −14.4% (contaminated, GPU 4-5 booted parallel with 35B on GPU 3) | +0.110% | `f4c772cb4d` |
| async fire         | 9B    | 26.77 | 26.75 | −0.10% | +0.004% | `56e8237098` |
| async fire         | 35B-A3B | 42.21 | 42.63 | +1.00% | +0.022% | `56e8237098` |
| async fire         | 122B   | 71.62 | 70.13 | −2.09% | +0.022% | `56e8237098` |

(Post-lock-fix re-run rows added when commit `9e6e349f50`'s perf
re-bench completes.)

**Initial 1-rep post-lock reading** (later determined to be 100%
within noise band — see N=3 study below):

| async + lock (1-rep) | 9B    | 27.12 | 28.15 | +3.8% (1-rep) | +0.004% | `9e6e349f50` |
| async + lock (1-rep) | 35B-A3B | 42.40 | 43.31 | +2.15% (1-rep) | +0.022% | `9e6e349f50` |
| async + lock (1-rep) | 122B   | 68.84 | 72.28 | +4.99% (1-rep) | +0.110% | `9e6e349f50` |

**N=3 noise-floor study (9B, sequential)** — `../../0_page_state_machine/alloc_lock/TODO.md` TODO 0:

| Config | TTFT_off (mean±σ) | TTFT_inter (mean±σ) | Δ (mean±σ) | thrpt Δ |
|---|---|---|---|---|
| with lock (3 reps) | 26.00 ± 0.20 ms | 27.46 ± 0.60 ms | +5.62 ± 2.47 % | ≤0.01% |
| without lock (3 reps) | 26.54 ± 0.94 ms | 27.94 ± 0.55 ms | +5.34 ± 3.04 % | ≤0.01% |

**Lock-induced Δ = +0.28pp ± 2.26pp pooled-SE → |Δ|/SE = 0.12 — pure
noise**. Earlier single-run "+3.8% / +2.15% / +4.99%" numbers were
indistinguishable from "+5.13% / +4.89% / +8.58%" — all sit in the
same ~5%±3% TTFT-inter-vs-off band that's intrinsic to running fires
at all. Lock-acquire overhead is negligible (~100ns × 100K calls /
180s = 0.005% of wall).

The persistent ~5% TTFT-inter > TTFT_off is the **async fire's
cap_barrier cost on the scheduler thread**, not the lock. At paper-
scale (122B) this is dominated by capacity-benefit per `idle_no_regression/README.md`
3-model table above (122B inter is FASTER than off — see `./README.md`).

Toolchain: `../../0_page_state_machine/alloc_lock/noise_compare.py` for re-running the
N-reps comparison on any pair of run-dir lists.

**Active-fix sweep (N=3 each, sequential)** — `xpool_planner.py`
persist consec counter now uses `usage_*_active` (live state only,
design.md §"Budgeter — steady-state pressure rebalance" calls the "admission ceiling") instead of total occupancy that
included radix-cached snapshots:

| Config | Model | TTFT Δ (mean ± σ) | fires |
|---|---|---|---|
| pre-fix (N=3) | 9B | +5.62 ± 2.47 % | 7 |
| **active-fix v2 (N=3)** | **9B** | **+3.20 ± 3.99 %** | **0** |
| pre-fix (N=1) | 122B | +4.99 % | 22 |
| **active-fix v2 (N=3)** | **122B** | **+1.90 ± 1.71 %** | **0** |

Throughput Δ ≤ 0.01% on all reps. Fires drop to 0 on idle workloads
(both models) — phantom fires from radix-cached saturation are
eliminated. Residual ~2-3% TTFT mean cost is INDEPENDENT of fires
(0 fires, still ~3% slower) — **root cause confirmed and fixed**:
`torch.cuda.MemPool` was disabling expandable_segments (pytorch issue
165419). Arena now defaults to from_blob path which bypasses MemPool;
the MemPool branch was deleted (commits `cd3902bcc6`, `241463552d`).
See `../../0_page_state_machine/alloc_lock/README.md` "TODO 1 closed" section for bisect data
(`bisect_arena_path.sh`, N=3 paired, |Δ|/SE=2.07). Post-fix
idle_no_regression sanity: Δ = -1.16% (one rep) / +2.70% (another), well within
the ~±5pp noise band measured at N=10.

Active-fix path: `agent.py` populates `snapshot["usage_kv_active"]`
and `snapshot["usage_mamba_active"]` (= total used − tree-cache
evictable); `xpool_planner.py:_decide_inner` (nb_direction_aware
branch) uses these for `_classify` and consec counter. Falsy-zero
bug in first version (`snap.get(k, fb) or fb`) was caught + regression-
guarded by `../../3_budgeter/no_spike/test_nb_multisource_unit.py::test_F`.
