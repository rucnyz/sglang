# κ_i recompute-curve calibration (#118 / #264)

Offline calibration of the recompute-cost curve `c_i(L)` that the
Admitter / fire-planner price evictions against. `c^evict` for a radix
leaf is `Σ_b n_b · c_i(s_b)`; this folder pins the `c_i` coefficients
(`κ_i`) for one (model, GPU).

```
c_KV(L) = kv_α·L² + kv_β·L + kv_γ      (attention stack, quadratic)
c_M(L)  = m_α·L  + m_β                 (mamba/recurrent stack, linear)
```

## Why this is offline, not a boot probe

`κ_i` is a pure kernel-launch + FLOPs property of `(model, hardware)`.
Unlike `c^xfer` (driver page-table / HBM thermal — drifts per boot) and
`c_m` (side-stream contention — drifts per deployment), it does **not**
change across boots, so it only needs measuring once per `(model, GPU)`
and checking the result into the repo. A boot probe would also have to
split the per-stack `c_KV` (L²) and `c_M` (L) wall out of a single
hybrid forward, which needs model-specific module profiling (fragile).
This tool sits outside the engine and stays model-agnostic — see the
`calibrate_kappa.py` module docstring and `boot_probe.py` for the
`c^xfer` / `c_m` probes that *do* run in-engine.

## How the cost is measured

We drive `sglang.bench_one_batch`, which times the **pure prefill GPU
forward** — its measured window is CUDA-synchronize bracketed and
contains no HTTP, tokenizer, scheduler, or decode work. We do **not**
subtract any "fixed-overhead floor": there is nothing extraneous in the
window, and the residual per-forward cost (kernel-launch × layers,
~16 ms, the `L≤256` plateau) is genuine recompute cost — every
re-prefill of an evicted prefix pays it — so it belongs in `kv_γ`.

The sweep runs from the floor plateau up into the multi-chunk regime.
This range matters: chunked prefill flattens attention's L² into
per-chunk increments, so the quadratic term only emerges past the chunk
size. On a short sweep linear and quadratic fits are indistinguishable
and `kv_α` is meaningless — `calibrate_kappa.py` warns when that happens.

### Throughput alignment (the subtle part)

`bench_one_batch` prints one **compile-warmup line** (the first
input-len, re-run) before the steady-state lines, so output row-order
does **not** map 1:1 to the input-len list. We recover each row's true
`L` from `L = round(latency · throughput)` (both on the same line)
rather than positional indexing. This is immune to the warmup row and
to a sweep that OOM-truncates at the top end, and it is the fix for an
earlier off-by-one that inflated the fit RMS ~13×.

## Usage

```bash
eval "$(bash dev/eval/cost_model/calibrate.sh Qwen/Qwen3.5-9B H200)"
```

`calibrate.sh` runs the bench sweep (`REPEATS=3` by default — median per
L denoises the small-L points) and pipes the logs to
`calibrate_kappa.py`, which fits the curve and prints `export
SGLANG_CSIGMA_*` lines on STDOUT (diagnostics + plot to STDERR). The
`eval` injects them into the environment; at engine start
`cost_model.set_cost_curves` reads `SGLANG_CSIGMA_*` (env takes
precedence — a calibrated curve is never overwritten by the builtin).

To re-fit from existing logs without re-running the GPU sweep:

```bash
.venv/bin/python dev/eval/cost_model/calibrate_kappa.py run*.log \
    --model Qwen/Qwen3.5-9B --device H200
```

## Unified profile — κ_i + c^xfer + c_m in one shot (#265)

`calibrate.sh` only does κ_i. For a deployment you usually want **all
three** cost constants the Admitter / fire-planner price against. The two
others can't be measured offline (they need the live VMM arena + actuator
+ mamba pool), so they're measured by booting the engine once with the
boot probe on:

```bash
bash dev/eval/cost_model/calibrate_profile.sh Qwen/Qwen3.5-9B H200 [gpu]
# → profiles/Qwen_Qwen3.5-9B_H200.sh  (+ .json)
source dev/eval/cost_model/profiles/Qwen_Qwen3.5-9B_H200.sh   # deploy
```

It runs three stages: (1) κ_i offline (`calibrate.sh`); (2) boots the
engine with `SGLANG_BUDGETER_BOOT_PROBE=1` + `SGLANG_BUDGETER_PROBE_DUMP`
so `_run_xfer_probe` / `_run_migrate_probe` write the measured `c^xfer`
µs/page and `c_m` µs/slot to JSON, polls for the dump, tears down; (3)
merges into one source-able profile.

| constant | role in profile | env it sets |
|---|---|---|
| κ_i | persistent truth (offline fit) | `SGLANG_CSIGMA_*` |
| c^xfer | cold-start **seed** — runtime EWMA drifts on top, per design | `SGLANG_XPOOL_NB_CHUNK_COST_INIT_US` |
| c_m | persistent truth — fixed-HW constant, env-precedence skips the boot probe | `SGLANG_CM_MAMBA_PER_SLOT_US` |

c^xfer is the only seed (it drifts: driver page-table / HBM thermal);
κ_i and c_m are fixed `(model, GPU)` constants. `SGLANG_CM_MAMBA_PER_SLOT_US`
takes precedence over the boot probe (same env-precedence contract as
κ_i's curves) — a profiled deployment skips re-measuring c_m.

## Calibrated result — Qwen3.5-9B / H200

`kappa_fit_qwen3.5-9b.{json,png}` (re-fit on every run as
`kappa_fit.{json,png}`; the `_qwen3.5-9b` copy is the checked-in
archive).

```
c_KV(L) = 1.093e-7·L² + 0.02469·L + 6.44   (ms)
m_α = m_β = 0          (hybrid: total wall fit as the KV curve — see below)
c^xfer  = 90.2 µs/page (boot-probe seed)    c_m = 114.5 µs/slot (fixed)
```

| metric | value |
|---|---|
| quad RMS | **6.3 ms** (compute region rel ≤15%, large-L ≤2.3%) |
| linear RMS | 77.8 ms (12× worse → the L² term is real and necessary) |
| cubic RMS | 6.3 ms (no meaningful gain → quadratic is the right order) |
| per-forward floor | 15.9 ms (`L≤256` plateau = kernel-launch × layers) |
| L²-term share @98k | 30% |
| log-log slope b | ≈1.0 over `L≥512` (near-linear with weak quadratic curvature) |

`kv_α ≈ 1.1e-7` is corroborated to three independent sources: this
bench (1.09e-7), an HTTP `/generate` sweep (1.09e-7), and the builtin
Qwen3.5-35B constant (1.19e-7) — strong evidence it is a real physical
constant, not a fit artifact.

### Hybrid caveat (why m = 0)

On a hybrid model a single prefill mixes the attention stack (L²) and
the mamba stack (L); the total wall can't be uniquely split into `c_KV`
and `c_M`. We fit the TOTAL as the KV curve and set `m_α = m_β = 0`.
This is functionally exact for the cross-pool system: every radix-cache
entry binds KV+mamba, so evicting a leaf recomputes the whole prefix and
`c^evict` uses `c_kv + c_m = total + 0 = total`. A KV-only model gives
`m = 0` anyway.

## Files

| file | role |
|---|---|
| `calibrate_profile.sh` | **unified** entry: κ_i offline + c^xfer/c_m boot probe → one source-able profile |
| `calibrate.sh` | κ_i only: bench sweep (N repeats) → fit → `export` lines |
| `calibrate_kappa.py` | parse bench logs (throughput-aligned), fit, emit exports + plot |
| `profiles/<model>_<device>.{sh,json}` | per-(model, GPU) profile (all 3 constants); `source` the `.sh` to deploy |
| `kappa_fit_qwen3.5-9b.json` | checked-in κ_i calibration for Qwen3.5-9B / H200 |
| `kappa_fit_qwen3.5-9b.png` | full-range + zoom (floor plateau & chunk onset) plot |
