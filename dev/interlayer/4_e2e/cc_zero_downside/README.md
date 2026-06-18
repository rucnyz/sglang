# cc_zero_downside — cross-pool rebalance: zero steady-state cost + pressure-scaled win

On real Claude-Code agent traces (`dev/eval/datasets/cc_long_traces.jsonl`, 106
hyperswitch sessions), the cross-pool KV↔mamba rebalance (Budgeter → XPoolPlanner
→ Admitter cross-fire) must clear two bars:

1. **Zero steady-state cost** — the always-on Budgeter/Admitter machinery adds no
   measurable per-iteration overhead when it is not firing.
2. **Pressure-scaled win** — when the mamba pool is undersized for the working
   set, the mechanism grows it (k2m cross-fire) and retains hot snapshots, so
   `cache_hit` climbs; when the pool has slack it stays dormant (no regression).

Both hold. Results are harvested into self-contained JSON under
`dev/interlayer/4_e2e/results/` (per-request cache_hit from the server's exported
metrics JSONL; never log-scraped) — see `harvest_run.py`.

## Gate 1 — steady-state cost → 0

### Root cause (2026-06-13): `free()`'s per-slot `torch.isin` over the 560k tail

The dynamic-cap arena KV allocator reserves a large page-id space `[1, max_size]`
but only a prefix is physically backed; the unbacked headroom tail (`~560k` ids)
is "capped". The **original** design ("Convention A", #134) kept those capped ids
*inside* `free_pages` and filtered them out everywhere — including a
`torch.isin(freed_slots, _capped_pages_560k)` + `.item()` on **every `free()`**
(every decode step frees finished/evicted slots). Micro-bench: that isin cost
**~370µs/free** (vs ~35µs for a plain free), which **halved decode throughput**
(e2e 804 vs 1519 tok/s). An earlier `need_sort=True` / `_capped_lo` /
`_n_allocatable` fix targeted `alloc`'s isin but left the dominant `free()` isin
in place — the real root cause.

### Fix — `CappedFreeList`: the tail is implicit, never materialized

`TokenToKVPoolAllocator` now delegates its free list to a self-contained
`CappedFreeList` (`mem_cache/capped_free_list.py`). The capped tail is an integer
boundary `tail_lo` (never a tensor); a drained mid-range page is a free id named
in a tiny `marks` set; `free_ids` holds every free id and `alloc` simply skips
`marks`. Consequences:

1. **`free()` is a plain append** — no capped filter at all. A capped page is
   never live (the cross-fire `cap_barrier` caps only genuinely-free pages), so it
   can never be freed. The 560k isin is *structurally* gone.
2. **`alloc()` pops the lowest n free ids** — sync-free, O(batch); when no drain
   is in flight (the steady state) zero capped checks; when a drain is in flight,
   one tiny `isin` over `marks` (not the 560k tail).
3. **Cross-fire `mark`/`unmark` are O(K)** — they edit only the small `marks` set,
   never reallocating `free_ids` (the per-fire scheduler-thread cost no longer
   scales with the free list). `need_sort=True` is still forced for arena KV, but
   now only to keep allocation low-id-first so the (high) drained pages stay out
   of the alloc head.

### Measured — the tax is gone, at the op level and end-to-end

**Micro-bench** (in-process, noise-free; `microbench_capped_free.py`, GPU7, near-
full 560k pool, bs=64): arena `free()` **9.6µs ≈ plain `free()` 9.7µs (0.99×)** —
was ~11×. The per-token allocator op is at parity.

**E2E decode throughput** (median `gen throughput`, Qwen3.5-9B, GPU7, 120s drive,
MODE=arena = budgeter on + full capped machinery + `need_sort` vs MODE=static =
vanilla allocator):

| | N=3 mean ± std | vs static |
|---|---:|---:|
| static (vanilla) | 1398.3 ± 45.5 | — |
| arena (capped machinery) | 1387.8 ± 23.1 | **−0.76%** |

The −0.76% is **noise, not signal**: static's own run-to-run std is ±45.5
(±3.3%), the arena−static gap is far inside it, and the sign **flips** across reps
(rep1 arena>static, rep2 static>arena, rep3 arena>static). Arena and static are
statistically indistinguishable — the pre-fix 804/1519 (53%) regression is gone
(now 99.2% of static, within noise). Pinned by `test_capped_free_list.py` (29
contract tests + the `mark`/`unmark`-no-realloc perf targets, by object-identity
and free-size-independence) and `test_capped_lo_fastpath.py` (allocator-level:
alloc never returns a capped id, available==alloc capacity, the cross-fire
sequence, the `_cap`-tracks-grow #320 fix). Two adversarial reviews (40 + 53
agents) confirmed the design sound; the one critical bug they found (a
drain-in-flight `alloc` short-tensor) is fixed with a TDD regression test.

## Gate 2 — pressure-curve win (#316)

The win was blocked by a planner mis-pricing: the agent charged a `kv_to_mamba`
fire the full reuse-aware cost of evicting `_n_pages_per_fire` KV pages, even
though the actuator (`XPoolFirePlanner.build`) sources **free-first** — under KV
slack a fire is `free=8 drain=0` and evicts nothing. The phantom cost drove
`NB[k2m]` negative and the planner fired only ~2× per run.

**Fix (single source of truth):** `SchedulerOwnerProvider.n_free_source_pages(direction)`
returns the same Stage-1 free supply the builder harvests. The agent prices the
drain on the volume the fire actually evicts —
`n_drain = max(0, _n_pages_per_fire − n_free)` — so a free-harvest fire is priced
at 0 and `NB[k2m]` stays positive. Symmetric for `mamba_to_kv` (a saturated mamba
has no free pages → full cost → still suppressed, preserving the #275 hot-cache
guard). Pinned by `test_drain_cost_free_netting_316.py` (7/7) and
`test_budgeter_drain_fire.py` test_E/test_E2/test_K (drained-volume pricing).

**Regression re-confirmation after the Gate-1 allocator fix** (need_sort=True for
arena-backed KV): the fire path runs through the same allocator that change
touched, so the cap=16 win was re-run request-bounded (N=3, both cells process an
identical 784-request set, parity 1.00; `run_zero_downside.sh MAX_MAMBA_CACHE=16`):

| rep | off cache_hit | inter cache_hit | Δ |
|---:|---:|---:|---:|
| 1 | 0.291 | 0.794 | +50.3 pp |
| 2 | 0.285 | 0.808 | +52.3 pp |
| 3 | 0.360 | 0.816 | +45.6 pp |

N=3 mean **+49.4 pp ± 2.8** cache_hit, output throughput **+80%** (median), TTFT
mean **−47%** / p99 **−38%**, 12 fires/rep, **0 alloc errors** — the harness
`ZERO-DOWNSIDE: PASS`. The win is intact under need_sort=True (matches the historical
+53.6 pp within rep variance); the unit-level fire-path check is
`test_capped_lo_fastpath.py::test_need_sort_cross_fire_sequence`.

**N=3 sweep over MAX_MAMBA_CACHE** (GPU7, LPB eviction, 10 sessions, conc 22,
median of per-run paired deltas; results in `dev/interlayer/4_e2e/results/d316_*.json`):

| MAX_MAMBA_CACHE | regime | fires (median) | cache_hit Δ | output_tps Δ | mean TTFT Δ |
|---:|---|---:|---:|---:|---:|
| 12 | mamba very bound | 117 | **+43.0 pp** | +28.9% | −32.3% |
| 16 | mamba bound | 85 | **+53.6 pp** | +53.3% | −51.2% |
| 256 | mamba slack | 0 | −0.3 pp | −4.4% | −0.3% |

`cache_hit` (Σcached/Σprompt over the server's per-request metrics) is the
load-bearing metric — it is retention-driven, not wall-clock, so it is robust to
host CPU contention. At cap=12/16 the mechanism fires repeatedly, grows mamba past
the working set, and lifts cache_hit from ~0.27 to ~0.70–0.82. At cap=256 it fires
**zero** times and cache_hit is flat (−0.3pp, within noise) — zero-downside.

`output_tps` / `mean_ttft` track the cache_hit gain (less recompute → faster) but
carry a host-contention caveat: the bench shared the node with another tenant, so
the cap=256 `−4.4%` tps is contention noise, not a mechanism effect (the mechanism
is provably dormant there — 0 fires).

## Run it

```bash
GPU=7 PORT=30097 TAG=d316_win MAX_MAMBA_CACHE=16 N_REPS=3 N_SESSIONS=10 \
  bash dev/interlayer/4_e2e/cc_zero_downside/run_zero_downside.sh
```

Sweep MAX_MAMBA_CACHE (12/16 bound, 256 slack) to trace the pressure curve. The
harness request-bounds the replay (both cells process the identical session set)
and auto-harvests into `results/`. `validate_zero_downside.py` asserts the
no-regression band on the paired deltas.
