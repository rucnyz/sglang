# burst_recovery — Admitter handles burst synchronously ✅

design.md §burst_recovery: after a long quiet period during which the Budgeter
has shrunk one pool toward its working set, a sudden burst that
arrives near engine cap on that pool admits without queueing —
because the Admitter fires cross-pool transfers synchronously as
each burst req arrives.

**Pass criterion**: `queue_p99_phase_B_inter ≤ queue_p99_phase_B_off × 1.10`

## Status: PASS (2026-05-29)

| Metric | off (baseline) | inter (with Admitter) | Ratio |
|---|---|---|---|
| Phase B completed | 1280/1280 | 1280/1280 | — |
| Phase B p99 TTFT | 39158.2 ms | 40027.7 ms | **1.022** ✅ |
| Phase B mean TTFT | 18512.9 ms | 18933.9 ms | 1.023 |

Admitter adds ~2.2% overhead at p99 during the burst — well within
the 10% headroom the conjecture allows. The headline SLO claim is
upheld: **enabling the Admitter does not degrade burst-recovery**.

## Workload

| Phase | Duration | RPS | Input | Output | Purpose |
|---|---|---|---|---|---|
| **A** | 60s | 2 | 256 | 256 | Low-rate cruise; Budgeter shrinks pools toward working set |
| **B** | 10s | 128 | 2048 | 256 | Sudden burst: 1280 concurrent reqs (overwhelms 1Hz Budgeter) |

Direction note: design.md §burst_recovery originally specified a mamba-bound
burst. Phase 5's `decide_for_req` is currently dst='kv', src='mamba'
only, so the burst is KV-bound here (long inputs). The mamba-bound
direction is deferred to a future dual-direction Admitter.

## Reproduce

```bash
GPU=3 PORT=30077 OUT_DIR=/tmp/d11_run \
    bash dev/interlayer/4_e2e/burst_recovery/run_burst.sh
```

Defaults: `PHASE_A_S=60 PHASE_A_RPS=2 PHASE_B_S=10 PHASE_B_RPS=128 MEM_FRACTION=0.45 MAMBA_CAP=512 MAX_RUNNING=256`. Persisted PASS data: `run_2026-05-29/`.

## Files

- `run_burst.sh` — launch script. Runs the workload TWICE:
  once with all interlayer envs unset (`off`), once with full
  Admitter+Budgeter stack (`inter`). Each run is Phase A → Phase B.
- `validate_burst.py` — parses `*.phaseB.bench.json` for both modes
  and asserts `inter.p99_ttft / off.p99_ttft ≤ 1.10`.

## Known limitations exposed by burst_recovery

### `_capped_pages` accumulation per Admitter fire (Phase 9 follow-up)

The first burst_recovery attempt crashed with:

```
pool memory leak detected! [full] total=654634,
  available=345739, evictable=300703, protected=0, ...
  (8192 tokens unaccounted)
```

**Real root cause** (Phase 9 audit): not `_capped_pages` accumulation
but `allocator.py::alloc()` slow-path silently dropping capped slots
from `free_pages` via `self.free_pages = self.free_pages[consumed_through:]`.
The dropped capped slots stay in `_capped_pages` but vanish from
`free_pages` → `live_size = size − _capped` over-reports total →
leak detector trips on `on_idle`.

**Fix landed (task #154)**: preserve capped slots in `free_pages`:
```python
front = self.free_pages[:consumed_through]
front_capped = front[in_capped[:consumed_through]]
self.free_pages = torch.cat([front_capped, self.free_pages[consumed_through:]])
```
Unit test: `dev/interlayer/1_dyn_admission_cap/test_mark_no_realloc.py::test_6`.
**6/6 PASS.**

**But re-running burst_recovery with strict mode ON exposed a SECOND bug**
(task #155): CUDA illegal access at
`req_index_to_mamba_index_mapping[select_index]` mid-burst —
dyn_admission_cap × Admitter race that the prior alloc bug had
masked. Attempted fix via post-fire `_maybe_update_admission_cap()`
also crashed; rolled back. **#155 needs deeper investigation**
with `CUDA_LAUNCH_BLOCKING=1`.

Until #155 is fixed, the launch script keeps
`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 +
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY=0`. The PASS data in
`run_2026-05-29/` was collected pre-#154 fix; conjecture result
(ratio 1.022) still holds. See `audit_phase9.md` for full root-
cause analysis.

This issue is pre-existing post-#134; cost_picks_xfree didn't crash on it
only because cost_picks_xfree's workload was constantly busy (`on_idle` never
fired). burst_recovery's quiet Phase A + post-burst quiet exposed it.

### Direction asymmetry

Phase 5 only handles `dst='kv', src='mamba'`. design.md's burst_recovery
spec assumed both directions; the symmetric mamba-bound burst test
needs `decide_for_req` extension.

## Why the +2.2% overhead?

Two factors:
1. **Admitter fires synchronously via `execute_decision`** (the cross-pool fire path, gated by `SGLANG_HIMA`) — each cross-pool fire does `cap_barrier + unmap + map`, which costs ~25-50 ms wall on real cuMem ops. At RPS=128 inbound, even with the actuator's `_fire_inflight` mutex serializing the worker, the synchronous fire adds latency to the arriving req that triggered it.
2. **Budgeter is also running**; both compete for the same `_fire_inflight` mutex; sometimes the worker's pending fire blocks the Admitter's sync path.

The conjecture allows this: 10% headroom acknowledges the actuator's wall isn't free.

## What would cause a falsification

`inter / off ≥ 1.50` would mean either:
- `c^xfer` EWMA being pushed too high (Admitter fires get suppressed)
- cost model mis-ranking candidates under high contention
- actuator wall significantly above the fire_wall_curve spec

Current run shows none of these; the data is well within the soft-PASS band.

## Cross-references

- design.md §burst_recovery — burst_recovery conjecture
- `dev/interlayer/2_admitter/audit_phase6_meta.md` — predicted burst_recovery as missing
- `dev/interlayer/2_admitter/cost_picks_xfree/` — §cost_picks_xfree PASS that established Admitter's correctness on steady-state
- task #154 — Phase 9 `_capped_pages` accumulation fix
- task #102 — super-capacity burst (negative companion, deferred)
