# G10 verify — daemon HBM occ = allocator truth

`probe.py` — 5 unit-style probes that test the daemon's `pool_pressure`
plumbing in isolation (no sglang dependency).  Run after any change to:

* `python/sglang/srt/mem_cache/unified_radix_cache.py::_aginfer_pool_usage`
* `dev/aginfer/daemon/kv_scheduler.py::_flatten_per_rank` / `build_paper_state`
* `dev/aginfer/daemon/admission_controller.py::_hbm_occ`
* `dev/aginfer/baselines/base.py::SchedulerState.pool_pressure`

```bash
cd /scratch/yuzhou/projects/sglang/dev/aginfer
python verify/g10/probe.py     # expect: 5 pass / 0 fail
```

`verify.py` — live-sglang integration test.  Requires sglang up at
`AGINFER_VERIFY_BASE` (default `http://127.0.0.1:30000`).  Confirms:

1. `/aginfer/state` response carries top-level `pool_usage` key
2. `pool_usage.HBM` has the documented sub-fields
3. `token_usage = max(0, cap - avail - evict) / cap` (matches sglang's
   `full_token_usage` formula)
4. Under sustained load, `pool_usage.HBM.token_usage > 0` (proves the
   allocator view is non-zero even when radix-tree `tier_usage` is)

```bash
# Bring up sglang via scenarios/_shared/run_k.sh, then:
python dev/aginfer/verify/g10/verify.py
```

## What G10 fixed

Before: `dump_aginfer_state.tier_usage.HBM.used_bytes` is computed by
walking the radix tree (committed prefix-shareable nodes only).  Under
in-flight decode the tree-view stays near 0 even when the allocator is
at 0.97 — daemon's `admission_controller` checks `occ_hbm < theta_hi`
and never fires.

After: new `pool_usage` field mirrors sglang's `full_token_usage`
formula (`pool_size - available - evictable`).  Daemon parses
`pool_usage.HBM.token_usage` into `SchedulerState.pool_pressure[HBM]`
and `_hbm_occ` prefers it.  Admission now sees the same pressure
sglang sees and fires under real load.

## Validating the fix end-to-end

The smoke cycle that follows the code commit:

```bash
RUN_K_RESULTS_TAG=g10_smoke bash scenarios/_shared/run_k.sh a3
```

Then parse:

```bash
python scenarios/_shared/parse_daemon_events.py \
    scenarios/2_hbm_pressure/arms/ours_full/cycles/v10_g10_smoke/
```

Look for:

* `state_fetched` events with `occ_hbm > 0` (was 0.000 always)
* `admission_controller pauses > 0` (was 0 always)
* HBM occupancy time series `max > 0.85` (theta_hi crossed)
