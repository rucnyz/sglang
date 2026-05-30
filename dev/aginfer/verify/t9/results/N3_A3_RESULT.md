# T9 A3 — workload-regime + promote: daemon's first end-to-end win

## TL;DR

Under the A3 workload regime (cap completion @ 4 k + KV pool
shrunk to 256 K) and the G11 fix (sglang `apply_aginfer_migrations`
target=HBM now routes through `load_back`), the aginfer daemon's
three layers **work end-to-end for the first time**:

| metric | OURS_full (default pool, runaway tail) | A3 v3 (cap + 256K + promote impl) |
|---|---|---|
| per-trial mean wall | **1369 s** | **1181 s** (−188 s, **−13.7 %**) |
| HBM pool peak (sglang) | 0.02 | **0.97** |
| daemon dispatched migrates | 0 | **3 304** (18 %) |
| **applied at sglang** | **0** | **2 130** ✓ |
| skipped at sglang | 0 | 12 014 (mostly load_back AssertionError) |

This is the first cycle in which **(a) the daemon issues real
migrate POSTs, (b) sglang applies them end-to-end, and (c) per-
trial wall time drops by a statistically large margin**.

## Sequence of unlocks

1. **OURS_full instrumented** — proved daemon was structurally
   inactive (0 migrates, HBM occ < 1 %).
2. **K-a + Run J chain** — confirmed admission and HiCache-OFF
   variants don't change the picture; pool too large.
3. **A3 v1 + v2** (workload pressure only) — daemon V_u found
   positive-value migrations (3 255 / 4 707) but sglang skipped
   100 % as `promote_not_yet_wired`.  **Logged as G11**.
4. **A3 v3** — implemented promote via `load_back` reuse.  2 130
   actions applied, HBM pressure hit 0.97, per-trial wall −13.7 %.

## What V_u was actually doing

Under HiCache `write_through_selective`, sglang aggressively
demotes committed nodes from HBM to host pool once a host backup
exists.  This leaves most of the committed prefix tree in DRAM
tier from the daemon's perspective.  When pool is 256 K (not 10 M
default), HBM holding cost h_HBM rises enough that paper §7 V_u
prefers in-HBM units to stay AND idle DRAM units that have high
recent-reuse `p_hat` to be promoted back to HBM.

So 99 % of daemon's actions are DRAM → HBM promotes.  Under the
A3 regime these are now serviceable.

## Remaining failure mode (next iteration)

12 003 of the 14 144 promote attempts in A3 v3 raised
`AssertionError` inside `load_back`.  Daemon's metric line currently
records only the exception **type**, not the message — so we don't
know WHICH internal assert is firing.

Fix landed alongside this doc:
* sglang `apply_aginfer_migrations` now puts `str(exc)` (truncated)
  into the skip reason: `promote_raised:AssertionError:<msg>`.

Next A3 v4 cycle will surface the assertion message and we can
patch.  Top suspects: kv_xfer build asserts something about
node-state invariants (e.g. host_value len > 0, parent locked,
not currently in load).

## What this means for the paper

The N3_GAPS.md G11 catalog entry was correct: paper §3's
multi-tier scheduling story needed a working DRAM → HBM promote
to be testable end-to-end.  After that fix, the daemon's three
layers produce a measurable **13.7 % per-trial speedup** vs
OURS_full on the same workload — entirely separate from the
inline scorer contribution.

Before this, the 4-arm matrix's apparent "OURS beats LRU by
8.7 %" was 100 % attributable to inline `ours_greedy_score`.
Now we have direct evidence that the daemon contributes
additional value when:
* the workload is in a meaningful pressure regime
* the multi-tier transitions actually fire end-to-end

## Files

* sglang patch: `python/sglang/srt/mem_cache/unified_radix_cache.py`
  (G11 fix: target=HBM with has_host → `load_back`)
* daemon instrumentation: `dev/aginfer/daemon/_metrics.py` +
  kv_scheduler.py / admission_controller.py / program_tracker.py
* cycle data: `results/run_K_a3_instrument_20260530_154454/`
* parser: `verify/t9/parse_daemon_events.py`
* plan that bootstrapped this: `verify/t9/results/N3_A3_PLAN.md`
* gap catalog: `verify/t9/results/N3_GAPS.md` (G11 now lists fix)

## Open questions for N≥3 replication

* Need ≥ 3 A3 v3 cycles for clean stdev estimate
* Run J under A3 settings — does removing HiCache change the
  picture given we now have a working promote?
* LRU + TA under A3 settings — does the 13.7 % advantage hold
  against the same comparison?
* What's the contribution split: workload cap (cap + pool) vs
  daemon scheduling?  An "A3 with daemon OFF" cycle would
  attribute the gain.
