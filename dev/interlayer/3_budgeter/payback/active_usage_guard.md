# Active-usage direction guard — grow the live-bottlenecked pool, not the churny one

## Problem (after the convergence backoff)

35B swarm still regressed (`full 765 < base 778`) even with the backoff throttling
fires 3549 → 246. Cutting the fire *count* did not recover throughput, so the
cost was not fire overhead — the residual fires still drifted the split
**mamba-ward**, which is the wrong split for this workload (base is KV-heavier and
faster).

## Root cause (verified by two adversarial workflows)

The planner picks the grow direction by harm rate `R = urgency·(r_evict + r_admit)`
and, with the queue empty, `r_admit ≈ 0` — so **only cache-eviction harm
(`r_evict`) drives the decision.** On 35B swarm the mamba pool is smaller (larger
per-slot recurrent state → fewer prefixes held), so it tombstones the hot
shared-prefix trunk and sheds **2.6× the re-prefill cost** of KV. The planner
therefore grows mamba — **even though KV carries more live work**
(`usage_kv_active 0.56 > usage_mamba_active 0.50`, true in 81% of ticks). Growing
mamba steals memory from the genuinely live-bottlenecked KV.

A tempting hypothesis — that the mamba-eviction loss is *overcharged* (charging
`c_kv` for a KV-surviving tombstone) — was **REFUTED** before any code change: a
hybrid model has no mamba-only replay (`design.md:384`), so rebuilding a recurrent
state needs the full prefix forward pass; `c_kv` is the correct cost. The
`r_evict_m` signal is *right*; the planner just follows it without weighing which
pool holds the live working set. `r_evict` is a cache-reuse signal — blind to the
fact that shrinking a pool also displaces its live work.

## Fix

`PaybackConfig.active_usage_guard` (default on): after the direction is chosen,
never grow a pool by shrinking one with strictly higher active usage. Completes
the demand accounting `r_evict` leaves out.

- 35B (`kv_active > mamba_active`, 81% of ticks): blocks the wrong k2m → split
  stays at base → budgeter cost-neutral.
- 9B (`kv_active > mamba_active`, 94%): the win direction is m2k (grow the
  more-active KV), which the guard *allows*; only the minority wrong k2m is
  blocked → win preserved (in fact improved).
- Genuine mamba-bound shift (`mamba_active > kv_active`): k2m allowed → correct.

Complementary to the backoff: the guard fixes the *direction*, the backoff throttles
inelastic-harm oscillation *within* the correct direction.

## Test

`test_payback_planner.py::TestActiveUsageGuard` (4; 18/18 total pass): blocks
growing the less-active pool / allows growing the more-active pool / allows the
9B m2k / ablation with the guard off restores the pathology. Existing tests are
unaffected (they set no active usage → guard is a no-op).

## A/B (perf gate) — RESULT

`guard_validate.sh` (swarm t12 @ conc64, current build):
- **9B full (backoff+guard): 812 / 799 (n=2) — +13.3% / +11.5%**, win kept; guard
  is harmless to 9B (blocks only the minority wrong k2m).
- **35B full (backoff+guard): 765 (n=2, −1.9%)** — fires 3549 → 246 (k2m 2618 →
  92, 1109 guard-blocks). The split is now correct, but throughput still low.
- **35B wo_admt (backoff+guard, Admitter OFF): 779.8 ≈ base 778 (+0.2%)** — with
  the guard the **budgeter is cost-neutral on 35B**. The remaining −1.9% in the
  full arm is the **Admitter's per-arrival decision cost** (it fired ~nothing),
  a separate issue tracked independently — not the budgeter/split.

Net: guard + backoff make the budgeter cost-neutral on the base-optimal 35B swarm
while keeping the 9B win. The 35B `full < base` residual is the Admitter, not the
budgeter.
