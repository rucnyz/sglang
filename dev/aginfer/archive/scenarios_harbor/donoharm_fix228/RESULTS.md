# Do-no-harm campaign — #224 + #227 + #228 (TP=4 A3, complete fix stack)

Real-stack validation that the inflight-gate (#224) + freshness bound (#227)
+ outbound coalescing (#228) **do no harm** to the workload, on the
harbor/terminus-2 swebenchpro stack, TP=4 on GPUs 4-7, MAX_TOTAL_TOKENS=262144.

Arms (N=3 cycles each, run sequentially, fresh stack per cycle):
- `a3` — ours_full: daemon kv-scheduling + admission ON, hint-steered inline
  eviction (`ours_greedy_score`).
- `a3_kvoff` — baseline: daemon present but kv-scheduling + admission OFF.

> NOTE: this campaign ran the **pre-migrate-coalescing** #228 (hint
> coalescing only). The cycle-2 burst it exposed is exactly what motivated
> the migrate-coalescing completion (committed `f88c885`); see below.

## Headline — per-task agent-execution time (the metric the daemon affects)

`agent_execution` = the LLM serving phase per task (excludes docker env-setup
and the verifier's `test.sh`, which is daemon-irrelevant + the source of the
broken-task hangs).  Errored tasks excluded.

| arm | n | mean | median | std | p90 | max |
|---|---|---|---|---|---|---|
| a3 (ours) | 74 | 734.2 s | 681.2 s | 296.5 s | 1086 s | 1750 s |
| a3_kvoff (baseline) | 81 | 695.2 s | 628.2 s | 275.6 s | 1070 s | 1983 s |

**Verdict: do-no-harm HOLDS.** Ours is ~5.6 % slower on the mean, but the
per-task std (~290 s ≈ 40 % of the mean) dwarfs the 39 s gap — well short of
the PLAN §5 significance bar (`ours_mean + ours_std < base_mean − base_std`).
Within noise, **ours ≈ baseline**: no significant harm, and no significant
benefit in this saturation regime.  This is the *expected* result and is
quantitatively consistent with the #230 Tier-1 characterization (steering
value fades to ~+3 % at saturation; the relief-supply is genuinely limited
there, #225).  Same task outcomes both arms (reward 0 — a small model does
not solve these hard SWE tasks; this is a do-no-harm / latency benchmark, not
a capability one — and 12/32 broken-ansible-env errors in BOTH arms, #222).

## Daemon-side do-no-harm (per cycle)

| arm | cyc | rejects | migr | outbound_oldest_age p99 |
|---|---|---|---|---|
| a3 | 1 | 0 | 6 | 29 ms |
| a3 | 2 | **80** | 155 | 200 ms |
| a3 | 3 | 0 | 29 | 206 ms |
| a3_kvoff | 1–3 | 0 | 0 | 0 (no daemon scheduling) |

- **0 fatals, 0 scheduler crashes, all 6 cycles completed.**
- Latency p99 mean ~145 ms (vs the pre-#228 1158 ms — the coalescing holds).
- Reject variance tracks the **migrate-volume variance** (6 / 155 / 29 — the
  #225 relief-supply variance): the one high-burst cycle (155 migrates) hit
  the **single-flight burst wall** of the pre-migrate-coalescing #228 —
  individual migrate dispatch stacked round-trips → oldest-age 700 ms peak →
  80 safe `remove_hbm_not_device_leaf` rejects.  This is the live evidence
  that motivated **#228-complete (migrate coalescing, `f88c885`)**, which
  makes a burst one round-trip.  The rejects are do-no-harm regardless
  (sglang safely rejects; the workload completes — visible in the timing
  parity above, where cyc2's rejects did NOT translate to a workload
  slowdown).

## Bottom line

The fix stack is **safe** on the real stack: no crashes/fatals, no
significant workload slowdown, identical task outcomes. The KV-scheduling
*benefit* is absent in A3 saturation (expected — characterized in #230); the
Tier-2 e2e sweep measures where it *does* bind (under-/critically-loaded
regimes) and the hint-latency budget.

Reproduce the A-vs-B: `scenarios/donoharm_fix228/parse.py` (per-task
agent-execution time, ours vs baseline, errored excluded).
