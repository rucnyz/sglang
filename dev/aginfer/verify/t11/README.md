# T11 — Empirical p_hat / scoring function (replace paper §7 1-step greedy)

## WHY THIS TASK EXISTS

T9 Run K full and K-a both came in ~1.76× slower than the inline-
scorer-only baseline (Run H' ≈ 885 s).  Disabling `admission_controller`
changed nothing.  Both ablations share the inline scorer +
`kv_scheduler`'s V_u-based decisions, and **V_u depends on `p_hat`,
which we currently estimate as `min(1.0, hits/age)`**.

Per user direction (memory:feedback-design-ideal-over-pragmatic):
"design follows the ideal — don't optimise for ease".  Per user
direction (this conversation, 2026-05-26): don't try to derive a
new closed-form formula; go **empirical** — measure the actual
reuse distribution and use that directly.

**T11 is a substantial reformulation.** Paper §7's 1-step greedy
V_u may be the wrong frame entirely for multi-turn agent workloads
(swebenchpro / terminus-2 / 200-turn rollouts).  The `hits/age`
proxy can't see the multi-turn reuse horizon.  Empirical p_hat is
the first thing to try before escalating to MDP / MPC / RL.

## SCOPE

Three sub-tasks, increasing difficulty:

### T11a — Trace harvesting

Instrument sglang's inline scorer + UnifiedRadixCache to log every
(unit_hash, access_ts, access_kind) tuple to a JSONL trace.  Enough
detail to reconstruct reuse distributions per unit type, per
program bucket, per workload phase.

* Output: `traces/run_<variant>_<timestamp>.jsonl`
* Storage: one entry per node access; ~10–100k entries per Run K.
* Implementation site: probably
  `python/sglang/srt/mem_cache/unified_cache_components/tree_component.py`
  (where `node.last_access_time = get_and_increase_time_counter()`
  lands).

### T11b — Empirical estimators (try cheapest first)

1. **Histogram + Bayesian smoothing** (target: ~half day)
   Sliding-window inter-access-interval histogram per
   (UnitType, Scope).  p_hat(τ) := P(next access within τ |
   smoothed posterior over observed intervals).  Beta prior so
   cold-start units don't get p_hat=0.

2. **Per-program-bucket priors** (target: ~1 day, needs T11a data)
   Cluster programs by early-life signatures (turns/sec, prefix
   branching factor).  Each cluster has a learned reuse curve.
   Cold-start uses cluster pooled curve.

3. **Hawkes / Pareto fit** (target: ~1–2 days)
   Fit an explicit intensity function (Hawkes self-exciting or
   Pareto tail) to T11a traces.  Ship fitted params as the
   default estimator.

### T11c — Integration + re-evaluation

Plug the chosen estimator into:
* `baselines/sglang_adapter.py:_node_to_unit` (inline scorer)
* `daemon/kv_scheduler.py:build_paper_state` (daemon V_u)

Re-run T9 Run K full; targets:
* If T11 mean ≈ Run H' 885 s → we've reverted to "no harm".  Good.
* If T11 mean < 885 s → empirical estimator beats LRU.  Paper-worthy.
* If T11 mean < Run G 666 s → empirical estimator beats TA.  Strong.

## OPEN QUESTIONS (track here, update as we learn)

1. **Is `hits/age` actually wrong, or just slightly off?**
   Run T9 kv_off variant (kv_scheduler OFF + inline scorer ON) to
   isolate.  If kv_off ≈ 885 s, kv_scheduler/V_u is the issue.  If
   still ~1550 s, inline scorer is also at fault (means even the
   eviction-heap-key form of V_u is wrong).

2. **Is multi-turn agent workload uniquely bad, or is V_u also bad
   on classical workloads?**
   Could collect a comparison trace on something like the OpenThoughts
   reasoning trace dataset.

3. **What's the right window for the histogram?**
   Too short → no signal; too long → no adaptation.  Suspect we need
   per-program window (long for stable programs, short for new ones).

4. **Should the estimator be online (updated as we serve) or
   offline (fit once, served as static lookup)?**
   Online matches paper §3 MDP framing better.  Offline simpler.

5. **What's the right object to estimate p_hat OF?**
   Currently per-(unit, t).  Could be per-(unit, t, program_state)
   if pause-affecting reuse patterns differ.

## TIMELINE

Not on T9's critical path.  Earliest realistic start: after T9 runs
J + kv_off finish (need data).  Plan ~1 week if T11a takes a day,
T11b 2 days, T11c re-runs 2 days.

## RESULTS

* date: _pending_
* T11a trace size: _pending_
* T11b estimator chosen: _pending_
* T11c re-run K mean: _pending_
* outcome (vs Run H' 885 s baseline): _pending_
