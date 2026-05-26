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

## SUBAGENT BACKGROUND (collected 2026-05-26, see notes/)

Four parallel subagents collected the prerequisite info.  Detailed
write-ups in `notes/literature_survey.md`,
`notes/trace_hooks.md`, `notes/phat_inventory.md`,
`notes/workload_characterisation.md`.

### Critical synthesis (cross-cutting takeaways)

**A. Workload is essentially DETERMINISTIC, not probabilistic
   (workload_characterisation):**
* Constant 200 turns per trial (hard `max_turns` cap; all trials hit it).
* **Per-turn prefix is monotonic extension** — every turn N's message
  list == turn (N-1)'s list + one `[assistant, user]` pair.  Byte-
  identical across 199/199 boundaries in sampled trials.  No
  summarisation, no truncation.
* First **~750 tokens shared across all trials** (system prompt +
  JSON protocol spec).  Diverges into per-task PR description after.
* Inter-turn timing: p50=1.1s, p90=14s, p99=88s, max=223s.
  44 % sub-second, 87 % under 10s.

**Implication for T11**: next-turn KV reuse is essentially CERTAIN
while session is active.  The right `p_hat` for trunk units in an
active session is **~1.0**, not a Hawkes-fit value.  Stale sessions
(no turn for ≥30s suspected) decay toward 0.  The current `hits/age`
proxy systematically **undervalues NEW units** (low age + low hits
→ tiny p_hat) — but if they're in an active session, they're
*guaranteed* to be reused next turn.  **This is the bug.**

**B. Literature (literature_survey):**
* No prior LLM KV system uses probabilistic per-block reuse modeling —
  T11 is new ground.
* Classical: LIRS / LRU-K encode reuse-distance, are the strongest
  *adaptive* baselines.  ARC / SLRU / TinyLFU insufficient for
  long-horizon prefix reuse.
* Learned: Belady-imitation (Glider-style GBDT) is the theoretical
  ceiling.  LeCaR / MAB skip (their ceiling = max(LRU,LFU)).
* Temporal point processes (Hawkes self-exciting) are the
  theoretically grounded choice if we go probabilistic — but per
  (A), this workload may not need probabilistic at all.

**C. Trace hooks (trace_hooks):**
Minimum 5 instrumentation points in sglang to reconstruct full
reuse pattern (file:line, expected ~1.2 µs total per request):
* `unified_radix_cache.py:849` — CACHE_HIT (prefix match)
* `unified_radix_cache.py:1019` — INSERT_OVERLAP
* `unified_radix_cache.py:1340` — EVICT_DEVICE_LEAF (lifetime end)
* `unified_cache_components/full_component.py:206` — LOCK_ACQUIRE
* `unified_radix_cache.py:1623` — BACKUP_STORAGE (L3 transition)

**D. p_hat swap points (phat_inventory):**
Two SWAP POINTS the new estimator must touch (both compute
`p_hat = min(1.0, hits/age)` and `lambda = max(1e-3, hits/age)`):
* SWAP 1 — `baselines/sglang_adapter.py:_node_to_unit` lines 69, 71
  (inline scorer; sglang hot path; no daemon state available here)
* SWAP 2 — `daemon/kv_scheduler.py:build_paper_state` lines 309, 310
  (daemon side; full ProgramTracker state available)
* These two MUST stay consistent (within an age-counter epoch) or
  inline evictions disagree with daemon migrations → conflict rate
  > 1 % violation.

### Design discipline: WORKLOAD-AGNOSTIC ESTIMATOR

⚠️ **Reformulation rejected (2026-05-26, user direction).**  Earlier
this section proposed a "T11x — session-aware p_hat" that hard-
coded `p_hat = 1.0 for REASONING/ACTING units, decay τ=88s for
idle, fall back to hits/age for orphans`.  That was overfitting to
terminus-2.  Paper §7's V_u is a general framework; T11 must
deliver a general estimator.

**The rule:** the p_hat estimator MUST be workload-agnostic.  It
learns from observed reuse intervals.  It does not encode "if
terminus-2 then 1.0".  Session-state and benchmark-shape are
treated as **features** the estimator can use, not as rules.

**Validation discipline:** on terminus-2 (deterministic monotonic
extension) a correctly-fit estimator SHOULD converge to ~1.0 for
trunk units in active sessions.  If it does not, the estimator is
buggy — fix the estimator, do not hard-code the answer.  On RAG /
code-completion / branching workloads the same estimator must
adapt to different reuse curves without code change.

**Original T11 plan is therefore reinstated** (T11a → T11b → T11c):

1. **T11a — Trace harvest** (workload-agnostic instrumentation).
   5 hooks in sglang (see notes/trace_hooks.md).  Log (unit_hash,
   ts, kind, holder_state, scope, depth, branching_factor, hits,
   age) — features that any estimator might want.  Harvest on
   Run K full + a second workload if available.

2. **T11b — Empirical estimator** (no workload assumption).  Try
   in order:
   * b1 — Histogram + Bayesian smoothing on inter-access intervals.
     Conditioned on (UnitType, Scope, holder_state_bucket).  Beta
     prior so cold-start ≠ 0.
   * b2 — Per-(program-cluster) prior.  Cluster programs by
     early-life features only (turns/sec, branching factor, prefix
     reuse rate at age=30s).  No benchmark label is allowed as a
     feature — cluster discovery only.
   * b3 — Hawkes / Pareto fit on per-unit-type intensity.  See
     notes/literature_survey.md — this was the lit-recommended
     starting point and remains a valid path.
   Pick whichever has lowest log-loss on held-out trace.

3. **T11c — Integration + re-eval.**  Plug winner into both swap
   points (sglang_adapter.py:69/71 and kv_scheduler.py:309/310).
   Re-run K.  Target ≤ Run H' 885s; ideally ≤ Run G 666s.

### What the workload_characterisation data IS for

`notes/workload_characterisation.md` keeps its findings, but its
role changes:
* **Sanity check** for T11b — on terminus-2 traces the fitted
  estimator should predict ~1.0 for active-session trunk units
  (deterministic ground truth).
* **Coverage check** — confirms our trace harvest captures the
  multi-turn structure (200 turns × 32 trials = ~6400 turn events
  per Run K, plenty of data).
* **NOT** a basis for hard-coded rules in the estimator code.

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
