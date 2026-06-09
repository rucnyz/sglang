# T11 — Empirical p_hat estimator (PLAN §1)

PLAN.md §1 calibration task.  Replace `OursGreedyPolicy._value`'s
`p_hat = min(1, hits/age)` proxy with a workload-agnostic estimator
(histogram / per-program-bucket / Hawkes fit) per the
[todo-empirical-phat] memory entry.  Session state (REASONING /
ACTING / PAUSED / ENDED) is a feature INPUT to the estimator, not a
switch (per feedback-workload-agnostic-phat memory).

> ⚠️ **Status: OPEN WORK / not started.**  Calibration task, not
> code refactor — deliverable is a model selection + residual report
> under `results/` rather than a verify.py.  No automated test
> harness yet.

* **T11a inline-side** (`baselines/sglang_adapter.py:_node_to_unit`)
  — deliberately NOT implemented; daemon-side already shows no
  signal under current workload.  Reopen when A3 (cap
  `max_completion_tokens`) is done.
* **T11b — empirical residual estimator** for orphan units (all
  holders ended).  Not started.  Needs trace harvest (see
  `notes/trace_hooks.md`) which is also not implemented.
* **T11c — re-eval matrix** after T11a/b land.  Not started.
* **Beyond §7 myopic 1-step V_u** — MDP solver / MPC / RL with
  multi-turn horizon.  Paper-level reformulation; not started.
* **Inline scorer ↔ daemon V_u sync mechanism** — currently both
  sides compute `p_hat` from `hits/age` independently and can
  drift.  No source-of-truth mechanism designed.

## WHY THIS TASK EXISTS (REWRITTEN 2026-05-29)

> ⚠️ **Original WHY was based on a debunked "1.76× slowdown" claim.**
> The N=3 matrix (2026-05-26) measured ours_full = 1344 ± 55 s vs
> kv_off baseline = 1389 ± 40 s, **Δ = −45 s, z = −1.16 — not
> significant**.  The historical Run H' 885 s baseline used a
> DIFFERENT setting (no `temperature=0.0`); the H'_now N=3 control
> (1392.8 ± 53.6 s, no daemon, current settings) confirmed there's
> no real ~500 s gap to explain.  See
> `verify/t9/results/N3_ROOT_CAUSE.md`.

**Why T11 is still worth doing — theoretical, not empirical**:

Paper §7's `p_hat = min(1.0, hits/age)` is a uniform-Poisson proxy.
For terminus-2 / swebenchpro / 200-turn agent rollouts, reuse is
clearly NOT uniform — the workload is monotonic-extension with
strong session-state structure (see
`notes/workload_characterisation.md`).  Whether or not §7's V_u
*currently* leaves performance on the table at temperature=0
(N=3 says: not measurably — runaway generation dominates), the
proxy is still theoretically wrong.

So T11 is now **future-work motivation**, not bug-fix.  When the
workload character changes (cap `max_completion_tokens` to remove
runaway, or use a non-runaway-prone agent), the prefix-reuse
story becomes load-bearing and `hits/age` quality starts to matter.

**Current T11 status (2026-05-29)**:
* T11a daemon-side (`kv_scheduler.py:build_paper_state` program-
  alive rule): **DONE**, N=3-tested, no significant Δ vs hits/age
  baseline.
* T11a inline-side (`sglang_adapter.py:_node_to_unit`):
  **DELIBERATELY NOT IMPLEMENTED** — H'_now N=3 control showed
  the daemon proxy isn't the bottleneck, so changing the inline
  scorer wouldn't help either while runaways dominate.
* T11b residual estimator: **not started** — same reason.

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

### Final T11 design (2026-05-26): program-alive rule + residual estimator

After two rounds of reformulation (T11x rejected for hard-coding
terminus-2 constants; then over-corrected to "no rules at all"),
the principled design that survived is:

**Core rule (workload-class-general, not benchmark-specific):**

```
p_hat(u | holder p) = 1.0               if p is alive
                    = ε  /  empirical    if p has ended
```

Why this is general, not benchmark-specific:
* The anchor is **ProgramTracker.is_alive(p)** — a queryable
  predicate that exists in T6, has no magic constants
  (no τ=88s, no 14s threshold).
* For **any monotonic-extension workload** (multi-turn agents,
  long conversations, MCTS branch exploration) an alive holder
  will reuse the prefix on its next step.
* For **stateless / single-shot workloads** (one-off completion,
  RAG) each request is a short-lived program; while in flight it
  reuses its prefix, when done it dies — same rule.
* The rule does NOT bake in terminus-2's specific timing
  (88s p99 inter-turn), nor any "if dataset == X" branch.

**System-prompt high value emerges from aggregation, not a
special rule:** with K alive holders each contributing
p_hat=1.0, the unit's aggregated V_u accumulates across holders
(via T8's shared-aware aggregation, currently equal-split,
ideally Shapley).  Trunk units have 1 alive holder.  System-
prompt shared head has ~32 alive holders → naturally rises to
the top of the heap with no special-casing.

**Why this addresses the K-full 1.76× slowdown:** trunk units in
active sessions are exactly where `hits/age` fails — young units
(low age) with few hits (~0.05) get evicted, then must be
refetched the next turn.  The rule promotes them to p_hat=1.0
(20× upward correction), matching the order of magnitude of the
observed regression.

**Why p_hat = 1.0 is now theoretically defensible:**
*conditional* on the queryable predicate "program alive AND
issuing more tokens", monotonic-extension means next-step reuse
is structurally certain.  The earlier "no probability truly = 1"
objection applied to *unconditional* p_hat; conditioning on
alive-program makes 1.0 exact within the 1-step horizon of
paper §7.

### Revised sub-task plan

**T11a — Implement the program-alive rule (PRIMARY).**

Touch points (see notes/phat_inventory.md):
* Daemon side: `daemon/kv_scheduler.py:build_paper_state`
  (lines 309–310) — easy, ProgramTracker is in-process.
* Inline side: `baselines/sglang_adapter.py:_node_to_unit`
  (lines 69, 71) — needs sglang to receive ProgramTracker liveness
  snapshots from daemon (NOT polling — push on state changes:
  pause/resume/observe_arrival/program_end).

State machinery already exists from T6.  The new wire is just
"daemon → sglang: here's the current set of alive program IDs",
event-driven.

Re-run K full with this rule.  Targets:
* T11a mean ≤ Run H' 885s → primary regression eliminated, ship.
* T11a mean ≤ Run G 666s → also beats ThunderAgent, paper-strong.
* T11a mean > 1000s → rule isn't enough, escalate to T11b.

**T11b — Empirical residual p_hat (cold path; only if T11a
under-delivers OR for orphan-unit valuation refinement).**

For units whose holders have all ended, what's the chance a
*future* program reattaches?  This affects the priority of dead-
program units only — strictly cold-path, not on the hot eviction
heap.  Approaches (b1 histogram / b2 cluster-prior /
b3 Hawkes-Pareto fit) remain valid; pick by held-out log-loss on
trace data.  Same workload-agnostic discipline: behavioural
features only, no benchmark labels.

**T11c — Integration + ablation.**

After T11a (and optionally T11b), re-run K full + a second
workload if available, to confirm the rule generalises beyond
terminus-2.

### Role of harvested traces

`notes/trace_hooks.md`'s instrumentation is still worth doing,
but its purpose is **validation**, not estimator construction:
* Verify on terminus-2 traces that alive-program trunk units do
  hit ~100% of the time within the next ACTING step.  If not,
  the rule has a flaw worth understanding.
* Provide training data for T11b's residual estimator (post-
  program-end reuse patterns).
* Coverage check: 200 turns × 32 trials per Run K ≈ 6400 turn
  events, plenty of data.

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
