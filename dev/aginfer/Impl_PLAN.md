# aginfer — implementation plan (`Impl_PLAN.md`)

Companion to `DESIGN.md`, the **implementation** side: empirical calibration (§1),
observability (§2), the sglang (§3) + daemon (§4) work, revisit triggers (§6), the
Dynamo↔sglang layering decision (§7), and the in-engine-plugin refactor (§8). The
**experiment** side — win scenarios, the value-aware-scorer factorial, and run
methodology — lives in `EXP_PLAN.md` (§5 here is just a pointer). Each section keys
back to the `DESIGN.md` section that motivates it.

## 1. Empirical calibration tasks

Tasks that produce **data** the design depends on but doesn't
prescribe a specific value for.  Each lands a small per-task
write-up under `verify/<task>/README.md` per the per-task RESULTS
doc convention.

### T11 — p_hat estimator (DESIGN §7)

Today `p_hat(u, Δt)` is computed from a session-state-conditional
estimator with the holder product `1 - Π(1 - p_access(u, s, Δt))`.
The per-holder `p_access` table is currently rule-based on
program state (REASONING / ACTING / PAUSED / ENDED).

Goal: replace the rule with a **workload-agnostic estimator**:

- Histogram-based / per-program-bucket / Hawkes-fit candidates
  (existing `MEMORY/todo-empirical-phat.md` for context)
- Must be **session-state-aware** but not session-state-branched —
  the program state is a feature input, not a switch (per
  `feedback-workload-agnostic-phat.md`)
- Evaluation: replay traces (agentreplay token-exact, or archived
  `archive/scenarios_harbor/` cycle runs) with each candidate
  estimator, score by `Σ |observed_access - predicted_p_hat|` on
  held-out segments

When this lands, the T12 follow-on (collect `(tier, subpool, occ,
marginal_V_u)` quadruples from the same scenario cycle runs +
falsify the linear placeholder) becomes unblocked — see task
#173 + PLAN §1 T12 status.

Open output: `verify/t11/README.md`.

### T12 — `h_(τ, sp)(occ)` shape calibration (DESIGN §7)

Spec uses linear placeholder `h_(τ, sp)_max × occ`.  Real shape
is unknown; candidate functional forms:

- linear `α × occ`
- power `α × occ^γ` for γ > 1 (right-tail-heavy)
- hyperbolic `α / (1 - occ)` (diverges as occ → 1, matches §9
  admission cap)

**Status (#173 — 2026-06-01)**:
* **DONE** in `verify/t12/`:
  * `fitter.py` — `fit_all` / `best_by_aic` over the 3 candidate
    shapes (least-squares + AIC picker, simpler model wins on ties)
  * `parse_t12_log_lines` — parser for the structured
    `aginfer_metric event=t12_calibration tier=… subpool=… occ=…
    marginal_v_u=…` log format (consumed by the calibration script
    that lands with T11's scenario-run data path)
  * verify: clean-data recovery for each shape, robustness under
    5% Gaussian noise, malformed-line drop, AIC tie-break
* **DEFERRED until T11's data path lands** (T11 owns the scenario
  cycle runs that emit `(tier, subpool, occ, marginal_V_u)`):
  * wire the daemon log line into the kv_scheduler hot path
    (gated by env var so prod runs aren't spammed)
  * collect data from a real scenario cycle (S1 at minimum)
  * pick the best shape per `(tier, subpool)`; falsify the linear
    placeholder if its residual is materially worse
  * if linear loses, swap the placeholder in
    `baselines/costs.py:holding_unit_cost` (or thread the picked
    shape into the policy)

Output: `verify/t12/README.md`.

### T13 — `bw_free` EMA validation (DESIGN §5 link_stats, §7 bw_free)

Sglang instruments `recent_throughput_bps` on each direction
(HiCache write_backup / load_back for HBM↔DRAM, Mooncake put / get
for DRAM↔DISK).  Validate that the EMA tracks reality.

**Status (#172 — 2026-06-01)**:
* **DONE** in `verify/t13/`:
  * sglang emission contract (4 directions × 3 keys; cold-start
    `recent_throughput_bps == 0`; `time_since_last_sample_s > LINK_IDLE_SECONDS`)
  * daemon `bw_free` branch logic: idle / busy / saturated /
    `peak <= 0` fatal / threshold boundary
* **DEFERRED to T26** (HiCache + Mooncake instrumentation must
  land first — no ground-truth signal to compare against):
  * compare `recent_throughput_bps` vs ground-truth wall-clock per
    migrate under idle + contended conditions
  * pin `time_since_last_sample_s` monotonicity across
    consecutive state-dumps on a quiet link
  * EMA decay-rate calibration

Output: `verify/t13/README.md`.

## 2. Observability instrumentation

The design pulls some failure-mode policies out of scope, replacing
them with metrics so we can spot the problem empirically before
designing the fix.

### T14 — State-dump cost (DESIGN §10 "Observability for state-dump cost")

- Sglang side: per-call wall-clock for `dump_aginfer_state`,
  emitted as a metric (histogram).  Trigger condition: p99 > 50 ms.
- Daemon side: queue depth at `event_router` entry, time-in-queue
  per event.  Trigger condition: queue depth > 64 sustained, or
  time-in-queue > 100 ms p99.
- Either trigger flips us back into a F3-revisit task to decide
  drop-on-full vs coalesce vs incremental-state.

**F3-revisit status (#160 — 2026-06-01)**:
* **Sglang trigger FIRED** on 2026-05-31 with peak p99 = 321.94 ms
  (originally reproduced N=3 today as 343.90 ± 0.65 ms).
* **Fix landed**: HTTP-layer cache + 50 ms-cadence background
  refresh in `python/sglang/srt/entrypoints/http_server.py`.
  This is the "coalesce + background refresh" option from the
  three F3 candidates.  Daemon's HTTP-observed p99 dropped from
  343 ms → **12.4 ms** (27× lower).
* **Remaining gap**: the SCHEDULER-INTERNAL `state_dump_metrics.
  p99_ms` still reads ~336 ms.  The cache hides this from the
  daemon, but the scheduler still spends real wall-clock on slow
  dumps when GIL-contended with prefill/decode.  Reducing the
  scheduler-side compute is a separate sglang architectural
  change (dedicated thread / lock-free walk) — tracked as #179.
* **Trigger re-scoping**: PLAN T14's `p99 > 50 ms` clause should
  be split into:
    1. **Daemon-facing latency** (HTTP-observed) — the trigger
       the F3-revisit options were designed to fix.  Currently
       **12.4 ms**; under threshold.
    2. **Scheduler-internal compute** (`state_dump_metrics.p99_ms`
       from inside `_dump_aginfer_state_impl`) — separate concern;
       reducing it is the #179 follow-on.

Output: `verify/t14/README.md`.

### T15 — Hint table cross-rank divergence (DESIGN §6 "Hint consistency", round-8 H3)

The eventual-consistent hint table is justified by the
"eviction is the cross-rank sync point" argument.  The argument
fails if two ranks ever evict **different** units near-simultaneously
under stale hints.  Test plan:

- TP > 1 deployment under high hint churn (forced by aggressive
  workload mix that pushes scorer decisions repeatedly)
- Log per-rank evicted-hash set per state-dump window
- Compare sets; any divergence breaks the §6 invariant

**Status (#174 — 2026-06-01)**:
* **DONE** in `verify/t15/`:
  * `detector.py` — `detect_divergence` (window-diff over time-series
    of per-rank state dumps) + `summarise` formatter
  * `verify.py` — 11 synthetic stages (single-rank no-op, identical
    eviction no-op, distinct-rank divergence, partial overlap, 3-rank
    2v1, sustained 4-window divergence, rank-set-change ValueError,
    time_counter propagation, summary smoke)
  * `run_dp2_real.py` — real DP=2 sglang launcher + churn driver +
    detector run.  Demonstrated 102 real per_rank JSON snapshots
    parsed cleanly (parser-integrity contract green).  The
    divergence count under DP=2 is NOT a §6 signal — without
    hicache content-hash mode active, unit hashes are per-process
    counters (`node-N`) and the comparison is meaningless; driver
    now WARNs on counter-format dominance (audit #175-round2)
* **BLOCKED on sglang patch (#174)** — the §6 invariant the probe
  is actually meant to catch is cross-TP-rank divergence (NOT
  cross-DP-rank).  But `/aginfer/state` aggregates across TP ranks
  inside the tokenizer-server fanout; the per-TP-rank pre-aggregation
  view is not exposed as any endpoint today.  To finish the spec:
  * patch sglang to expose `/aginfer/state?per_tp_rank=1` (or a
    debug endpoint) that returns each TP rank's local view BEFORE
    the multi-rank aggregator in `http_server.py` runs
  * re-run the detector against TP > 1 sglang under churn
  * any non-empty report = §6 invariant break

Output: `verify/t15/README.md`.

### T16 — Coresidence budget `re_use` no-double-count (DESIGN §8 capacity_fits / re_use)

Verify post-fix B1: a paused program whose units are still
HBM-resident (kept alive by other holders) has `re_use[sp] == 0`
for those units.  Round-9 part 1 fixed the formula; this verifies
the actual daemon code agrees.

Open output: `verify/t16/README.md`.

## 3. sglang implementation work

Concrete sglang patches needed to align the code with the design.
Order roughly by dependency.

### Schema-side

1. **T17 — State-dump schema upgrade** (DESIGN §5):
   - `pool_usage.<tier>.subpools` dict — replaces the current flat
     pool_usage.HBM fields and the current SWA optional fields
     (the upgrade *replaces*, does not add-alongside; the daemon
     halts loudly if the new schema isn't present)
   - `pool_usage.<tier>.subpools[sp].page_bytes` per subpool
   - `per_program_usage[p].hbm.committed` / `inflight` as
     per-subpool dicts
   - `per_program_usage[p].state` and `pre_pause_state`
     authoritative (round-6 H2)
   - `per_program_usage[p].unit_hashes` materialised list
   - `units[i].residence: list[Tier]` (round-9 part 4b)
   - `units[i].n_bytes: {tier: {subpool: bytes}}` nested
   - `link_stats` per direction: peak_bw_bps,
     recent_throughput_bps, time_since_last_sample_s
     (no samples_in_window — the daemon needs the idle gap, not
     a sample count, per round-14 F2)
   - `tier_holding_cost` per-(tier, subpool)
   - `throughput_ema.prefill_bps`, `throughput_ema.decode_per_program[<pid>]`
   - Drop the current top-level `page_size` field
   - Drop the current SWA `swa_*` optional fields (subsumed by
     subpools dict)

2. **T18 — State-dump internal consistency** (DESIGN §10 R1):
   - single-snapshot under one read-lock so `units[*]` and
     `per_program_usage[*].unit_hashes` and `pool_usage` refer to
     the same logical timestamp
   - **Status (#170 audit — 2026-06-01)**: OPEN.  No `_lock` around
     `_dump_aginfer_state_impl` walk; readers can see mid-mutation
     state.  Tracked as **#180**.

3. **T19 — Atomic unit visibility** (DESIGN §10 D4):
   - units appear in `/aginfer/state.units` only after
     page-aligned chunk commit; partial-prefill chunks not exposed
   - **Status (#170 audit — 2026-06-01)**: STRUCTURAL.  `insert()`
     calls `key.page_aligned(self.page_size)` at lines 574/599/731
     of `unified_radix_cache.py` BEFORE the value is stored, so by
     construction every unit's `n_tokens` is a multiple of
     `page_size`.  No dedicated verify dir (the property is a
     compile-time invariant of the insert path).  Run-time pin
     deferred — should be added as a single assertion in
     `verify/integration_stress/` flavor B
     (`all(u.n_tokens % page_size == 0 for u in dump.units)`).

### Endpoints

4. **T20 — `POST /aginfer/migrate` payload** (DESIGN §6):
   - accept `{add_tiers, remove_tiers}` per action
   - implement residence-set transitions
   - add `action_id` opaque correlator
   - skip-reason classes extended with `write_through_declined:*`

5. **T21 — `PUT /aginfer/program_paused`** (DESIGN §6 round-6 H2):
   - new endpoint
   - writes `state` and `pre_pause_state` into `per_program_usage`
   - **Status (#181 closure — 2026-06-01)**: DONE.  End-to-end
     wired:
     * io_struct: `UpdateAginferProgramPausedReq` + `Output`
     * `unified_radix_cache.set_aginfer_program_state(pid, state,
       pre_pause_state)` + storage in `_aginfer_program_states`
       dict
     * Both dump paths (dict + bytes) overlay stored state onto
       `per_program_usage[pid]`; unit-less PAUSED programs still
       appear
     * scheduler.update_aginfer_program_paused dispatcher
     * tokenizer_control_mixin fan-out across DP ranks
     * http_server PUT /aginfer/program_paused
     * Idempotent (applied=0 on re-apply at same value)
     * verify/t21/: 13 stages, all green (setter validation,
       idempotency, REAL overlay-helper echo, ENDED-GC, unit-less
       programs, legacy-cache rejection, HTTP body validation)
     * **#186 audit closure**: factored the dump-path overlay into
       a single `_aginfer_overlay_program_states` helper (both
       paths call it — no divergence by construction); added lazy
       GC of ENDED-no-units entries (was unbounded growth); fixed
       HTTP `str(pid)` coercion that bypassed the empty-pid guard
       (now `_validate_program_paused_body` type-checks first);
       converted the previously-fake replica tests to call real
       production code.
     * Unblocks #183 (T30+T39 proxy disconnect) and #185 (T41
       SESSION_END for PAUSED handler).

6. **T22 — `GET /aginfer/thresholds` + `PUT /aginfer/thresholds`**
   (DESIGN §6 round-6 H3):
   - bootstrap fetch returns canonical thresholds
   - `PUT` from daemon → sglang updates its in-memory thresholds
     atomically (no local cache file — round-14 dropped the cache;
     sglang halts at bootstrap if the daemon is unreachable)
   - sglang halts on first-launch-without-cache-or-daemon

7. **T23 — `APPLY_FAILED` webhook** (DESIGN §4 round-9 B4):
   - on action apply failure, sglang fires this webhook back to
     daemon with `{endpoint, action_id, reason}`

8. **T24 — `HASH_COLLISION` webhook + detection in `apply_aginfer_migrations`**
   (DESIGN §4 + §10 round-15/16/17):
   - `apply_aginfer_migrations` already builds a `{hash → node}` map
     via a single DFS over the radix tree (`unified_radix_cache.py`
     around line 2280) — extend it to detect collision at that point
   - on collision: fire the `HASH_COLLISION` webhook with payload
     `{hash, node_a_summary, node_b_summary}`
   - daemon `fatal()`s on receipt — deployment-bug class
   - cost is one extra comparison per node in an already-O(N) DFS;
     hash computation (`compute_node_hash_values`) is already done
     lazily upstream when KV-event emission or migrate-action
     processing requires it
   - **Status (#182 closure — 2026-06-01)**: DONE.  Webhook firing
     wired end-to-end:
     * sglang side: `unified_radix_cache._aginfer_node_summary` +
       `apply_aginfer_migrations` returns `hash_collisions[]`;
       `aginfer_webhook.AginferWebhookFirer.fire_hash_collision` +
       `_send_hash_collision` (3-attempt POST, non-blocking);
       `scheduler._fire_hash_collisions` invoked inside
       `migrate_aginfer`.
     * daemon side: `EventKind.HASH_COLLISION`,
       `_hash_collision_handler` calls
       `fatal('hash_collision', ...)`, `attach_hash_collision_handler`
       wired in `main.py`.
     * verify/t24/: 8 stages incl. D0 subprocess end-to-end
       (daemon exits 1, forensic JSON has all 5 ctx keys).
     * Dedupe via `_aginfer_collision_seen` set means a persistent
       collision triggers exactly one daemon fatal per (node_a,
       node_b) pair — supervisor decides restart policy.

9. **T25 — All action endpoints idempotent** (DESIGN §10 R2):
   - re-applying the same action returns 200 with `applied=0`
   - migrate, pause/resume, hint PUT, threshold PUT
   - **Status (#170 audit — 2026-06-01)**: PARTIAL.
     * **migrate**: idempotent (skip-reasons `race:*`, `not_a_leaf`
       in `verify/t20/`).
     * **threshold PUT**: atomic apply contract tested in
       `verify/integration_stress/` flavor E.
     * **pause/resume PUT**: gated on T21 (#181).
     * **hint PUT**: idempotent overwrite-by-stamp (T40 #184) —
       equal stamp → applied=0 (verify/t40 D1 + e2e F0); newer
       stamp wins; stale stamp dropped.

### Instrumentation hooks

10. **T26 — HiCache + Mooncake throughput EMA** (DESIGN §7 bw_free, M4a):
   - HBM↔DRAM: CUDA-event bracket each `write_backup` / `load_back`
     pair; maintain EMA per direction
   - DRAM↔DISK: wall-clock bracket each Mooncake `put` / `get`;
     maintain EMA per direction
   - Expose via `state.link_stats`
   - When this lands, T13 ground-truth-vs-EMA comparison + monotonicity
     pins (deferred from T13) must be wired into `verify/t13/` (or a
     follow-on stage in `verify/t26/`).
   - **Status (#170 audit — 2026-06-01)**: OPEN.  Sglang emits
     `recent_throughput_bps = 0` cold-start placeholder; no
     instrumentation in HiCache or Mooncake yet.  Gates #172
     (T13 deferred half) which is the calibration pin.

11. **T27 — Hint clear ordering** (DESIGN §10 R3):
    - scorer's heap-iteration read happens-before eviction commit
      happens-before hint clear
    - **Status (#188 closure — 2026-06-02)**: DONE — the full hint-
      table CONSUMER (scorer-reads + birth-seed + clear ordering).
      * **Hint-aware scorer**: `SGLANG_KV_POLICY_MODULE=aginfer:hint_v_u`
        binds `UnifiedRadixCache._aginfer_eviction_score` (a cache-
        bound method reading `_aginfer_hints` by node hash) which
        computes the §7 V_u via the adapter's `hint_v_u` (shares
        `_v_u_from_unit` with `ours_greedy_score` → one formula, no
        drift; A2 drift guard).  Absent hint → local fallback, never
        bare LRU.
      * **Birth-seed**: `_aginfer_seed_birth` (p_hat≈1, no clobber)
        from BOTH `_add_new_node` (leaf) AND `_split_node` (internal) —
        the e2e showed split nodes were uncovered without the latter;
        with it `n_aginfer_hints == live_units` (full DESIGN §3
        coverage).
      * **Clear ordering**: `clear_aginfer_hint` from
        `_remove_leaf_from_parent` (the single death/commit chokepoint
        for device-evict-death / host-evict / tombstone / migrate-
        DROP), AFTER the detach → DESIGN §10 ordering.  Bounds the
        table to live units (e2e: 300 churned prefixes → bounded, no
        leak).
      * **Atomicity** (DESIGN §10 seqlock): satisfied trivially —
        single-threaded scheduler serialises the PUT handler and the
        eviction path; no CAS needed (DESIGN-vs-code note in
        verify/t27).
      * verify/t27/: 13 stages + live GPU e2e (kv_policy_loaded,
        full coverage gap=0, clear-bounded, hint PUT round-trip).
        Regression: T28 / T38 / T40 / integration_stress green.
      * **Deferred**: the V_u-aware WRITE-THROUGH (`V_u(res∪{DRAM}) >
        V_u(res)`, the #178 hook's aginfer registration) — sibling of
        the eviction scorer, same hint table.

12. **T28 — `should_write_through(node)` plugin point** (DESIGN §3
    superset framing):
    - factor out the `hit_count >= write_through_threshold` check
      into a pluggable hook
    - default implementation preserves current behaviour
    - aginfer registers a V_u-aware version when daemon is attached
    - **Status (#178 closure — 2026-06-02)**: DONE (plugin point +
      default).  `_load_write_through_policy()` +
      `_default_should_write_through(node, threshold)` +
      `self._write_through_policy` wired into `_inc_hit_count`
      (`SGLANG_WRITE_THROUGH_MODULE` env override, T9 startup line).
      Default = historical `hit_count >= threshold`, byte-identical.
      verify/t28/.  The daemon V_u-aware version (`V_u(res ∪ {DRAM})
      > V_u(res)`) is the consumer side, deferred with T27 (#188).

13. **T29 — `SGLANG_KV_POLICY_MODULE` eviction scorer plugin**
    (DESIGN §3 superset framing — partially exists):
    - default module: LRU-equivalent V_u (last_access as p_hat
      surrogate) so baseline runs match historical behaviour
    - aginfer registers its hint-table-aware V_u
    - **Status (#170 audit — 2026-06-01)**: DONE.
      `_load_eviction_scorer` exists in `unified_radix_cache.py:69`;
      `verify/t38/` covers spec resolution end-to-end (stage C1).

14. **T30 — Proxy gate disconnect awareness** (DESIGN §10 F1):
    - daemon's proxy gate awaits BOTH gate condition AND
      `request.is_disconnected()`
    - on disconnect: release gate, respond 499, transition program
      to ENDED via outbound queue
    - **Status (#183 closure — 2026-06-02)**: DONE (paired with T39).
      * `proxy._gate_or_disconnect` races `wait_if_paused` against
        `_until_disconnected` (polls `is_disconnected()` at 0.1s —
        per-request detection, not policy polling); cancels the
        loser (and awaits it, so `wait_if_paused`'s gated-count
        decrement lands)
      * the disconnect race runs ONLY when the program is `PAUSED`
        (would actually park); non-gated requests keep the plain
        `wait_if_paused` verdict-only fast path and never call
        `is_disconnected()` (audit-fix: an always-race version
        timed out T4's real-uvicorn requests by polling Starlette's
        receive channel on every request)
      * chat_completions gate: `disconnect` → end the program ONLY
        if no sibling connection is still parked on the pid
        (`has_gated_waiters`), then client_disconnected +
        enqueue_program_paused(ENDED) + 499; `ended` (F5) → 499;
        `proceed` → forward
      * per-connection-vs-per-program fix: `end(release_gate=...)`
        + `_gated_count` so a single connection's disconnect does
        not 499 a live sibling or leak the `_ended_while_gated`
        verdict to the cohort
      * reuses #185's ENDED state + wait_if_paused verdict +
        enqueue_program_paused
      * verify/t30/: 14 stages, all green (race outcomes + loser-
        cancel + client_disconnected/no-release + real-proxy
        disconnect/ended/proceed/mid-park/sibling paths)
      * Regression: T4 / T6 / T41 / T36 green.

15. **T31 — `harbor /aginfer/session_end` endpoint** (DESIGN §4):
    - out-of-band signal channel the client uses to declare
      "session done"
    - lives outside sglang, on the harbor / agent client side
    - **Status (#170 audit — 2026-06-01)**: OUT OF SCOPE.  Lives
      in the harbor / agent client, not in sglang/daemon.  No
      task tracked; will be picked up when harbor adds the
      endpoint.

### Multi-rank correctness

16. **T32 — All-rank atomic actions** (DESIGN §6 multi-rank): every
    `migrate` / `program_paused` / `hints` action is applied
    atomically across TP/EP ranks via the existing
    tokenizer-server fanout
    - **Status (#170 audit — 2026-06-01)**: DONE (impl) +
      smoke-tested (integration).  Tokenizer-server fanout exists
      across schedulers (`tokenizer_control_mixin.py`
      `*_communicator` pattern); `verify/t15/run_dp2_real.py`
      exercises multi-rank wire format end-to-end with DP=2.
      Per-action atomicity at the TP level relies on the same
      fanout for migrate (T20) / thresholds (T22); when T21 +
      T40 land they need to use the same primitive.

## 4. Daemon implementation work

Concrete daemon patches needed.  Most of the core decision rule is
already implemented in some form; the big work is the residence-set
generalization.

1. **T33 — Residence-set candidate generator** (DESIGN §7):
   - `migrate_candidates(state, D_t)` emits `(add_tiers,
     remove_tiers)` tuples
   - the 6 meaningful transitions per unit per §7

2. **T34 — Multi-axis sparse DP** (DESIGN §9):
   - `knapsack_min_cost_multi` with per-axis `bucket_size`
   - `knapsack_max_value_multi` for headroom phase
   - dict-keyed by `(tier, subpool)` tuple
   - infeasibility (no subset hits every relief target under
     destination caps) calls `fatal()` with a forensic dump per
     §10; this is an algorithm bug — DROP + Pause are always
     feasible so reaching here means top-k under-sized or a
     filter dropped a candidate it shouldn't have
   - **Status (#156 closure — 2026-06-03)**: DONE (the DP primitives).
     `baselines/knapsack.py`: `knapsack_min_cost_multi` /
     `knapsack_max_value_multi` (sparse multi-axis, per-axis
     `bucket_size`, string-tier (tier,sp) axes) + the candidate
     contract (`Migrate(cost,relief,acquired)` / `Pause(cost,relief)`
     / `Resume(gain,re_use)`).  Infeasibility raises
     `KnapsackInfeasibleError` (forensic ctx) for `joint_decide` to map
     to `fatal()` — pure + testable.  **DESIGN-vs-code**: replaced the
     DESIGN's subtract-the-delta traceback with PARENT-POINTER
     reconstruction — the relief `min(W,·)` cap is non-invertible, so
     subtracting recovers a wrong subset (cost stays right); verify/t34
     A2 caught it via a brute-force oracle.  verify/t34/: 12 stages
     (exact-vs-brute over 120 random fixtures + quantisation +
     infeasibility + Pause-always-feasible), all green.
   - **joint_decide integration (#194 — DONE, 2026-06-04)**: wired the
     primitives into the live decision path.  `migrate_candidates`
     (`baselines/ours_greedy.py`, §7), `forecast` / `pause_candidates` /
     `resume_candidates` (`daemon/admission_controller.py`, §8), and
     `joint_decide` (`daemon/joint_decide.py`, §9) replace the greedy
     `OursGreedyPolicy.decide` single-axis check AND the sequential
     admission `_on_pressure`/`_on_resolved` "Gauss-Seidel decompose"
     (both removed).  `KvScheduler.handle` runs ONE joint decision and
     `_dispatch_plan` routes the mixed plan (Migrate→POST, Pause→
     tracker.pause+PUT, Resume→tracker.resume+PUT); admission is the §8
     candidate generator gated by `KvScheduler.admission_enabled` (off =
     kv-only Run K arm).  **Three DESIGN-vs-code corrections** (see
     `verify/joint_decide/README.md`): (1) multiple-choice (at-most-one
     per unit) over a unit's transitions, not plain 0/1 — plain 0/1
     double-counts a unit's relief; (2) `cap_left` clamps to `max(0,·)`
     — a negative (over-subscribed) destination budget spuriously
     rejects zero-acquire DROP; (3) pressure infeasibility →
     **best-effort** (free max relief, re-evaluate next event), NOT
     `fatal` — in-flight-dominated pressure with no Pause candidate is a
     workload reality (caught by `verify/integration_stress`).  Verified:
     `verify/joint_decide/` (5 stages incl. brute-force MCKP oracle) +
     `verify/integration_stress/` (7 flavors green on the real B300
     stack — stage D migrate-under-traffic, stage G SESSION_END demoted
     6→5 HBM units, stage F no spurious fatal).
   - **Forecast trajectory term (#199 — DONE, 2026-06-04)**:
     `forecast_inflight_demand` + `pause_relief.future_inflight_savings`
     now assemble the full §8 product, gated per input.
     `bytes_per_token_in_subpool` is exposed (sglang
     `_aginfer_decode_bytes_per_token` →
     `pool_usage[*].subpools[sp].decode_bytes_per_token`, attention=bpt /
     Mamba=0; carried into `TierUsage.decode_bytes_per_token`).
     `E[remaining_tokens]` reads an optional per-program field and returns
     None (NOT the `max_completion_tokens` bootstrap) when absent —
     skipping the program rather than over-pausing ~5× (§8 anti-pattern).
     The term is 0 in production (decode_per_program / per-program inflight
     are sglang placeholders → forecast = used_bytes, behaviour-
     preserving); `verify/admission_controller` trajectory stage pins the
     product + T26/T11/Mamba gating with synthetic inputs.
   - **T26 measurement (#200 — DONE, 2026-06-05)**: sglang now MEASURES
     the three inputs (were `0.0`/`{}` placeholders).  The scheduler
     keeps a per-program decode tokens/sec EMA + a prefill bytes/sec EMA
     (sampled in `run_batch` — the forward strips the batch by
     `process_batch_result` time) and computes per-program in-flight HBM
     bytes (`kv_allocated_len × bpt`; `allocated − committed ≡ 0` in
     sglang), pushing all three onto the cache before each dump
     (`set_aginfer_runtime_metrics`).  Pure helpers in
     `mem_cache/aginfer_metrics.py` (`verify/t26`); live-verified by
     `verify/integration_stress` **stage T26** (real B300 stack:
     `prefill_bps≈9e8`, `decode≈700 tok/s × 4`, inflight populated).
     This ACTIVATES `marginal_pause_cost` (prefill_bps) + the in-flight
     half of `pause_relief` (inflight) in production.  The forecast
     **trajectory** term still waits on **T11 (#126)** to populate
     `expected_remaining_tokens` (gated to avoid the bootstrap over-pause).
     Follow-on **#205**: the inflight (full-KV) ∩ committed (radix)
     snapshot overlap in `pause_relief` (bounded over-count).

3. **T35 — `authoritative_tier(residence)`** (DESIGN §7):
   - HBM if present else DRAM else DISK
   - **Status (#170 audit — 2026-06-01)**: DONE.
     `baselines/base.py:ReuseUnit.authoritative_tier` @property
     implements the rule + raises ValueError on empty residence.
     `verify/t35/` (8 stages) pins all combinations:
     {HBM}/{DRAM}/{DISK}/{HBM,DRAM}/{HBM,DISK}/{DRAM,DISK}/
     {HBM,DRAM,DISK}/{}-raises.

4. **T36 — Outbound action queue + worker** (DESIGN §6 B4):
   - in-memory `asyncio.Queue[Action]`
   - dedicated worker task drains via `httpx.AsyncClient`
   - handler enqueues and returns immediately
   - `action_id` UUID assigned at enqueue time

5. **T37 — `APPLY_FAILED` event handler** (DESIGN §4 B4):
   - daemon registers a handler for the new event kind
   - default behaviour: log + let next `joint_decide` re-evaluate
   - failure-class metrics emitted

6. **T38 — Default-policy module** (DESIGN §3 superset framing):
   - LRU-equivalent V_u (uses `last_access_time` as p_hat
     surrogate, `hit_count`-aware)
   - matches sglang's historical scorer exactly when daemon is
     absent or hints are uninitialized

   **Status (#169 + #176 + #177/#178 closure — 2026-06-02)**:
   * **DONE** in `verify/t38/`: callable
     `baselines.sglang_adapter:default_policy_score(node, layer)`
     = bare `last_access_time` (the LRU-equivalent V_u); pluggable
     via `SGLANG_KV_POLICY_MODULE`; 7 verify stages.
   * **#177/#178 closure (verify/t28/, 2026-06-02)**:
     * **#177 — default scorer settled as bare LRU.**  sglang's
       in-process `_default_eviction_score` and the adapter
       `default_policy_score` are both `float(last_access_time)`,
       byte-identical → DESIGN §3 "one code path" (no-env baseline
       == default policy module).  The `hit_count·2^-50` eviction
       tie-break from #169/#176 was REMOVED: non-functional (below
       the float64 ULP at realistic `last_access_time`) and moot
       (the cache spaces every node's `last_access_time` distinctly,
       so exact ties never occur).  `hit_count`'s role per DESIGN §3
       is the write-through trigger, not eviction (→ #178).
     * **#178 — `should_write_through(node, threshold)` plugin
       point** added (`SGLANG_WRITE_THROUGH_MODULE`, mirrors
       `_load_eviction_scorer` incl. the T9 `write_through_loaded=`
       startup line).  `_inc_hit_count` now calls
       `self._write_through_policy`; the default
       (`_default_should_write_through`) is the historical
       `hit_count >= write_through_threshold`, byte-identical to
       pre-#178 (verify/t28 B4).
     * verify/t28/: 10 stages (A eviction default + cross-tree drift
       guard, B write-through plugin incl. callsite integration),
       all green.  Regression: T38 (7) + integration_stress.
   * **DEFERRED — the daemon-attached V_u-aware versions** of both
     plugins (hint-table-aware eviction scorer; V_u-aware
     write-through `V_u(res ∪ {DRAM}) > V_u(res)`) are the consumer
     side, wired with the hint-table consumer (T27 #188).

7. **T39 — F1 proxy-gate disconnect handler** (DESIGN §10 F1):
   - `program_tracker.client_disconnected(p)` API
   - enqueue `PUT /aginfer/program_paused {END}` on disconnect
   - **Status (#183 closure — 2026-06-02)**: DONE.
     `ProgramTracker.client_disconnected(pid)` (thin wrapper over
     `end()` + distinct metric) transitions ENDED + releases the
     parked gate with the 499 verdict.  The PUT enqueue lives in
     the proxy (owns the Request + outbound), a cleaner split than
     DESIGN's "tracker enqueues" sketch — same net effect.  See T30
     status + verify/t30/.

8. **T40 — F2 hint emitter** (DESIGN §6 `PUT /aginfer/hints` + §10):
   - re-score `D_t` units per event, push all to sglang
     unconditionally
   - **no shadow `{hash: last_pushed_value}` map**
   - **Status (#184 closure — 2026-06-02)**: DONE (full round-trip).
     * daemon emitter: `kv_scheduler.hints_from_state(sched_state)`
       builds one `{hash, p_hat, lambda, stamp}` per D_t unit
       (`stamp = int(sched_state.t)` = sglang's own `time_counter`,
       a monotonic restart-surviving token — NO wall-clock in the
       policy path); `handle()` dispatches them via
       `_dispatch_hints` → `outbound.enqueue_hints` BEFORE and
       independent of the migrate decision, EVERY event with a
       non-empty D_t.  No shadow cache — re-pushes are absorbed by
       sglang's overwrite-by-stamp.
     * sglang receiving side (full chain): `PUT /aginfer/hints` +
       `_validate_hints_body` (http_server) → `UpdateAginferHintsReq`
       (io_struct) → `update_aginfer_hints` rank fan-out
       (tokenizer_control_mixin) → scheduler handler →
       `UnifiedRadixCache.set_aginfer_hints` overwrite-by-stamp
       storage (`_aginfer_hints` dict) + `get_aginfer_hint` /
       `clear_aginfer_hint`; `/aginfer/state` echoes
       `n_aginfer_hints` (count, both dump paths).
     * overwrite-by-stamp: newer stamp wins; equal stamp =
       idempotent no-op (DESIGN §10 R2); older stamp = stale drop.
     * verify/t40/: 14 stages (A enqueue, B emitter incl.
       unconditional-push + no-shadow-cache, C validator, D
       overwrite-by-stamp, E daemon-wire round-trip, F0 LIVE e2e
       against a real sglang launch).  All green.
     * **Scope boundary** (deferred, separate tasks): the inline
       scorer CONSUMING the hint table for eviction order, unit-
       birth seeding (`p_hat≈1`), eviction-time hint clear ordering
       (T27 #—), V_u-aware `should_write_through` (T28 #178),
       cross-rank hint fan-out atomicity (#174 probe / T15).  This
       task is the emitter + storage + overwrite contract only.
     * Unblocks T27 (hint clear ordering — the table now exists).

9. **T41 — F5 SESSION_END-for-PAUSED handler** (DESIGN §11 F5):
   - on SESSION_END for PAUSED program: release gate with HTTP
     499, transition ENDED, enqueue PUT
   - **Status (#185 closure — 2026-06-02)**: DONE.
     * `ProgramTracker.State.ENDED` + `end(pid)` (returns prior
       state; releases the gate + marks the 499 verdict if PAUSED)
     * `wait_if_paused(pid) -> bool` verdict (False = ended-while-
       gated → proxy 499); read-once so re-arrival isn't aborted
     * `proxy.py` returns `Response(499)` on a False verdict
     * `OutboundBatch.method` (default POST) + `_post_one` dispatch
       (POST byte-identical; PUT via `.request`); `OutboundQueue.
       enqueue_program_paused`
     * `EventKind.SESSION_END` + `make_session_end_handler` +
       `attach_session_end_handler` (wired in main.py after
       kv_scheduler so F5 owns SESSION_END)
     * verify/t41/: 14 stages, all green
     * **Scope boundary**: this handler owns the F5 state-transition
       + gate-release + PUT.  The SESSION_END migrate D_t
       (`session_scoped_units` demote/drop, DESIGN §7) is the
       kv_scheduler's "SESSION_END normal path" — DONE in #187
       (the handler now composes F5 + the migrate).
   - Regression: T6 / T36 / T164 / T21 / T24 / T4 all green.

   - **T187 (#187) — SESSION_END migrate D_t** (DESIGN §4 / §7
     "SESSION_END normal path"): DONE.  The data-plane half of
     SESSION_END (the demote/drop of the ending program's exclusive
     KV), wired onto the same handler as F5.
     * `_build_decision_set` SESSION_END → `session_scoped_units(p)`
       (units with holders == {p}; shared units excluded → survive p).
     * `build_paper_state` p_hat: an ENDED holder no longer counts as
       "alive" — a unit held only by ended programs falls back to the
       workload-prior `min(1, hits/age)` (was stuck at 1.0 forever
       post-#185 because `State.ENDED != None`).  A live co-holder
       still pins p_hat=1.0.  This is the DESIGN §4 "contributes 0 to
       future p_hat" rule; also a latent-bug fix (ended programs'
       units no longer pinned in HBM).
     * SESSION_END handler composes: `end()` FIRST (so the scorer sees
       ENDED), THEN `kv_scheduler.handle` (migrate D_t = session_
       scoped, now low p_hat → demote/drop), THEN PUT {ENDED}.
       `make_session_end_handler`/`attach_session_end_handler` take an
       optional `kv_scheduler` (main.py passes `sched`; None → pure
       F5, back-compat).
     * verify/t187/: 10 stages (decision_set / p_hat ENDED carve-out /
       composed handler / composed router / real-policy keep-value /
       shared-survives), all green.
     * Regression: kv_scheduler_value_rule / T41 / T40 / T6 /
       integration_stress green.

10. **T42 — Observability logging** (DESIGN §10 F3):
    - state-fetch latency p50 / p95 / p99
    - event-queue depth at handler entry
    - time-in-queue per event
    - failure-class counters for APPLY_FAILED breakdown

11. **T43 — `fatal(reason, **context)` helper** (DESIGN §10):
    - shared entry point for every deployment-bug-class halt
    - serialises `(event, state, candidates, dp inputs, reason,
      traceback)` to `<daemon-data>/forensic/<reason>_<ts>.json`
    - logs fatal-level line pointing at the file path
    - `sys.exit(1)`; supervisor decides restart policy
    - call sites: `joint_decide` infeasibility; `bytes_at` τ-not-in-
      residence assertion; missing `link_stats` / `tier_holding_cost`
      / `throughput_ema` fields; cross-rank subpool-key disagreement;
      `peak_bw_bps ≤ 0`; daemon-attached mode losing daemon mid-run

## 5. Scenario verification & win experiments → see `EXP_PLAN.md`

The per-scenario WIN experiments (S1 predictive-promote, S2 holder-count), the
3-config factorial that proves the value-aware scorer, the per-architecture
verification (DESIGN §12 S1/S2/S3/S5), and the #230 eviction characterization now
live in **`EXP_PLAN.md`** (execution → `sglang/reproduce/RQ1/`). This file keeps only
the implementation / calibration / observability / layering / refactor work.

## 6. Revisit triggers

Conditions that flip an out-of-scope item back into the work plan.

- **F3 state-dump backpressure**: triggered by either of
  - sglang-side `dump_aginfer_state` p99 > 50 ms, or
  - daemon-side event-queue depth > 64 sustained, or
  - time-in-queue > 100 ms p99
- **Hint cross-rank divergence**: triggered if T15 ever observes
  per-rank evicted-hash divergence under TP > 1
- **Linear `h_(τ, sp)(occ)` placeholder**: replaced by whatever
  T12 calibration picks; if T12 finds linear *is* the best fit,
  the placeholder graduates from "Planned" to spec
- **Multi-LoRA scenario**: triggered when the workload mix
  includes per-request LoRA adapter activations occupying material
  HBM bytes


---

## 7. Layering decision: Dynamo (policy) ↔ sglang (mechanism)

> Folded in from the former `LAYERING_dynamo_sglang.md` (decision record 2026-06-14).

Decision record (2026-06-14). Captures *where aginfer's logic lives*, *why*, *what each
PR target gets*, and the empirically-established fact that **no Dynamo Rust change is
needed** — for the algorithm or for token-exact replay. Not yet PR'd (per request).

### TL;DR

- **aginfer is a SUPERSET of ThunderAgent**, not a same-layer peer. ThunderAgent is a
  Dynamo **router** that is **KV-blind** — its only lever is pause/admit/route. aginfer
  adds the layer ThunderAgent can't reach: **engine-level value-aware KV-tier control**
  (HBM/DRAM/Disk/Drop migrate/evict/promote).
- **Policy lives at the router layer (Dynamo)** — same layer as ThunderAgent, so the A/B
  is apples-to-apples. **Mechanism lives in the engine (sglang)**: the KV-tier action API
  + a hint-driven hot-path eviction.
- **No Dynamo Rust change is required** — neither for the runtime algorithm nor for
  token-exact replay. (Details below.)

### Layering

```
┌─ Dynamo: aginfer_router  (Python component, peer of thunderagent_router) ─────────────┐
│  POLICY (all event-scale, NOT hot-path):                                              │
│   - value model / reuse prediction (p_hat), program lifecycle                          │
│   - pause / admit decisions                                                            │
│   - strategic migrate / promote / demote across tiers                                  │
│   - pushes per-unit value as HINTS to the engine                                       │
└───────────────┬───────────────────────────────────────────────────────────────────────┘
                │  side-channel HTTP to the engine (NOT Dynamo's request path):
                │   PUT /aginfer/hints      (value priorities)
                │   POST /aginfer/migrate   (explicit tier actions)
                ▼
┌─ sglang engine ────────────────────────────────────────────────────────────────────────┐
│  MECHANISM:                                                                              │
│   (a) KV-tier action API: migrate / evict / promote primitives                          │
│   (b) hint-driven LOCAL fast eviction on the hot path: read the pushed per-unit          │
│       priority and evict lowest — a cheap local lookup, NO router round-trip             │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

### Decision vs execution — why network overhead does NOT push policy into the engine

| operation | time-scale | home | hot-path? |
|---|---|---|---|
| value/p_hat compute, predictive promote/demote, pause/admit | program-event scale (per turn / tool-call / arrival), ~100ms–s | **Dynamo router** | no |
| "which block to evict" when the pool is full | allocation loop, microseconds | **engine** | yes |

The genuinely smart parts of aginfer are **event-scale, not hot-path** → they sit happily
at the Dynamo router (ThunderAgent's pause runs at the same cadence). The only microsecond
operation is *emergency eviction selection*, and that does **not** make a decision — it
reads a **pre-pushed priority** (computed by the router's value model, pushed on program
events) and evicts the lowest. Value changes at turn-scale, not microsecond-scale, so the
hint is fresh enough; the residual staleness is bounded and managed (#227 freshness-bound,
#228 decision→apply latency). So: **policy → router; engine just needs the action API +
a hint-driven fast-evict.**

### ThunderAgent comparison (the framing)

Both are Dynamo routers → same layer → fair A/B (`aginfer_router` vs `thunderagent_router`,
same workload, same end metrics: re-prefill / makespan / cache-hit). Our edge is explicitly
the cross-layer reach ThunderAgent lacks: it can only pause; we additionally push value
hints and issue KV-tier actions the engine executes. The paper's novelty is this
**router-decides-KV / engine-executes-KV coordination**.

### Does Dynamo need a Rust change? NO.

1. **Runtime algorithm:** `aginfer_router` is a Python Dynamo component (like
   `thunderagent_router`); hints/actions reach the engine via sglang's own HTTP side-channel
   (`/aginfer/hints`, `/aginfer/migrate`), NOT through Dynamo's request schema. This is the
   #246 result: **ZERO dynamo-core change**.
2. **Token-exact replay (testing):** the OpenAI HTTP frontend 400-rejects `custom_params`
   (that path *would* need Rust) — but we do **not** use it. The **native endpoint**
   `dynamo.thunderagent_router.generate` (dynamo Python runtime client) carries `token_ids`
   + `sampling_options.custom_params` unfiltered to the worker, **still through the router**,
   **zero Rust** — proven live in `aginfer_dyn`. The only missing piece is that the
   container's sglang fork lacks our `_aginfer_force_token` hook; porting that Python hook
   (from `python/sglang/srt/managers/scheduler_components/batch_result_processor.py`) +
   restarting the worker makes forcing fire. See `dynamo-frontend-blocks-custom-params`
   memory for the full chain + payload schema.

The only scenario that needs Rust is insisting on token-exact replay through the OpenAI
HTTP frontend specifically — which we avoid via the native endpoint.

### PR split (when we PR — not yet)

- **Dynamo PR:** `aginfer_router` — the KV-aware agent scheduling policy (Python component,
  peer of thunderagent_router). Optionally richer `agent_context` fields if the design needs
  more than {trajectory_id, session_id, session_type_id} (+ priority hint).
- **sglang PR:** the engine mechanism — KV-tier action API (migrate/evict/promote) + hint
  consumption (`/aginfer/hints`) + value-aware local eviction. This is the layer ThunderAgent
  lacks.

### Experiment platforms

| goal | platform | token-exact today? |
|---|---|---|
| aginfer vs default (value-eviction, S1, S2, RQ1) — core contribution | **sglang-direct** (agentreplay → `/generate`) | ✅ yes (proven: 112/112, len_match=1.0) |
| aginfer vs ThunderAgent — SOTA baseline | **Dynamo** (native endpoint) | needs the forcing hook ported into the container sglang |

Most of the work runs on sglang-direct, no Dynamo. Dynamo is only for the ThunderAgent
comparison (its router lives there).

### Open items

- Container sglang (`/sgl-workspace/sglang`) is **stale** vs canonical `aginfer-synced` fork
  (lacks `_aginfer_force_token`). Clean fix = sync the container's sglang to the canonical
  fork (not cherry-pick one hook).
- `agentreplay` `dynamo-native` backend = an **isolated, optional** adapter (lazy `dynamo`
  import) so the reusable harness core stays HTTP/token-space pure.
- Reconcile #251 (in-engine scheduler, delete daemon) vs #246 (router-layer policy): they
  correspond to two deployment targets — in-engine for sglang-direct (no router), router-layer
  for the Dynamo/ThunderAgent comparison. Same core algorithm, different integration per
  platform.

---

## 8. Refactor: aginfer daemon → in-engine pluggable scheduler

> Folded in from the former `REFACTOR_engine_plugin.md` (original plan 2026-06-13; task #251).

**Status:** IN PROGRESS. Stage A (pure-policy extraction) DONE + verified; Stage B (the in-engine
driver) STARTED. The original plan + feasibility record follows verbatim below §8.2.

### 8.1 Verified state + plan (2026-06-16, adversarially verified, high confidence)

**Already done (Stage A):** the value model + the cache MECHANISM are fully in-engine
(`python/sglang/srt/mem_cache/aginfer/`): `base/ours_greedy/joint_decide/knapsack/admission_controller/
program_tracker/action_timeline/costs/events/eta_estimator` + plugin hooks `cache_hooks/cache_policy/
state_dump/http_*`. The engine can already STORE hints, APPLY migrations, SCORE eviction, DUMP state
in-process. The daemon's `joint_decide.py`/`admission_controller.py`/etc. are now 6-line alias-shims
to the in-engine modules — ONE canonical copy of the value math. #253 added `_aginfer_value_aware`
so default LRU honours `--radix-eviction-policy` (do-no-harm).

**THE BLOCKER (verified):** there is NO in-engine DRIVER LOOP. A grep for callers of
`joint_decide`/`migrate_candidates` in `python/sglang/srt/` returns ZERO — the policy code is
colocated in-engine but inert; the only consumer is the daemon's `kv_scheduler.handle()` over HTTP.
`scheduler.on_idle()` is pure housekeeping. So the decision brain still runs out-of-process; every
downstream deletion (proxy/outbound/event_router/daemon) is blocked until an in-engine driver exists.

**ADMISSION SPLITS (verified, mandatory):** migration → engine plugin (delete that half of the
daemon); ADMISSION (the `wait_if_paused` ingress gate + pause/resume decision) → the **Dynamo
ROUTER**, NOT the engine. Two load-bearing reasons: (1) enforcement is a request-INGRESS blocking
gate (`program_tracker.wait_if_paused`, an `await asyncio.Event`); the engine's
`update_aginfer_program_paused` is a passthrough STORE that gates nothing, and `tokenizer_manager`
has no admission gate — so a pause stored in-engine is a no-op. (2) the pause DECISION
(`pause_relief` = shadow-price `bytes//n_holders`) needs the cross-rank fleet view a per-rank engine
plugin structurally lacks. SESSION_END must ride a terminal flag on the final request from the router
(idle is indistinguishable from a slow tool-call). Net: full daemon deletion is achievable but
admission RELOCATES to the router, it does not collapse into the engine.

**Ordered remaining work (9 steps, risk-tagged):**
1. (high) **Stage B driver** — `AginferDriver` that builds D_t, calls in-engine `joint_decide`, applies
   via `apply_aginfer_migrations`/`set_aginfer_hints` in-process. Cadence: pressure-gated scheduler hook
   (occ>θ_lo + min-interval), NOT on_idle-only (on_idle only fires when idle = the LOW-pressure regime;
   the win is under flood). Gate the migrate arm behind a flag (off ⇒ stock LRU, do-no-harm).
   *(increment 1 LANDED: post-`joint_decide` decision subsystem — saturation-yield EMA #240 +
   cooldown filter #223 + `decide()` — in `aginfer/scheduler_driver.py`; daemon delegates.
   increment 2 LANDED: in-process apply — `apply_plan` (→ `cache.apply_aginfer_migrations`) +
   `apply_hints` (→ `set_aginfer_hints`) + `assignments_to_wire`/`tier_to_wire` single-sourced
   (daemon re-exports); the "no HTTP" half. increment 3 LANDED: the cadence gate `should_tick`
   (pressure-gate occ≥θ_lo + interval-throttle — the #1 hard problem) + `tick()` + the engine hook
   `Scheduler._aginfer_maybe_tick` wired into BOTH event loops next to the webhook fire,
   gated on env `SGLANG_AGINFER_IN_ENGINE` (default OFF = byte-for-byte unchanged, do-no-harm by
   construction) + try/except crash-isolation (a policy bug disables the feature, never kills the
   loop). `verify/scheduler_driver` now A–F (cadence gate + flag-off inertness incl.); scheduler.py
   imports clean; regression suites green. STILL OPEN in step 1: the tick body's decide/apply needs
   `build_paper_state` in-engine (it is still daemon-side, kv_scheduler.py:624) → increment 4; today
   tick() fires the trigger + the in-process dump only (value no-op, so flag-on is safe). Live A/B
   under load awaits the V4 stack fix (S2_RESULTS).)*
2. (med) Move occupancy/watermark detection in-process (port `AginferWebhookFirer.maybe_fire`).
3. (med) Wrap the driver tick in try/except — crash isolation is lost without the separate process;
   a policy exception must disable aginfer for the session, never kill the scheduler loop.
4. (med) Replace outbound transport with direct calls (delete `outbound.py` + `aginfer_bridge.py`);
   decide coalescing fate (#228 brought latency 1.8s→<1s — re-measure, don't naively drop).
5. (high) **Admission → Dynamo router** — port `wait_if_paused` gate + pause/resume + the
   REASONING/ACTING/PAUSED/ENDED machine + disconnect race (#183) + SESSION_END-while-gated (#185);
   emit PAUSED in the engine dump; router-side pause decision reads it.
6. (high) Move the action-timeline (predictive promote) in-process + resolve #235 ETA due_time drift.
7. (high) **Stage D — delete the daemon** (`event_router/main/proxy/outbound/kv_scheduler` + shims +
   bridge); demote the now-unneeded `/aginfer/*` HTTP endpoints; re-run aginfer-vs-thunderagent +
   the S2 holder-count + S1 predictive-promote A/Bs on the live stack with NO daemon.
8. (high) Knapsack DP isolation — worker thread + ~100ms timeout, pressure-gated; greedy fallback;
   verify TTFT stays flat under flood with the plugin on.
9. (med) Close deferred test gates (#252 server-free migrate test; single-rank n_holders; e2e A/Bs).

**Hard / unsolved (the genuinely non-mechanical ones):** (a) driver cadence under never-idle high
load — on_idle is the wrong hook; needs an event-driven trigger in the hot step with no per-step
latency. (b) state freshness vs decision latency — in-process dump is a 5-50ms tree walk (#160)
competing with the single-threaded step; #180 read-lock snapshot vs accept-stale both open. (c)
admission cross-rank view (structural → router). (d) two state machines (belief vs residence)
merging across the router↔engine boundary — no merge protocol designed; SESSION_END not
self-derivable in-engine. (e) action-timeline belief coupling — ETA lives at proxy/router, the
promote migrate at engine. (f) saturation-yield async feedback loop into a single-threaded step.
(g) coalescing latency floor (#228/#230). (h) crash-isolation loss (#220/#221 still open).

### 8.2 Original plan + feasibility record (verbatim, 2026-06-13)

### 0. Principle (the standard this plan is held to)

Per-unit KV tiering is an **engine** concern — the engine owns the KV bytes, the tiers
(HBM/DRAM/DISK), the radix state, and the eviction callsite. Program-level pause/resume is an
**orchestrator** concern. The external daemon reaching *into* the engine over a bridge was
**wrong-layer coupling** (managing an engine concern from above its abstraction). This refactor
puts each lever in its rightful layer, deletes the cross-process plumbing, and keeps the design's
**one V_u** — realised as a *plugin* (mechanism/policy split), NOT a separate process. Design = ideal:
we take the clean decomposition even though it's more work than keeping the bridge hack.

### 1. Feasibility — FEASIBLE-WITH-WORK (clean)

1. **The engine already owns the whole mechanism.** `dump_aginfer_state()` (unified_radix_cache.py
   :3846) and `apply_aginfer_migrations()` (:2770) are direct in-process calls — no HTTP, no lock
   (the scheduler thread serialises cache access). The entire daemon transport tier
   (proxy/event_router/outbound/webhook) exists **only** to bridge a process boundary that disappears.
2. **A zero-cost-when-busy hook already exists.** `on_idle()` (scheduler.py:3424) + the
   `aginfer_webhook.maybe_fire(occ)` cadence in both event loops (:1526/:1601) — a prefill-safe place
   to run the cadenced decision; pays only when the system is light/idle.
3. **The engine self-derives every §4 event.** `Req.program_id` is already present; request
   arrival/completion use the same `perf_counter` clock the proxy used. The proxy was never an
   oracle — it re-observed boundaries the engine already sees. (Only SESSION_END is not
   self-derivable — see §4.)
4. **The policy core is pure stdlib Python** (`build_paper_state`, `joint_decide`,
   `migrate_candidates`, `ETAEstimator`, `ProgramTracker`, the V_u/cost model) — it moves **verbatim**.

### 2. The clean 3-component architecture

| Component | Owns | Layer |
|---|---|---|
| **Engine plugin `AginferScheduler`** (new, in sglang) | §7 per-unit tiering (reactive scorer is already here; add the proactive migrate) | engine |
| **`aginfer_router`** (exists, in dynamo fork) | §8 program pause/resume (fleet-wide view) + stamps `program_id` + forwards SESSION_END | orchestrator |
| **DELETE**: daemon process + `aginfer_bridge.py` + proxy + outbound + event_router + webhook | — | (cross-process plumbing — gone) |

**One V_u, two timescales, both in-engine, sharing one policy module:** the reactive eviction scorer
(`SGLANG_KV_POLICY_MODULE`, already in-engine) and the proactive `AginferScheduler` plugin read the
**same** V_u module — value-consistency (DESIGN §3) preserved, now without a process boundary.

### 3. Migration map

**MOVE → engine plugin** (pure policy, verbatim): `build_paper_state` + `_build_decision_set` +
`_top_k_by_regret` (kv_scheduler.py); `joint_decide` (joint_decide.py); `migrate_candidates` /
`value_residence` / cost fns (ours_greedy.py); `knapsack` (baselines/knapsack.py); `ETAEstimator`
(eta_estimator.py); `ProgramTracker` sync core (program_tracker.py); `ActionTimeline` /
`PromoteAction` (action_timeline.py); cost constants (costs.py); `Event`/`EventKind` as a thin
internal struct.

**STAYS in engine** (mechanism, now called directly not over HTTP): `dump_aginfer_state` (:3846),
`apply_aginfer_migrations` (:2770), `_aginfer_eviction_score` + `set_aginfer_hints` (:3318/:3203),
`set_aginfer_program_state` / `set_aginfer_runtime_metrics`, `node.session_ids` propagation, the
`on_idle`/webhook cadence sites.

**MOVES → Dynamo router**: `_extract_program_id` (proxy → router stamps it); `pause_candidates` /
`resume_candidates` + the `wait_if_paused` gate (admission = the fleet-wide pause/resume decision).

**DELETE** (transport plumbing that bridged a now-gone process boundary): `EventRouter.fetch_state`
+ HTTP GET `/aginfer/state`; `EventRouter._event_worker`; `OutboundQueue`/`OutboundWorker`; the
proxy; the webhook handlers (APPLY_FAILED collapses to the in-process migrate return dict;
SESSION_END → router signal); `main.py` process wiring; `aginfer_bridge.py` (Dynamo).

### 4. Interfaces after the refactor (minimal, clean)

- **Router → Engine**: (a) `program_id` stamped on each request (engine already reads it); (b) ONE
  `SESSION_END` signal per ended program — rides the request channel (terminal flag on the last
  request), NOT a new HTTP endpoint. **This is the only signal the engine cannot self-derive** (a
  program going idle is indistinguishable from a long tool call without it).
- **Engine internal**: plugin ↔ cache mechanism = in-process direct calls (no RPC).
- **No new external API** on either side. The engine gains zero new HTTP surface; Dynamo needs no new
  migrate/state API. This is the answer to "does Dynamo lack an API" — **after this refactor, no.**

### 5. The two design decisions (both resolved BY the layering principle, not compromises)

1. **Knapsack DP isolation.** The exact 0/1 knapsack (`joint_decide`) can take ~20s worst-case
   (capped at 10⁶ cells, top-k 256). It must never block the loop thread → run it on a **worker
   thread with a ~100ms timeout, pressure-gated** (only when `occ` crosses `theta_hi`); the common
   path drains the action-timeline + applies the cached/incremental decision. Preserves the paper's
   exact-optimality (§9) while guaranteeing zero TTFT cost. (Greedy-approx is the fallback if the
   thread harness proves fragile.)
2. **Admission (pause/resume) → router, NOT the engine plugin.** Pause/resume needs the **fleet-wide
   / cross-rank inflight sum** — a property only the orchestrator has. The per-unit §7 migrate is
   **per-rank-local** (a single worker decides if *this* unit leaves *its* HBM) → engine plugin.
   This split **is** the clean layering we agreed: programs → router, KV bytes → engine.

### 6. Risks / invariants to hold

- **do-no-harm**: plugin fully inert when the flag is off; the default path stays the exact stock
  LRU walk at zero plugin cost. (Below-baseline is a bug, not a tuning gap.)
- **Crash isolation lost** (no separate process): wrap the entire plugin tick in try/except → on a
  policy exception, log + disable aginfer for the session, never kill the engine.
- **Event-clock staleness** under bursty arrival: action-timeline is designed for "past-due fires on
  next drain"; use `perf_counter` as a floor, keep monotonic gating.
- **n_holders correctness** on the single-rank path after dropping `_flatten_per_rank` (S2 boost).

### 7. Staged implementation (each stage independently reviewable + testable)

- **Stage A — lift policy core into sglang, no behaviour change.** Move the pure modules into
  `python/sglang/srt/mem_cache/aginfer_policy/` (or similar). The existing `dev/aginfer/verify/`
  suite must still pass against the moved code (import-path update only). *Review:* diff is a move +
  import rewire; verify suite green.
- **Stage B — wire `AginferScheduler` into the loop (in-process).** Cadence at `on_idle` +
  pressure-gate; read state via direct `dump_aginfer_state`, apply via `apply_aginfer_migrations`,
  hints in-process. *Review:* do-no-harm A/B (flag off == stock) + the eviction-value win
  (ours-engine 39/39 reproduced WITHOUT the daemon).
- **Stage C — knapsack isolation** (worker-thread+timeout, pressure-gated). *Review:* TTFT under
  flood is flat with the plugin on.
- **Stage D — delete daemon/bridge/proxy; move admission + program_id + SESSION_END to the router.**
  *Review:* the router A/B (aginfer_router vs thunderagent_router) runs with NO daemon/bridge in the
  picture; the only router→engine wires are program_id + SESSION_END.

### 8. Review checklist (how to review this)

1. **Layer ownership** — does each piece end up in the layer that *owns* its data? (engine = KV
   bytes/tiers; router = programs.) No component reaches below its abstraction.
2. **Interface minimality** — router→engine is exactly {program_id, SESSION_END}; no new API.
3. **One V_u** — the reactive scorer and the proactive plugin share one policy module (not two
   copies that can drift).
4. **do-no-harm** — flag-off path is byte-identical stock; plugin pays zero when off / busy.
5. **No hidden process** — after Stage D, `grep -r aginfer_bridge|daemon.main|OutboundQueue` is empty
   on the Dynamo path.
6. **Tests** — `dev/aginfer/verify/` green after the move; do-no-harm + eviction-win A/B reproduced
   in-engine without the daemon.

---

### 9. Engine fork delta — KEEP / DELETE / EXTRACT / CLEAN (audited vs clean upstream `ca0b7ea4f6^2`)

Our engine delta = **+4158 / −17 across 22 files** (purely additive — we touch ~0 upstream lines, low
merge-conflict surface). But it's **buried**:

| bucket | lines | % | |
|---|---|---|---|
| **NECESSARY-CORE** (real sglang PR) | ~900 | 22% | clean, well-formed; default path = byte-for-byte stock LRU when unset |
| **DAEMON-PLUMBING** (deletes with refactor) | ~3180 | **76%** | pure HTTP/webhook/IPC transport to the external daemon — no hot path depends on it |
| **CLEANLINESS-FIX** | ~80 | 2% | S2DBG hooks, dup docstring, mid-file imports, daemon flags in core args |

#### 9a. DELETE — daemon transport (~3180 lines), in full with the refactor
- `aginfer_webhook.py` (NEW +745) → DELETE ~675 (firing/threading/bootstrap/payloads); **extract ~70**
  (`WatermarkState` + `classify()` hysteresis + `derive_kind()`) → `aginfer/occupancy_detector.py`.
- `http_server.py` (+416) → DELETE all 5 `/aginfer/*` endpoints + 50ms refresh-cache; **extract ~75**
  (`_validate_hints_body` NaN/inf/[0,1] bounds) → `aginfer/validation.py` *iff* external hints persist.
- `unified_radix_cache.py` dump/serialization (~1265 of +1822) → DELETE: `dump_aginfer_state*`,
  `_dump_aginfer_state_impl/_dict`, `_StateDumpMetrics`, `_aginfer_program_states`+`set_aginfer_program_state`
  +overlay, `_aginfer_runtime_metrics` store, `_aginfer_pool_usage/link_stats/tier_holding_cost/
  bytes_per_token/node_summary`, collision payloads. (All echo state back over HTTP; zero engine decisions —
  the in-process plugin reads `_aginfer_hints` / `node.session_ids` / EMA dicts directly.)
- `scheduler.py` HTTP handlers (~290 of +532) → DELETE: `get/migrate/update_*` handlers, webhook init/
  cleanup/firing, `_fire_apply_failed/_fire_hash_collisions`, communicator registrations.
- `io_struct.py` (+111 of +136) → DELETE 10 IPC dataclasses (`GetAginferState*/Migrate*/UpdateThresholds*/
  UpdateProgramPaused*/UpdateHints*`).
- `tokenizer_control_mixin.py` (+91) → DELETE (5 pure pass-through communicator methods).
- `server_args.py` (~48 of +60) → DELETE daemon flags (`aginfer_notify_url/heartbeat_s/theta_*`,
  `bootstrap_thresholds_into_server_args`) — daemon-owned policy, not sglang config.
- `hybrid_cache_controller.py` `_last_load_decline` strings (~10) → DELETE (dump-only diagnostic).

#### 9b. KEEP — the clean sglang PR (~900 lines), BUT extract inline → `aginfer/` module (thin-hook principle)
- `full_component.py` (+13/−8) — **the model thin hook**: swaps `eviction_strategy.get_priority(n)` →
  `self.cache._eviction_scorer(n, EvictLayer.*)`. Minimal, LRU-identical when unset. KEEP as-is.
- `unified_radix_cache.py` (~520 necessary, currently INLINE) — the pluggable-policy framework
  (`_load_eviction_scorer/_load_write_through_policy`), `_aginfer_hints` table + accessors, `_aginfer_
  eviction_score` (V_u), `apply_aginfer_migrations`, birth-seed, `node.session_ids` propagation, clear-on-
  evict. → **EXTRACT into `aginfer/` module; leave thin hooks** (a scorer-callsite call + the session_ids
  tag on insert/split + a couple accessors). This is the bulk of the thin-hook work.
- `scheduler.py` (~240 necessary, INLINE) — throughput/EMA state + `_aginfer_record_*`/`_update_*`/
  `record_spec_decode`, `program_id` propagation, the on_idle/cadence site. → **EXTRACT the EMA/throughput
  logic into `aginfer/metrics_hooks.py`; leave thin hooks** (pre/post-forward one-liners + the tick call).
- `aginfer_metrics.py` (+144, whole file) — pure stateless utils → **move into `aginfer/` as-is**.
- `batch_result_processor.py` (+40) — `_aginfer_force_token` teacher-forcing (replay) → KEEP, thin hook.
- `schedule_batch.py` (+55) — `_sanitize_program_id` + `Req.program_id` → KEEP (minimal).
- `swa_component.py` (+42) — host-lock pair + `load_back` race short-circuit → KEEP (**V4 correctness**).
- `hybrid_cache_controller.py` `kv_buffer is None` anchor-skip → KEEP (**V4 correctness**).
- `protocol.py` (+28, dedupe docstring), `io_struct.py` (+13: program_id), `base_prefix_cache.py` (+6),
  `tree_component.py` (+11: `peek_time_counter`), `__init__.py` (+2), `session_controller.py` (+7),
  `encode_receiver.py` (+5), serving/tokenizer (+1×3) — `program_id` threaded through every entrypoint/
  disagg/session handoff. KEEP (minimal).

#### 9c. CLEANLINESS-FIX (~80 lines, do anytime — independent of the refactor)
1. DELETE 6 S2DBG debug lines: `unified_radix_cache.py:3252–3256, 3327–3330`.
2. DRY the duplicated 20-line `program_id` docstring in `protocol.py` → one `_PROGRAM_ID_DOC` constant.
3. Hoist mid-file `import os, threading` in `http_server.py:~687` to top (moot if file deleted).
4. Consolidate scattered validators → one `aginfer/validation.py`.
5. Isolate daemon flags out of core `server_args` (subsumed by 9a delete).

#### 9d. Net result
After delete + extract: the sglang PR = **~900 lines, ~13 files**, of which the upstream-file footprint
is a **handful of thin `# aginfer hook` lines** + the self-contained `aginfer/` module + the 3 small
retained kernels (`occupancy_detector.py` ~70, `validation.py` ~75, runtime-metrics push as a plugin hook).
**Reviewable, minimal, do-no-harm at the code level.**

---

### 10. Progress log

Canonical engine package: `python/sglang/srt/mem_cache/aginfer/`.

- **Stage A — DONE + verified.** 14-file policy closure (base/costs/knapsack/ours_greedy/sglang_adapter/
  events/program_tracker/eta_estimator/action_timeline/admission_controller/joint_decide/_metrics/
  _admission_math/_fatal) moved `dev/aginfer/{baselines,daemon}/` → the engine package; old paths are
  `sys.modules` alias-shims (value-model shims stay for dev research; pure-policy shims die in Stage D).
  Verify suite green against the moved code.
- **Stage A.2 increment 1 — DONE + verified.** Module-level plugin framework (154 lines: env-symbol
  loaders, the LRU-equivalent `_default_eviction_score`, write-through trigger, birth-seed constants)
  `unified_radix_cache.py:67–220` → `aginfer/cache_policy.py`; upstream = a 15-line re-import hook.
  `_default_eviction_score` **identity preserved** (swa_component's `is not` check holds). t27 17/17, t38 7/7.
- **Stage A.2 increment 2 — DONE + verified.** Contiguous hint-table + value-eviction scorer block
  (157 lines: `set_aginfer_hints` / `get`/`clear_aginfer_hint` / `_aginfer_unit_hash` /
  `_init_aginfer_eviction_scoring` / `_aginfer_eviction_score` / `_aginfer_hint_should_write_through` /
  `_aginfer_seed_birth`) `unified_radix_cache.py:3064–3220` → `aginfer/cache_hooks.py` as free functions
  over `cache`; upstream = 8 thin `(*a,**k)` delegators + 1 import. Folded-in cleanups: `hint_v_u` now
  imported via the in-package path (`...aginfer.sglang_adapter`, no `sys.path` dependence) not
  `baselines.*`; added missing `import os`. **Verify:** engine import OK · delegators wired ·
  `_default_eviction_score` identity preserved · t27 17/17 · t38 7/7 · **s2_holder_scorer PASS** (hint→
  `_v_u_from_unit`→scorer, "no drift") · kv_scheduler_value_rule/joint_decide/admission_controller/
  program_tracker PASS. (`action_timeline` FAIL is a pre-existing daemon ETA `due_time` drift, #235 —
  pulls **none** of the changed modules into its import graph, provably independent.)
- **Stage A.2 increment 3 — DONE (code-move verified by construction; 39/39 full-stack DEFERRED).**
  `apply_aginfer_migrations` (the §7 migrate EXECUTOR, 392 lines, `unified_radix_cache.py:2632–3023`) →
  `aginfer/cache_hooks.py` as a free function over `cache`; upstream = a 2-line delegator. The move is
  **behaviour-preserving by construction**: uniform de-indent + whole-word `self`→`cache` (28 subs, every
  `self` token verified a code ref — def sig + 1 `getattr(self,…)` + all `self.X`; no string/comment
  `self`). **Forward proof: 0 diffs** — every moved line equals `dedent(orig)`+`self→cache`. Tier names
  HBM/DRAM/DISK/HOST are string literals (no import); the only runtime deps imported are `EvictLayer` +
  `BASE_COMPONENT_TYPE` (circular-safe — `unified_cache_components` is already loaded before the
  `cache_hooks` import; the swa→urc back-edge is a lazy in-method import). **Verify:** compile + import OK,
  deps resolve, delegator present, t27/t38/s2_holder_scorer/kv_scheduler_value_rule(25/25)/joint_decide
  all PASS. ⚠️ **DEFERRED gate (task #252):** `apply_aginfer_migrations` has no server-free unit test
  (t20/t24 POST a live `:30001`), so the runtime migrate path is unconfirmed until the next
  `dynamo.sglang` restart runs the daemon→engine round-trip + do-no-harm + 39/39-without-daemon.
- **Stage A.2 increment 4 — DONE (code-move verified by construction + full-import; live EMA DEFERRED).**
  The 8 scheduler EMA/throughput hooks (`_aginfer_record_throughput`(+`_inner`), `_aginfer_extend_token_count`
  [@staticmethod], `_aginfer_update_prefill`/`_update_decode`, `_aginfer_record_spec_throughput`/`_spec_decode`,
  `_aginfer_push_runtime_metrics`; `scheduler.py:3674–3895`, 222 ln) → NEW `aginfer/metrics_hooks.py` as free
  functions over `sched`; upstream = 8 delegators (7 instance + 1 `@staticmethod`). The now-block-only
  `aginfer_metrics` import was **relocated** out of `scheduler.py` (5 names had zero out-of-block refs).
  Move = de-indent + strip the lone `@staticmethod` + whole-word `self`→`sched` (NOT `cache` — bodies use a
  local `cache` var). Annotations made lazy via `from __future__ import annotations`; runtime deps imported
  (`ForwardMode`, the 5 `aginfer_metrics` fns; `BASE_COMPONENT_TYPE` covered by the in-method import).
  **Verify:** 0-diff forward proof (accounting for the stripped decorator) · py_compile · **FULL
  `import scheduler` OK (no circular import)** · 8 delegators present + `_metrics_hooks` wired +
  `aginfer_metrics` relocated + staticmethod preserved · t26/t27/t38/s2_holder_scorer green. ⚠️ DEFERRED
  (task #252): the live decode/prefill EMA populate-under-load (#204) re-confirms on the next restart.

- **Stage A.2 increment 5 — DONE.** The write-through policy RESOLUTION (4 inline lines in
  `unified_radix_cache.__init__`) → `_init_aginfer_write_through(cache)` in `cache_hooks.py` + a thin
  delegator + a one-line callsite — making it the symmetric twin of `_init_aginfer_eviction_scoring`. The
  `_load_write_through_policy` re-import stays in urc only as a re-export for verify/t28. Verify: compile +
  import + `_init_aginfer_write_through` wired + re-exports intact + t28/t38/t27/t21/s2/kv_scheduler/joint_decide green.

**A.2 EXTRACTION COMPLETE.** All separable aginfer logic now lives in the self-contained package; the only
aginfer code left inline in the two big upstream files is (a) thin hooks — ~10+8 delegators, the re-import
block, ~8 callsites, 3 irreducible `node.session_ids` tags — and (b) the ~1265-line daemon dump/state plumbing
that **DELETEs (not moves)** in Stage D when the daemon HTTP path is removed (`set_aginfer_program_state`,
`_aginfer_program_states`, `_aginfer_node_summary`, `dump_aginfer_state*`, …; a few — `_aginfer_bytes_per_token`,
`_aginfer_subpool_name`, `set_aginfer_runtime_metrics` — are cross-used by `metrics_hooks` and stay). The
`node.session_ids` propagation is genuinely irreducible (1–2-line node tagging on tree split/insert) — a thin
hook by nature, not an extraction target.

#### A.2 final independent verification (3-lens) — completeness PASS, 1 fix, 1 pre-existing gap tasked

A second adversarial pass on the COMPLETE state (inc1–5 + fixes): **Completeness = PASS** (every inline aginfer
ref is a thin hook, a Stage-D-delete daemon-plumbing method, or one of the 3 cross-used keepers
`_aginfer_bytes_per_token`/`_aginfer_subpool_name`/`set_aginfer_runtime_metrics`); **inc5 + all 3 prior fixes
verified correct** (byte-identical move, `_AGINFER_VALID_STATES` is a class attr, re-exports load-bearing,
future-annotations present); **do-no-harm intact for the default config** (env-unset => `_default_eviction_score`
= `last_access_time` = stock LRU, identity preserved through the re-import chain). Findings:
- **FIXED:** dead `import os` at urc:70 (inc5 orphaned it; stale comment) — removed.
- **#253 (pre-existing, NOT A.2):** the aginfer eviction-scorer plugin replaced `eviction_strategy.get_priority(n)`
  with `_eviction_scorer`, whose default hardwires LRU => `--radix-eviction-policy lfu/slru/priority` silently
  diverges from stock even with env unset. Matches stock only for the default `lru` (what all experiments use).
  A do-no-harm gap for a clean upstream PR; tasked, not folded into A.2.
- **Stage-D note:** the duplicated T5 webhook-firing block in `scheduler.py` (event_loop_normal/overlap) is
  pre-existing daemon-plane code; it deletes (or extracts) in Stage D.

#### Round-3 review — #253 fix + dedup confirmed do-no-harm; 4 minor non-regressions

After fixing the above: **#253 (FIXED)** — `full_component._evict_keyfn` now branches on a single
`cache._aginfer_value_aware` flag (set in `_init_aginfer_eviction_scoring`): default path keys on
`eviction_strategy.get_priority` (honors `--radix-eviction-policy`), value path on the scorer. swa's
`_value_ordered()` reads the same flag. **Regression gate:** `verify/radix_eviction_policy` (12/12). The
**dead `import os`** and the **duplicated webhook block** (→ `_aginfer_maybe_fire_webhook()`) are also fixed.
Round-3 (3-lens) verdict: 2 PASS + 1 framing-only; do-no-harm proven at source + runtime —
`LRUStrategy.get_priority == last_access_time == _default_eviction_score`, so lru + value paths are
byte-identical no-ops; only non-LRU defaults change (the fix). 4 minor non-regressions, all dispositioned:
(1) swa value-heap is #248's code (framing); (2) #253 boundary already documented; (3) **MambaComponent +
SWA are LRU-list-only — STOCK behavior** (Mamba is byte-for-byte unchanged vs HEAD; honoring non-LRU there
would be a feature *beyond* stock, not do-no-harm); (4) `full_component` heap tuple `(key, n)` matches
**stock** `radix_cache.py` and is crash-safe via `UnifiedTreeNode.__lt__` — the swa `n.id` tiebreaker is
#248's own choice, not required here.

#### A.2 adversarial review (5-lens panel + per-finding verify) — 1 CRITICAL fixed, 2 cleanups, 1 mis-attribution

Ran an adversarial review of inc1–4 (5 independent lenses → verify each finding). Lenses B (imports/
circular), C (callsites), D (identity invariants) returned **CLEAN**. 5 findings, all triaged + resolved:

- **CRITICAL (FIXED + gated):** inc3's block boundary swept the trailing CLASS attribute
  `_AGINFER_VALID_STATES` into `cache_hooks.py` module scope while its consumer `set_aginfer_program_state`
  stayed inline → `AttributeError` on the program-state path. The 0-diff forward-proof validated the
  TRANSFORM but **not the BOUNDARY** (the swept line was a class attr, not part of `apply_aginfer_migrations`);
  compile/import can't catch runtime attribute resolution. FIX: restored `_AGINFER_VALID_STATES` as a class
  attribute (`unified_radix_cache.py:2635`), deleted the orphan. **Regression gate: t21** (exercises
  `set_aginfer_program_state` 17×) green + now in the standing set. (Lesson: also run the test for the
  methods left BEHIND adjacent to a cut, and grep the moved module for stray module-level `NAME =` — only
  `logger` should remain. Verified: both new modules have only `logger` at module scope.)
- **Minor (FIXED):** `cache_hooks.py` lacked `from __future__ import annotations` (bare `UnifiedTreeNode`
  in a local annotation — safe per PEP 526 but inconsistent with `metrics_hooks`) → added. 3 dead re-imports
  in the urc hook block (`_AGINFER_HINT_SCORER_SPEC`/`_AGINFER_BIRTH_LAMBDA`/`_AGINFER_BIRTH_STAMP` — no inline
  use, no external consumer) → deleted; kept `_AGINFER_BIRTH_PHAT` (read by verify/t27 via `_urc.`).
- **Mis-attribution (NO CHANGE):** the newborn birth-seed `p_hat` 1.0→`_AGINFER_BIRTH_PHAT`=0.1 is
  **PRE-EXISTING WIP** — the reviewer diffed against HEAD, which predates it; the inc1/inc2 scripts did only
  de-indent + self→X + byte-copy and so CANNOT have introduced it. do-no-harm floor intact (the seed fires
  only on the opt-in `aginfer:hint_v_u` path; env-unset = stock LRU).

Post-fix verify (all green): compile + full `import scheduler` + t21/t26/t27/t28/t38/s2_holder_scorer/
kv_scheduler_value_rule(25)/joint_decide. The 925-line, 3-file extraction stands clean + reviewed.

**Conclusion — re-sequence.** Because the remaining cache-method KEEP code is tangled with the
DELETE-in-Stage-D plumbing AND has no unit-level gate, the clean order is **Stage B (in-engine
`AginferScheduler` plugin) + Stage D (delete daemon transport) BEFORE the final cache-method extraction** —
deleting the ~1265 interleaved dump/state lines de-tangles the region, and Stage B/D need the live stack
anyway. All three remaining pieces share one gate: restart `dynamo.sglang` → do-no-harm (default == stock
LRU) + ours-engine **39/39 reproduced without the daemon**. That is the next focused, live-stack effort.
