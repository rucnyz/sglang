# T7 — kv_scheduler event handlers (paper §4 events → migrations)

## WHAT WE PROMISED

**Capability**
* For each paper §4 event arriving on the daemon's event queue,
  invoke `OursGreedyPolicy.decide(state, event_kind, decision_set)`
  using the **shared** policy from `baselines/ours_greedy.py`.
* `decision_set` is built per paper §4's table — for each event kind
  in ``daemon.events.EventKind`` (see audit round-1 M4 reconciliation
  below for the events not yet implemented):
  - `session_arrival`: shared platform / tool_def / subagent_ctx
    units (held by ≥ 2 programs; v1 heuristic until T3's typed
    metadata reaches the daemon)
  - `llm_prefill`: ``[]`` (informational; no migrate decision)
  - `tool_call_start`: caller's **exclusive** KV tail (units whose
    holders set is exactly the caller) — demote candidates
  - `tool_call_end`: caller's exclusive KV tail — promote candidates
  - `sub_dispatch_blocking`: parent's exclusive tail + shared prefix
  - `sub_dispatch_async`: shared prefix only
  - `memory_pressure` / `pressure_resolved`: top-k by regret
    (paper §7.1; ``_DEFAULT_MEMORY_PRESSURE_TOPK = 256``)
  - **Deferred to T10** (not yet in EventKind): `sub_return` (parent
    tail + child output), `session_end` (session-scoped GC).
* Translate `Action.assignments` → list of `(hash, target_tier)` →
  `POST /aginfer/migrate`.
* Lambda for a unit owned by program p in ACTING state = `1 /
  E[tool_duration]` (low); REASONING = baseline.

**Cost ceiling**
* `decide()` < 50 ms at 1 k units (regime we hit in Run F/F'/H').
* For `memory_pressure` event: decision_set capped at top-k (k = 256
  default), keeping decide() bounded regardless of tree size.
* `paper_§4_table()` helper that builds decision_set: < 5 ms per event.
* Total event-to-migrate latency contribution: ≤ 60 ms, fits in T5's
  80 ms budget.

## HOW WE VERIFY

Mechanism. `verify/t7_policy.py`:

```
1. Build a synthetic state with 4 programs:
   - shared 1 k-token "platform" prefix (high reuse, high p_hat)
   - per-program 4 k-token "session" tail (varying age)
   HBM cap deliberately too small to hold all session tails.
2. Set program_tracker:
   - p1, p2 REASONING (active)
   - p3, p4 ACTING (in tool call)
3. Replay paper §4 events through the event_worker and assert the
   migrate-set for each:
   - session_arrival(p5):  expect only platform/tool_def units
     in decision_set; targets all HBM.
   - tool_call_start(p1):  expect p1's session tail in decision_set;
     targets DRAM (low p_hat for tail when in tool call).
   - tool_call_end(p3):    expect p3's session tail in decision_set;
     targets HBM (it's resuming).
   - sub_dispatch_blocking(p2, child=p2c): expect p2's tail demoted,
     platform/tool_def kept on HBM.
   - memory_pressure (synthetic, occ=0.95): expect top-k by regret;
     assert k <= 256.
4. Wire dispatch to a stub /aginfer/migrate server; capture body;
   assert each event produces a migrate POST whose `actions` matches
   the expected Action.assignments translated to hash form.
5. Per-event timing: 10 runs each kind, 1 k units. Report mean.
6. State-construction adapter (audit Q4.2 fix): take a captured real
   /aginfer/state JSON, run `build_paper_state(state, events, tracker)`,
   feed result into `OursGreedyPolicy.decide()` *without modification*.
   Assert no exceptions and Action.assignments is non-empty.
```

## CALIBRATION

* `λ_u` for unit owned by program in ACTING state: default **1/5 s**
  = 0.2 (assume mean tool-call duration 5 s; swebenchpro's terminus-2
  ranges 1 s to 30 s, 5 s is mean).
* `λ_u` for REASONING: derived from observed `hits / age` as in the
  inline scorer (`baselines/sglang_adapter.py:_node_to_unit`).
* `π_u` and `h_τ` come from `baselines/costs.py` unchanged (shared
  with inline scorer + simulator).
* Sensitivity check: at λ_ACTING = 1/30 (very slow tool) or 1/1 (very
  fast tool), the demote/promote decisions on ACTING units should still
  point in the right direction (negative V → demote). Assert this in
  T7 verify with a parameter sweep.

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| All units high-V (nothing should be evicted) | Build state where every unit has high p_hat + low age | `decide()` returns empty Action; no migrate POST sent; no exception | dispatch stub captures 0 POSTs |
| Empty decision_set on memory_pressure | Set top-k to 0 in config | `decide()` returns empty Action; no migrate POST sent | same |
| Adversarial state: ACTING program has high-p_hat units | Program p1 in ACTING tracker but its units have hit_count=1000 (clearly hot) | `λ_ACTING` floor (0.2) wins; demote decision still fires; verify the math goes the intended way | inspect Action.assignments; assert p1's units have τ_target=DRAM |
| Mismatched λ_ACTING calibration | Set λ_ACTING=1.0 (1-second tool, very fast) | Should still demote (V_u still slightly negative for in-tool units, just less aggressively) | parameter sweep |
| λ_ACTING miscalibrated (audit #15, simulator-grounded) | Run `baselines.compare` with λ_ACTING ∈ {1/30, 1/5, 1/1, 2/1} on the seed-20260523 fixture; record `reward / total_runtime / hit%` for each | reward must stay ≥ LRU's reward (≈ Run F'-band) in all four settings; if it falls below LRU at the extremes, ACTING signal is over-trusted — clamp λ_ACTING ∈ [1/30, 1/1] in production | simulator runs report 4 numbers; compare against LRU baseline 42.7 |
## RESULTS

**PASSED** — all 13 verify steps + (9 round-1 + 4 round-2 + 6 round-3 +
1 round-3.5) bisect demos in regression_probe.py, ~15 s on the
agsched env.

### Audit round-3.5: dead-defensive-code cleanup

After round-3 the user asked "are there more meaningless fallbacks?"
A scan turned up one real B1-class bug AND a pile of dead null-defense
that was preventing fail-loud diagnosis.  Per user direction
**"raise > silent failure"**, all fallbacks were removed:

| ID | Finding | Fix |
|---|---|---|
| **R3.5-1** | `raw.get("tier", "HBM")` defaulted MISSING tier → HBM (B1's silent-mis-classification bug, just for a different trigger) | `raw.get("tier", "")` → empty string flows into `_tier_from_string` → returns None → unit skipped + logged (same path as unknown labels). |
| **R3.5-2** | `.get(x, {}) or {}` / `.get(x, []) or []` / `.get(x, 0) or 0` defensive null-guards on fields sglang always emits with proper types | Direct `state_json["tier_usage"]` / `raw["n_tokens"]` etc.  Missing field → KeyError → propagates to handle()'s try/except → log + bow out. |
| **R3.5-3** | `unknown_tier_log: Optional[set] = None` had `if not None:` branch that silently skipped logging when unset | Made required positional kwarg.  All callers pass an explicit set (KvScheduler instance attr in production, `set()` in tests). |
| **R3.5-4** | `now_counter` parameter had multi-tier fallback (arg → state field → max(last_access)+1) | Removed; trust `state["time_counter"]` (sglang emits it). |
| **R3.5-5** | `float(lambda_acting)` redundant cast in `KvScheduler.__init__` (already float-typed) | Dropped. |
| **R3.5-6** | `_flatten_per_rank`'s `isinstance(rank, dict): continue` and similar guards on sglang-emitted containers | Removed; trust shape. |

Net effect on `daemon/kv_scheduler.py`: **-13 LoC** (code is shorter
+ failure modes are now visible in logs).

### Audit round-3 findings + bisect-style fixes

The third audit was a **test-quality audit** (are tests fake / vacuous
/ missing depth) rather than a code-quality audit.  Found 1 FAKE +
3 VACUOUS + 2 MAJOR depth + 2 MINOR depth.  One depth finding
(DEPTH-1) initially looked like a prod bug but the right fix was
to SIMPLIFY: remove the `_BYTES_PER_TOKEN` env var + `n_bytes`
fallback entirely (sglang's state always emits `n_bytes` directly,
so the fallback was dead defensive code).

| ID | Finding | Fix layer |
|---|---|---|
| **R3-DEPTH-1** | `n_bytes` fallback via env var was load-bearing only if sglang's state dropped the field; in practice it's always emitted, so the fallback was dead code that masked a state-emission regression | **Production simplify**: removed `_BYTES_PER_TOKEN` env var + fallback entirely; `n_bytes = int(raw.get("n_bytes", 0) or 0)`.  Probe pins state's n_bytes flows through verbatim. |
| **R3-DEPTH-5** | Paper §9 "fresh state per event" / no-caching contract was unpinned — a regression caching fetch_state across events would slip past every existing test | **Test only** (prod was correct): two-event probe that mutates state between emits and asserts the second decision reflects the new state. |
| **R3-FAKE-1** | `probe_b1` monkey-patches `_tier_from_string` but not the `build_paper_state` skip-None branch; a parallel regression deleting only the skip branch would slip | **Test**: documented sub-probe NOTE acknowledging that orthogonal pin requires composition with the B1 primary (skip-None is reachable only when helper returns None). |
| **R3-VACUOUS-1** | `step_all_event_kinds_registered` used bound-method `==` which is fragile (functools.partial wrapper would falsely fail) | **Test**: switched to FUNCTIONAL pin — fire one event per EventKind, assert `last_decision_set_size` sentinel was overwritten (proves scheduler.handle, not _noop_handler, ran). |
| **R3-VACUOUS-2** | `AGINFER_LAMBDA_ACTING=not_a_float` crashed module import with a vague `ValueError: could not convert` (no mention of env var name) | **Production**: added `_env_float` / `_env_int` helpers that re-raise with the env var name in the message. |
| **R3-VACUOUS-3** | Round-2's probe_b2 bug variant (`lambda j: j`) was equivalent to the production early-return on non-per_rank, so it pinned "triggered when present" not "aggregation result valid" | **Test**: new probe with a bug that returns correct shape but empty units (over-eager dedupe regression class). |
| **R3-DEPTH-3** | Malformed unit fields (`n_tokens=None`, `tier=""`) not pinned | **Test**: new probe feeding a state with three malformed units; assert (a) `u-ok` survives, (b) `u-empty-tier` is skipped (UnknownTierError caught), (c) `u-bad-tokens` is included with n_tokens=0 defaults. |
| **R3-DEPTH-4** | `fetch_state` returning JSON `null` (non-dict) not pinned | **Test**: new probe with stub returning `null` first then valid state; assert worker survives, `handler_failures == 0`. |

### Audit round-2 findings + bisect-style fixes

The second-pass audit on the round-1 code found 1 BLOCKER + 2 MAJORs
+ 3 MINORs + 2 NITs.  Notably, the BLOCKER (R2-B1) was a **fix-
introduced bug** in round-1 itself: the per_rank hash prefix would
never resolve on sglang's side and migrations would silently no-op.
All fixes follow the same protocol (probe re-injects regression →
new assertion FAILS → fix applied → assertion PASSES); raw logs in
`results/<...>_r2*.log`.

| ID | Finding | Fix layer |
|---|---|---|
| **R2-B1** | Round-1's `_flatten_per_rank` prefixed hashes with `rN/`; sglang's exact-hash lookup would miss every action → 200 OK with empty `applied_hashes`, daemon never knows | **Production**: drop the prefix (sglang hashes are globally unique); dedupe via `seen_hashes` set; broadcast migrate handles replicated prefixes. New probe round-trips through a stub that mimics sglang's exact-hash lookup. |
| **R2-M1** | ACTING-floor λ only fired for `State.ACTING`; PAUSED programs (admission_controller pinned mid-tool-call) fell back to `hits/age` proxy — exact opposite of paper §7 intent | **Production**: `if st in (State.ACTING, State.PAUSED)` |
| **R2-M2** | Round-1's `probe_n3` was a tautology: subprocess literally read a renamed env key without importing `kv_scheduler` at all | **Test**: rewritten to lay down a shadow copy of `daemon/` + `baselines/` with the env var key sed-replaced; subprocess imports the shadow; observes that the OLD env var no longer drives the constant. |
| **R2-N1** | `_units_for_session` had a redundant `holders == [s] or set(holders) == {s}` — set form was already correct, but a future revert to list-only would silently miss duplicate-holder edge cases | **Production**: use set semantics exclusively (paper meaning). Probe demonstrates list-only would miss `["s","s"]`. |
| **R2-N2** | `_logged_unknown_tiers` was a module-global set → test #1 triggering the warn suppressed it for all subsequent tests in the same process | **Production**: moved to `KvScheduler._unknown_tier_log` instance attribute; `build_paper_state` takes optional `unknown_tier_log` param. Daemon restart naturally resets. |
| **R2-N3** | `regression_probe.py`'s `run_server` lacked the startup-failure guard `verify.py` has → silent vacuous passes if uvicorn fails to bind | **Test**: mirror `verify.py:73`'s `raise RuntimeError`. |
| **R2-X1** | `probe_n4` built its own fixture rather than calling `step_idempotent_repeat_event` (stand-in drift risk) | **Doc only**: explicit STAND-IN docstring; refactor would require invasive surgery on verify module. |
| **R2-X2** | `probe_m2` had `endswith("u-tail-2")` in one path and `== "u-tail-2"` in another | **Test**: aligned to `==` (no `rN/` prefix after R2-B1 fix). |

### Audit round-1 findings + bisect-style fixes

The audit-of-audit subagent caught 2 BLOCKERs + 4 MAJORs + 7 MINORs.
For each, we (a) wrote a test that exposes the regression, (b) ran
it against the buggy code → FAIL, (c) applied the fix, (d) re-ran
→ PASS.  All bisect demos live in `regression_probe.py`; raw logs
in `results/<...>_regression_probe.log`.

| ID | Finding | Fix layer |
|---|---|---|
| **B1** | `_tier_from_string("ZSTD_DISK")` silently fell back to `Tier.HBM` — any new sglang tier label mis-classified | **Production code**: `_tier_from_string` now returns `Optional[Tier]`; unknown labels are skipped + logged once per label. |
| **B2** | Multi-rank `/aginfer/state` (`{"per_rank":[...]}`) was silently zero'd — DP > 1 deployments saw no units / no decisions | **Production code**: new `_flatten_per_rank` aggregator (sum used/cap, concat units prefixed with `rN/`). |
| **M1+N7** | Top-k bound was pinned only by length; a sort-direction flip (`items[-k:]`) would invert the policy and pass | **Test**: step [2] now plants 10 low-V sentinels among 9 990 high-V fillers; asserts all sentinels appear in the returned 256. |
| **M2** | Step [3] accepted `target_tier != "HBM"` which allows DROP (catastrophic) | **Test**: tightened to `in ("DRAM","DISK")`. |
| **M3** | `PRESSURE_RESOLVED` was never fired in verify; a missing registration in `attach_kv_scheduler` would fall back to `_noop_handler` silently | **Test**: new step [11] iterates `EventKind`, asserts each value maps to `scheduler.handle`; also fires a real PRESSURE_RESOLVED end-to-end. |
| **M4** | README §WHAT WE PROMISED listed `sub_return` / `session_end` but enum has neither | **Doc**: reconciled bullet list to match `EventKind`; deferred sub_return / session_end to T10. |
| **N1** | λ sweep tested ceiling clamp but not floor | **Test**: added λ=1/100 to the sweep; assert action set equals λ=1/30. |
| **N2** | `_dispatch_migrate` 5xx path (log + continue) was unpinned | **Test**: new step [12] stub returns 500; assert `handler_failures == 0` AND worker keeps draining. |
| **N3** | `AGINFER_*` env vars → module constants binding was unpinned (rename would silently strand operators on default) | **Test**: new step [13] subprocess-probes the binding for all three env vars. |
| **N4** | Step [9] `if migrate_calls:` was self-skipping if the policy declined to migrate | **Test**: fixture forces a migrate (tight HBM cap + sentinels); assertion is now `len(migrate_calls) == 3`. |
| **N5** | decide() ceiling 50 ms vs actual ~1.4 ms (33× headroom) | **Test**: tightened to `mean+3σ < 5 ms` (still ~3× headroom; catches an O(N²) re-introduction). |
| **N6** | `_top_k_by_regret` docstring contradicted the slice ("near the bottom" while `items[:k]` returns the top) | **Doc**: rewrote docstring to match the slice and warn future maintainers off the `items[-k:]` "fix". |

### Verify summary

* date: 2026-05-26
* daemon code: ~480 LoC `daemon/kv_scheduler.py` (incl. audit round-1
  fixes); reuses `baselines/ours_greedy.py` + `baselines/costs.py`
  unchanged.
* per-event decision_set correct (paper §4 table, 6 event kinds):
  ✓ [1] — SESSION_ARRIVAL → shared platform/tool_def units only;
  LLM_PREFILL → []; TOOL_CALL_START/END → caller's exclusive tail
  (NOT shared prefix); SUB_DISPATCH_BLOCKING → tail + shared;
  SUB_DISPATCH_ASYNC → shared only.
* memory_pressure top-k bounded AND content-pinned: ✓ [2] —
  10 000-unit state → decision_set has exactly 256 units AND
  contains the 10 planted low-V sentinels (was: length only).
* state-construction adapter feeds OursGreedyPolicy unmodified: ✓
  [1] [3] — `build_paper_state` produces a `SchedulerState` that
  `OursGreedyPolicy.decide()` consumes without any test-only shims.
* event → state-fetch → decide() → migrate dispatch with correct
  direction: ✓ [3] — body's `target_tier` is DRAM/DISK, NEVER DROP.
* WORST CASE — nothing worth moving → 0 migrate POSTs: ✓ [4].
* WORST CASE — /aginfer/state 500 → log + continue: ✓ [5].
* λ_ACTING calibration sweep {1/100, 1/30, 1/5, 1/1, 2/1}: ✓ [7] —
  both ceiling AND floor clamps saturate; no DROP migration in any
  in-envelope λ.
* assignments_to_wire schema: ✓ [8].
* paper §9 idempotence (forced 3× migrate, all bodies identical): ✓ [9].
* all 8 EventKinds routed to kv_scheduler (incl. PRESSURE_RESOLVED): ✓ [11].
* migrate 5xx log+continue (handler_failures stays 0): ✓ [12].
* AGINFER_* env vars bind to module constants: ✓ [13].

### Latency (multi-run, per memory:feedback-latency-multi-run)

5 independent trials at 1 000 units in the synthetic state.

| stage                  | mean ± std       | budget (audit round-1) |
|---|---|---|
| `build_paper_state`    | 2.92 ± 0.31 ms   | < 5 ms                 |
| `OursGreedyPolicy.decide()` | 1.40 ± 0.04 ms | < 5 ms (was 50 ms; tightened per N5) |

Assertion: `mean + 3σ < 5 ms` for both stages.  Current envelopes
≈ 3.9 ms / 1.5 ms.

* raw logs (relative to this directory):
  * `results/20260526_PREFIX_audit1_baseline.log` — pre-round-1
    baseline (10 steps; loose floors)
  * `results/<YYYYMMDD_HHMMSS>_run3_audit1.log` — post round-1
    (13 steps; tightened floors + B1/B2 prod fixes)
  * `results/<YYYYMMDD_HHMMSS>_run4_r2.log` — post round-2
    (13 steps; B1/B2 fix corrected to drop hash prefix; PAUSED in
    ACTING-floor; set-only `_units_for_session`; instance log set)
  * `results/<YYYYMMDD_HHMMSS>_r2b1_PRE_fix.log` — R2-B1 pre-fix
    demonstrating `applied_hashes` empty (multi-rank routes broken)
  * `results/<YYYYMMDD_HHMMSS>_r2b1_POST_fix.log` — R2-B1 post-fix
  * `results/<YYYYMMDD_HHMMSS>_r2m1_PRE_fix2.log` — R2-M1 pre-fix
    (PAUSED λ ≠ 0.2)
  * `results/<YYYYMMDD_HHMMSS>_r2m1_POST_fix.log` — R2-M1 post-fix
  * `results/<YYYYMMDD_HHMMSS>_r2n2_POST_fix.log` — R2-N2 final state
    (all probes pass)
  * `results/<YYYYMMDD_HHMMSS>_regression_probe.log` — round-1 only
    (9 bisect demos)
  * `results/<YYYYMMDD_HHMMSS>_regression_probe_r2.log` — round-1
    + round-2 (13 bisect demos)
