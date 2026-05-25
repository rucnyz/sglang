# T7 — kv_scheduler event handlers (paper §4 events → migrations)

## WHAT WE PROMISED

**Capability**
* For each paper §4 event arriving on the daemon's event queue,
  invoke `OursGreedyPolicy.decide(state, event_kind, decision_set)`
  using the **shared** policy from `baselines/ours_greedy.py`.
* `decision_set` is built per paper §4's table — for each event kind:
  - `session_arrival`: platform / tool_def / subagent_ctx units on
    the new session's prefix path
  - `tool_call_start`: caller's KV tail (demote candidates)
  - `tool_call_end`: caller's KV tail (promote candidates)
  - `sub_dispatch_blocking`: parent tail + shared platform/tool_def
  - `sub_dispatch_async`: shared platform/tool_def only
  - `sub_return`: parent tail + child output
  - `session_end`: session-scoped units
  - `memory_pressure`: top-k by regret (paper §7.1)
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
* date: _pending_
* daemon sha:
* per-event decision_set correct: _pending_
* ACTING units demoted, REASONING kept: _pending_
* memory_pressure top-k bounded: _pending_
* state-construction adapter feeds unmodified `OursGreedyPolicy.decide`: _pending_
* decide() mean time @ 1k units: _pending_
* event-to-migrate p99: _pending_
* raw log: `verify/results/t7_<datetime>.log`
