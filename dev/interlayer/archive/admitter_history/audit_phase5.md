# Phase 5 — scheduler hook + JSONL log: audit + gap fixes

Combined test-depth + implementation audit (subagent). Reviewed:
- `python/sglang/srt/budgeter/admitter.py` (decide_for_req, _log_decision, close, ctor JSONL setup)
- `python/sglang/srt/managers/scheduler.py` (Admitter construction at 1010-1040, hook at 2231-2253)
- `dev/interlayer/admitter/test_scheduler_hook.py` (initial 8 tests)

## Severity summary

| # | Concern | Severity | Status |
|---|---|---|---|
| 1 | `admitter.close()` never called at shutdown | HIGH | ✅ Fixed (atexit.register) |
| 2 | `SGLANG_ADMITTER=0` construction gate not tested | MEDIUM | ✅ test_10 |
| 3 | `SGLANG_ADMITTER_CROSS_FIRE` not tested | MEDIUM | ✅ test_11 |
| 4 | Hook placement (NULL-disagg branch) not verified | MEDIUM | ✅ test_12 (source-grep) |
| 5 | JSONL grows unbounded | MEDIUM | ✅ Docstring note (logrotate compatible) |
| 6 | Log mode `"a"` + multi-process PID | LOW | (deferred — JSONL atomicity holds) |
| 7 | Empty / None `origin_input_ids` untested | LOW | ✅ test_13 |
| 8 | JSONL schema candidate-set drift untested | LOW | ✅ test_14 (strict set equality) |
| 9 | `inf → None` JSON convention undocumented | LOW | ✅ Comment in _log_decision |
| 10-18 | Exception swallowing / lookup / staleness / etc. | OK | (no fix needed; rationale per audit) |

## Fixes landed

### HIGH #1 — atexit.register(admitter.close)

`scheduler.py` Admitter construction now does
`atexit.register(self.admitter.close)` so JSONL is flushed + handle
closed on process exit. Scheduler has no explicit shutdown today
(Budgeter has the same pattern of lifetime-tied-to-process); atexit
is the right minimum.

Test 9 verifies `close()` flushes pending entries and is idempotent.

### MEDIUM #2 — Admitter construction gate test

`test_10_admitter_construction_gate`: parameterized over
{None, "0", "false", ""} → admitter not constructed; "1" → constructed.
Mirrors the exact gate logic in `scheduler.py:1019`.

### MEDIUM #3 — Cross-fire env gate test

`test_11_cross_fire_env_gate`: parameterized over {"1", "0", None}
→ asserts the env value threads through to `Admitter.cross_fire_enabled`.

### MEDIUM #4 — Hook placement source-grep

`test_12_hook_inside_null_disagg_branch`: parses
`_add_request_to_queue` source, asserts `admitter.decide_for_req` is
INSIDE the `DisaggregationMode.NULL` branch (between the NULL `if`
and the next `elif`). Asserts it does NOT appear in the
PREFILL/DECODE branches. A refactor that mis-locates the hook would
fail this test.

### MEDIUM #5 — JSONL rotation docstring

`Admitter.__init__` docstring now explicitly notes:
- Mode is `"a"` (append) so logrotate works
- Lines are well under PIPE_BUF → atomic across processes
- Operators are expected to use logrotate; no built-in rotation

### LOW #7 — Empty input

`test_13_empty_input_doesnt_crash`: req.origin_input_ids = [] AND
= None both yield x_tokens=0 and own_free (sensible degenerate
case).

### LOW #8 — Strict JSONL candidate set

`test_14_jsonl_candidate_set_is_exactly_five`: asserts
`set(entry["candidate_costs_us"].keys()) == {own_free, own_evict,
cross_free, cross_evict, defer}`. A refactor that adds a new
candidate without updating downstream parsers fails this test.

### LOW #9 — Inf→None convention

`_log_decision` now has an explicit comment:
> JSON has no Infinity; map infeasible candidates to JSON `null`.
> Consumers should treat null as "infeasible / cost = +inf".

## Deferred (rationale)

- **#6 PID in filename**: JSONL line atomicity holds (PIPE_BUF >
  line size). Operators can grep by source process if needed.
  Optional `pid` field per entry is a nice-to-have, not a bug.
- **#10-18 (audit's OK section)**: zero-overhead-when-disabled,
  exception swallowing, mamba pool fallback, leak detector
  compatibility, log-level placement, env-at-construction binding,
  concurrent JSONL writes — all confirmed correct by audit.

## Phase 5 final status

**14/14 tests PASS.** Subagent verdict pre-fixes: "Ship after fixes."
All HIGH + MEDIUM gaps addressed.

| Test | What it pins |
|---|---|
| 1 | decide_for_req derives kv_free / evictable / Q / x_tokens from scheduler |
| 2 | JSONL log records every decision with full schema |
| 3 | SGLANG_ADMITTER_LOG unset → no file, no crash |
| 4 | P99 decide_for_req < 100 µs (measured 2.4 µs over 10⁴ arrivals) |
| 5 | Non-NULL disagg mode → decide_for_req returns None |
| 6 | JSONL captures fire_result fields when set |
| 7 | Cold-start picks own_free when KV has capacity |
| 8 | x_tokens derived from len(req.origin_input_ids) |
| 9 | close() flushes log + idempotent |
| 10 | SGLANG_ADMITTER construction gate over {None,0,false,'',1} |
| 11 | SGLANG_ADMITTER_CROSS_FIRE threads through to ctor kw |
| 12 | Hook is inside NULL-disagg branch only (source-grep) |
| 13 | Empty / None origin_input_ids → x_tokens=0 → own_free |
| 14 | JSONL candidate keys == {own_*, cross_*, defer} exactly |

## Performance

- **P99 decide_for_req**: 2.4 µs over 10⁴ arrivals (median 2.3 µs).
  Budget was 100 µs; achieved 40× headroom.
- **Hot path when disabled**: one `is not None` check before the
  hook → ~5 ns. Effectively free.

## Verdict

**Ship.** All audit gaps addressed; 14/14 tests pass; 76/76 across all
five Admitter phases + dyn_admission_cap regression suite. Phase 6
(D6 / D6n / D3 live workload tests) unblocked.
