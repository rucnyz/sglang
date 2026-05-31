# T43 — `fatal(reason, **context)` helper (DESIGN §10)

PLAN.md §4 T43.  Shared entry point for every **deployment-bug-class
halt** in the daemon.  Defines the two-fault-class split (DESIGN §10):

* **deployment-bug** — schema mismatch, missing required state fields,
  joint_decide infeasibility, `peak_bw_bps ≤ 0`, mode-switch attempt,
  hash collision.  "This should never happen in a correct deployment."
  → `fatal(...)` dumps forensic state and `sys.exit(1)`.  Supervisor
  decides restart policy.
* **load** — `apply_failed` race, sglang briefly slow, transient
  outbound queue depth.  "This is just how the system handles bursty
  workload."  → log + continue.

T43 is the **pre-req for all other fail-fast work** (T22, T34, T36/T37,
HASH_COLLISION) — those landers convert their bug-class halts to
`fatal(...)` instead of inventing per-site dump logic.

## WHAT WE PROMISED

**API.**

```python
from daemon._fatal import fatal

fatal(
    "joint_decide_infeasible",
    event=event,
    state=state,
    candidates=items,
    dp_inputs={"bytes_needed": bytes_needed, "cap_left": cap_left,
               "bucket_size": bucket_size, "dp_size": len(dp)},
)
# fatal() never returns.
```

**Effects** (in order):

1. Resolves `<daemon-data>` via `$AGINFER_DATA_DIR` env var (default
   `<sglang>/dev/aginfer/data`).
2. Writes `<daemon-data>/forensic/<reason>_<ns_ts>_<pid>.json`
   containing:
   * `reason` — the supplied slug
   * `timestamp_unix` / `timestamp_iso` — when the fatal fired
   * `pid` — process id (multi-rank concurrent fatals don't clobber)
   * `traceback` — `traceback.format_stack()` if no exception in
     flight, else `format_exception()`.  Captures the call site
     even for `assert`-style halts.
   * `context` — every `**kwarg` recursively JSON-coerced.
     `dataclasses.asdict()` for dataclasses, `.value` for Enums,
     `model_dump()` for pydantic, `repr(...)` as the final fallback
     so unserialisable fields (e.g. live sockets) **do not abort
     the dump**.
3. Logs a `CRITICAL` line via `logger("aginfer.daemon.fatal")`
   naming the file path:
   `FATAL reason=<reason> forensic_file=<path> pid=<pid>`
4. `sys.exit(1)`.

**Never raises** — forensic preservation under bug conditions is the
whole point.  If the data dir is unwritable, the helper still logs
`CRITICAL` (with the full payload in the message) and exits 1.  If a
single context field fails to serialise, that field is replaced with
`repr(value)` and the rest of the dump proceeds.

## CALL SITES

T43 itself converts the existing-code instances on the deployment-bug
list (PLAN T43 enumerates seven; the rest land with the features that
introduce them):

| Site                                                | Reason slug                       | Status |
|---|---|---|
| `_flatten_per_rank` — empty `per_rank` list          | `per_rank_empty`                   | ✓ T43 |
| `_flatten_per_rank` — cross-rank subpool key mismatch | `subpool_key_mismatch_across_ranks` | ✓ T43 |
| `_flatten_per_rank` — cross-rank `n_bytes` disagreement (DESIGN §6 L736) | `n_bytes_disagreement_across_ranks` | ✓ T43 (audit-driven) |
| `build_paper_state` — `unsupported_tree_cache`       | `unsupported_tree_cache`           | ✓ T43 |
| `build_paper_state` — missing required field          | `missing_state_field`              | ✓ T43 |
| `build_paper_state` — `peak_bw_bps ≤ 0`               | `peak_bw_bps_non_positive`         | ✓ T43 |
| `build_paper_state` — `h_max_per_byte_sec ≤ 0` (DESIGN §10 L2319) | `holding_cost_non_positive`        | ✓ T43 (audit-driven) |
| `build_paper_state` — `prefill_bps ≤ 0` once prefill has run (DESIGN §10 L2319) | `prefill_bps_non_positive_with_traffic` | ✓ T43 (audit-driven) |
| `joint_decide` infeasibility                          | `joint_decide_infeasible`          | T34 (#156) |
| `bytes_at(τ)` — τ not in residence                    | `bytes_at_tier_not_in_residence`   | T34 (#156) |
| `HASH_COLLISION` webhook receipt                      | `hash_collision`                   | T23+T37 (#153) |
| Daemon-attached mode losing daemon mid-run            | `daemon_attach_lost`               | (deferred) |

The deferred sites do not exist as code yet; T43 only owns the helper
+ the five sites that already had `raise ValueError(...)` or were
missing a check.

## WORST CASE

| Failure mode                                       | How to force | Predicted floor | Assertion |
|---|---|---|---|
| `$AGINFER_DATA_DIR` unwritable                     | chmod 000 the dir | helper logs CRITICAL with payload-in-message + exit 1; no traceback lost | (manual) |
| Unserialisable context value (live socket, etc.)   | pass `socket.socket()` as a kwarg | helper falls back to `repr(value)` for that field; other fields preserved | Stage 2 |
| Two fatals fire concurrently from different ranks  | (synthetic) | distinct files via `_<ns>_<pid>.json` suffix; no clobber | (filename design) |
| Fatal fires inside an active exception             | `try/except` then call fatal | `traceback` field captures the exception, not just the call stack | (manual; helper uses `sys.exc_info()`) |

## HOW WE VERIFY

`verify/t43/verify.py` runs entirely in-process for the harness logic
and spawns subprocesses to exercise each fatal path (since `fatal()`
terminates the interpreter):

```
Stage 0  fatal helper contract
         subprocess calls fatal('schema_sanity', foo=1, bar=[1,2,3]);
         expect exit=1, <data_dir>/forensic/schema_sanity_*.json
         exists with the contract keys, CRITICAL line on stderr.

Stage 1  traceback captured (no exception)
         subprocess calls fatal('tb_check') from outer_frame →
         inner_frame; expect both frame names present in
         payload.traceback.

Stage 2  unserialisable context falls back to repr
         pass socket.socket() as a context field; expect exit=1,
         file lands, the bad field becomes repr() but a sibling
         JSON-safe field is preserved.

Stage 3  cross-rank subpool key mismatch
         per_rank=[rank0={attn}, rank1={attn, moe_expert}];
         expect fatal('subpool_key_mismatch_across_ranks').

Stage 4  peak_bw_bps non-positive
         HBM->DRAM peak_bw_bps=0 in a single-rank state; expect
         fatal('peak_bw_bps_non_positive').

Stage 5  missing throughput_ema field
         delete throughput_ema; expect fatal('missing_state_field')
         with context.missing == 'throughput_ema'.

Stage 6  per_rank empty list
         {per_rank: []}; expect fatal('per_rank_empty').

Stage 7  unsupported_tree_cache field
         set state.unsupported_tree_cache='HiRadixCache'; expect
         fatal('unsupported_tree_cache').

Stage 8  happy-path sanity (no fatal)
         the seed-valid state runs through build_paper_state to
         completion with exit=0 and no forensic file written.
         Catches the over-tightening regression: the new checks
         must not bleed into the green path.

Stage 9  h_max_per_byte_sec non-positive (audit-driven)
         tier_holding_cost.HBM.attn.h_max_per_byte_sec = 0 → fatal
         (sub-stage: -1e-9 also → fatal, defends ``>= 0`` regression).

Stage 10 prefill_bps positivity (conditional, audit-driven)
         Three sub-cases proving the "(once any prefill has run)"
         qualifier from DESIGN §10:
         (a) startup (prefill_bps=0, units=[], time_counter=0) →
             NO fatal (legitimate cold-start state).
         (b) prefill_bps=-1 (negative) → fatal regardless
             (negative throughput is structurally nonsense).
         (c) prefill_bps=0 + units present + time_counter>0 → fatal
             (we have evidence prefill ran; EMA reporting 0 is a bug).

Stage 11 cross-rank n_bytes disagreement (DESIGN §6 L736, audit-driven)
         Two-rank state, same hash, n_bytes[HBM][attn]=4096 on
         rank-0 vs 8192 on rank-1 → fatal.  Pre-audit T43 took
         max() over disagreeing values, silently absorbing the bug;
         DESIGN explicitly flags this as bug-class.
```

## REPRODUCING

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh && conda activate agsched
cd /scratch/yuzhou/projects/sglang

python dev/aginfer/verify/t43/verify.py
```

No sglang launch required — the helper is daemon-side and all stages
run via Python subprocesses.

## RESULTS

**PASSED** — all 12 stages (initial 9 + audit-driven 3 for DESIGN
§10 / §6 alignment).

* date: 2026-05-31
* lines: ~190 in new `daemon/_fatal.py`; ~95 net in
  `daemon/kv_scheduler.py` (3 `raise ValueError` → `fatal(...)`
  conversions + 5 new checks: per-link `peak_bw_bps > 0`, top-level
  field-presence guard, h_max positivity, prefill_bps positivity,
  cross-rank n_bytes equality).

| Stage | Result |
|---|---|
| 0  fatal helper contract | PASS |
| 1  traceback captured (no exception) | PASS — `outer_frame_name` / `inner_frame_name` both present |
| 2  unserialisable context falls back to repr | PASS — `socket` repr stored, sibling JSON-safe field preserved |
| 3  cross-rank subpool key mismatch | PASS — `subpool_key_mismatch_across_ranks` |
| 4  peak_bw_bps non-positive | PASS — `peak_bw_bps_non_positive` |
| 5  missing throughput_ema field | PASS — `missing_state_field` |
| 6  per_rank empty list | PASS — `per_rank_empty` |
| 7  unsupported_tree_cache field | PASS — `unsupported_tree_cache` |
| 8  happy-path sanity (no fatal) | PASS — exit=0, no forensic file |
| 9  h_max_per_byte_sec non-positive | PASS — `holding_cost_non_positive` (0 AND -1e-9 both fatal) |
| 10 prefill_bps positivity (conditional) | PASS — (a) startup exits 0, (b) -1 fatals, (c) 0+units fatals |
| 11 cross-rank n_bytes disagreement | PASS — `n_bytes_disagreement_across_ranks` |

* raw run log: `results/20260531_t43_initial_pass.log` (initial 9)
* audit re-run log: `results/20260531_t43_audit_aligned_pass.log` (12 stages)

### Known stale-test impact

The new field-presence guard (`missing_state_field`) tightens
`build_paper_state`'s entry contract.  Two verify modules still use
pre-T17 stub-states (`tier_usage`-keyed, flat `tier`, integer
`n_bytes`) and now trip `fatal()` at the first call:

* `verify/kv_scheduler_value_rule/` — covered by pending task #146
  "full rewrite for post-T33 contract"
* `verify/admission_controller/regression_probe.py` — same #146 scope

Those probes were already non-functional against the post-T17 schema;
the T43 guard just makes the failure loud.  Fixing them is out of
scope for T43 (they will be rebuilt under #146 with the new
residence-set candidate generator path).
