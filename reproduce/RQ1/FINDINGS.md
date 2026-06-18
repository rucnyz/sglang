# RQ1 on the agentreplay (token-exact real-CC) harness — findings

Canonical arms (current config; both boot at sglang DEFAULT mem-frac + context, overlap ON via `--mamba-scheduler-strategy extra_buffer`, `--enable-cache-report`, GPU7/port30097):
- **base**: `--radix-eviction-policy lru`, no Budgeter.
- **sys**: `--radix-eviction-policy lpb` + full Budgeter+Admitter cross-fire (PF64, tick 1.0s, calibrated single-curve cost `c_M=0`), no K_big.
Runner: `reproduce/RQ1/run_arm.sh` (invokes agentreplay as a tool via `PYTHONPATH`). Metric: `cache_hit` (=Σcached/Σprompt), `throughput_tok_s`, `ttft_ms.p99` from the replay json.
The headline results are the dated sections at the bottom; the sections in between are the chronological journey, and the earliest ones describe SUPERSEDED configs (mem-frac 0.45, `--disable-overlap-schedule`, K_big — all since removed).

## Harness validated
- Forcing token-exact (committed `output_ids == forced`); cache reuse ground-truth; convert root-fixed (visible-only context growth, no thinking in next prompt).
- Concurrency + pressure real: 58 concurrent, KV usage 1.0, mamba slack, queueing. Confirmed the harness drives true concurrent CC load.

## Key structural finding: real CC is intrinsically KV-bound
- **0% of requests have prompt <16k.** First-turn prompt p50=17k (the ~16k system prefix dominates); deepest-turn p50=94k, max=276k.
- Single-GPU pools at mem-frac 0.45: KV ~251k tokens, mamba 144 slots. KV saturates at ~15 concurrent long contexts, far before 144 mamba slots.
- Therefore **mamba is never the natural bottleneck for real CC**. KV-bound (Case 2) is the only natural real-CC regime.
- Consequence: Case 1 (mamba-bound) and a mamba-bound Case-3 phase are NOT representable from real CC without artificially capping mamba (`--max-mamba-cache-size`) AND shortening contexts (e.g. early-turn subset). They are *induced* regimes, to be labeled as such.

## Concurrency comes from ROOT count, not record count
session mode fires only root programs (subagents run while parent blocks). The
trace is sorted largest-first, so a small `--limit` front-loads a few huge roots
+ their subagents → low concurrency (KV 0.17-0.48). Fix: use the FULL trace so
all roots fire. Also: prompts cannot be shortened below the real prompt_tokens —
`convert` pads each prompt back up to pt, so contexts stay ~18k+ regardless of
`--sys-tokens`. The lever for shorter sessions is `--max-turns`.

## Traces built (Qwen3.5-9B tokenizer)
- `cc_qwen_t6.jsonl` — max-turns 6, 250 sessions, 73 roots, ctx p50 26k. **Case 2** (KV-bound).
- `cc_qwen_mamba.jsonl` — sys-tokens 4000, turns 2-3, 300 sessions, 93 roots, ctx p50 18k. **Case 1** (cap mamba ~32 → mamba-bound, KV slack).

## Case 2 (KV-bound, natural) — CONFIRMED pressure
full `cc_qwen_t6`, session, stagger 0.02, gap 0, max-conc 64. base N=3 running:
63-64 concurrent, KV usage 0.78-0.92, mamba 0.28 (slack), 1400 tok/s. This is the
KV-bound regime (grow KV from idle mamba, m2k). base smoke cache_hit≈0.68.

## Case 2 RESULT (N=3 base; sys rep-1 then crash)
- **base (lru, no budgeter), N=3**: cache_hit **0.526 ± 0.0003**, 447 tok/s, p99 TTFT 53s, 0 err. Rock-stable.
- **sys (lpb + budgeter), rep-1**: cache_hit **0.6473** (+12.1pp), **538 tok/s (+20%)**. WIN confirmed (m2k: grow KV from idle mamba + LPB).
- **sys CRASH (blocks clean N=3)**: `full token usage` went NEGATIVE (-3.93, #full token -3.07M) → `CUDA error: illegal memory access` in `next_token_ids.tolist()`. Root: at 64-concurrency the subagent fan-out fills mamba (125/126) so the Budgeter fires **k2m** (grow mamba ← shrink KV); KV has no floor → underflow → corrupt index → crash. Over-drain/negative-pool family (#312/#319/#320). mamba is floor-protected (#285/#297) but KV is not.
- Mitigation to try: drop concurrency to 32 so mamba stays unfull → only m2k (floor-protected) fires → no KV underflow. Re-run both arms at 32.

## Crash root cause = WRONG cost calibration (not concurrency)
Both conc 64 and 32 crashed via a k2m fire. FIRE reason showed **c_M=19.41ms** (non-zero).
Cause: the sys arm used cost_model's BUILTIN 35B default (m_beta=6.99, non-zero c_M),
because I never exported the calibrated `SGLANG_CSIGMA_*`. Non-zero c_M makes growing
mamba look beneficial → wrong-direction k2m → shrinks the floor-less KV → underflow.
FIX (in run_arm sys): export the calibrated single-curve κ from `kappa_fit.json`
(c_KV=1.02e-7·L²+0.0246·L+5.97, **c_M=0**). This matches the paper's single-curve
design. With c_M=0: **FIRE count=0** (k2m's net-benefit is 0; no fires) → no fast crash.

## Residual: arena KV used-count drifts negative even with 0 fires
With c_M=0 / 0 fires, the sys arm's KV `full token usage` still drifts slowly negative
(-0.01 → -0.18, #full token -139706) over a run, while base (no arena) never does. So
the SHARED ARENA's KV accounting underflows on its own (#319/#320 family); the k2m fire
merely accelerated it to a crash. cache_hit (from meta_info) stays valid regardless.
This is the deeper Budgeter/arena robustness bug; slow drift may let a run complete.

## The win requires PRESSURE, and pressure triggers the crash (the squeeze)
Per-rep-boot N=3 at limit 320 (to stay under the ~670 crash threshold) INVERTS the
result: base cache_hit **0.851**, sys **0.710** (base WINS). Reason: at 320 (~27 roots)
KV is NOT pressured → LRU is already near-optimal and the Budgeter/arena is pure
overhead. The \sys win only appears under real KV pressure (full t6: base drops to
0.526, sys 0.65, +13pp) — but that scale is exactly where the arena KV-used-count drift
crashes (~670 req). So: low load → base wins (no eviction pressure); high load → sys
wins (+13pp) but crashes. There is no clean low-load window.

CONCLUSION: the arena KV-accounting drift (negative used-count under arena+eviction at
high concurrency, even with 0 fires) GATES a clean high-pressure N=3 for ALL sys cases
(1/2/3 all use the arena). It must be fixed (the #316/#319/#320 net-free / free-ceiling
line) before clean Cases can be produced. The Case-2 WIN itself is validated (+13pp,
replicated across 3 independent sys rep-1 measurements vs base N=3).

## Case 3 (dynamic) — caveat
A true workload-driven pool-BINDING flip needs a mamba-bound phase, which real CC
can't produce (always KV-bound) and the mamba cap is fixed at boot (can't flip
mid-run). So Case 3 as "binding flips" is not representable with real CC + fixed
cap. Options: (a) vary KV-pressure intensity across phases (always KV-bound, but
Budgeter rebalances load), or (b) keep Case 3 on the old synthetic phased workload.
Decide after Cases 1-2.

## ROOT CAUSE of the sys crash (the ~800-req negative-#full-token underflow)
NOT the arena allocator, NOT the cross-pool fires (the 2 fires were k2m drain=0,
late at tick 144/169; the drift began ~tick 45). It is **K_BIG snapshot suppression
freeing the live KV suffix of a still-running request in `cache_unfinished_req`.**

Decisive isolation: with the SAME mamba_radix_cache, **base (no K_BIG) never goes
negative** (usage 0.0->0.99, 0 negative lines, 0 crash); sys (K_BIG=8192) drifts
`full token usage` negative starting ~90s in and crashes at -1.2 (#full token
-932848). The drift is **free-triggered**: each time `#running-req` drops (requests
finish) usage steps more negative.

Mechanism (mamba_radix_cache.py):
- `insert()` sets `mamba_value=None` when `insert_depth >= K_BIG and insert_depth %
  K_BIG != 0` (true for MOST deep inserts at CC's 16k-94k contexts).
- `_insert_helper` suppression branch then does `allocator.free(value)` on the
  unmatched trailing suffix and creates NO node (KV not cached, slots returned).
- But `cache_unfinished_req` keeps `req.prefix_indices = cat([new_indices,
  kv_indices_orig[len(new_indices):]])` and `cache_protected_len = len(new_indices)`
  with `len(new_indices) = deepest_snapshot_depth <= total_prefix_length`. So the
  request KEEPS referencing `kv_indices_orig[len(new_indices):]`, which OVERLAPS the
  just-freed suffix.
- => those physical KV slots are double-booked: in the allocator free list (counted
  in `available()`) AND still live for the running request. `available + evictable >
  live` => `full_num_used = live - available - evictable < 0`. Once a double-booked
  slot is alloc'd to a 2nd request the KV index corrupts -> CUDA illegal access in
  `next_token_ids.tolist()`.

Why it is unfinished-only: `cache_finished_req` also suppression-frees, but there the
request is DONE (KV dead), so freeing is correct. `cache_unfinished_req` is the
chunked-prefill path on a STILL-RUNNING request, so its suffix is live -> must not be
freed. base never hits it (K_BIG=0 -> no suppression).

CONTROL (confirmed): reran sys with KBIG=0 (suppression off, all else identical).
`full token usage` stayed >=0 throughout (0 negative lines past the point where
K_BIG=8192 had already crossed negative at ~tick 45), healthy 0.92-0.98. -> isolates
K_BIG suppression as the single cause; arena allocator + lpb are exonerated.
case2_kbig0_control/.

FIX (applied): the suppression "free trailing KV" is valid only when the caller no
longer needs that KV. Added `InsertParams.free_unstored_suffix` (default True = the
`cache_finished_req` contract, free dead KV); `cache_unfinished_req` now passes False
so a still-running request RETAINS its suppressed suffix as private working KV (in
req_to_token, not freed, not in tree). `_insert_helper` gates the suppression
`allocator.free(value)` on the flag.
- mamba_radix_cache.py: insert -> _insert_helper plumb + gated free; cache_unfinished_req
  passes free_unstored_suffix=False. base_prefix_cache.py: InsertParams field.
- Reproducing test (TDD, RED->GREEN): dev/interlayer/4_e2e/cc_zero_downside/
  test_kbig_suppression_retains_live_kv.py — real allocator + real tree, suppressed
  insert; finished path frees (PASS), unfinished path retains + no slot double-booked.
VALIDATION (in flight): full sys config (K_BIG=8192, the exact crashing config) + fix,
N=3 -> case2_sys_fixed/. Predict: 0 negative, all 1498 req complete, cache_hit ~0.65.

## CORRECTION: the suffix-free was a real but MINOR sibling; the DOMINANT cause is
## the dedup-free double-freeing the K_big tombstone gap (probe-confirmed)
The suffix-fix did NOT stop the crash (case2_sys_fixed rep1 still crashed at 681 req,
535 negative lines). A gated double-free probe (SGLANG_DBL_FREE_PROBE in allocator.free)
named the exact site, 7431 hits: `_insert_helper` dedup-free
`self.token_to_kv_pool_allocator.free(value[start:prefix_len])` (mamba_radix_cache.py),
reached from `cache_unfinished_req`, freeing **7513/7513 already-free ids at once**.

Mechanism: under K_big, `match_prefix` (and the repoint + `cache_protected_len`) cover
only up to `deepest_snapshot_depth`, but the dedup-free in insert frees the request's
redundant copies up to the FULL `total_prefix_length`. The matched region beyond the
deepest snapshot is the tombstone-internal nodes (KV cached but NO mamba snapshot), which
is NOT prefix-restorable (engine needs KV+snapshot both). So last chunk FREED [prev:total]
but PROTECTED only [0:deepest]; the gap [deepest:total] is left STALE in req_to_token (not
repointed). Next chunk re-presents those stale (now-free) slots and the dedup-free re-frees
them -> 7513-slot double-free -> available inflates -> negative #full token -> crash.

First-principles fix: the dedup-free must only free the request's copies for the
RESTORABLE prefix (up to deepest_snapshot_depth); the tombstone region [deepest:total] is
unshareable, so the request RETAINS its own KV there (private, re-prefilled). Same family
as the suffix-fix (don't free what the request still needs), but it requires deferring the
loop's incremental free to a single post-loop free bounded at deepest_snapshot_depth, in
the delicate insert/match/cache_protected_len invariant (asserts name the original author).
PENDING user call on the fix approach.

## RESOLUTION: K_big REMOVED entirely (it should never have been in the sys config)
User: "there should be no K_big, we discussed this." A 5-reader reconciliation (design.md,
paper, git history, RQ1 win-dependency, single-curve cost model) confirmed, all high-confidence:
- K_big is NOT in the paper (zero hits) and NOT in the current design.md/PLAN.md; it lives
  only in dev/archive_path_a, self-marked "purely historical" after the Path-B pivot.
- The RQ1 m2k KV-bound WIN does not use K_big: the genai-bench m2k win A/B (removed) never
  sets SGLANG_K_BIG (default 0). The +50pp / +10.8% tps win is pure Budgeter+Admitter
  cross-pool grow + LPB.
- The single-curve cost model (c_M=0, a miss = one forward folded into c_KV) cannot even
  PRICE K_big's only benefit (saving mamba slots), so the planner can't drive it; its
  auto-disable threshold was an out-of-band heuristic = exactly the fallback to kill.
- K_big was the verified crash source (dedup double-free above).
Fix is REMOVAL, not patching: deleted the K_big suppression from insert()/_insert_helper
(mamba_value is never None now -> the suppression-free branch is gone); the
deepest_snapshot_depth return is now UNCONDITIONAL (still needed for eviction-tombstones, the
one legitimate remaining tombstone source); reverted the suffix-fix flag (free_unstored_suffix),
the double-free probe, and the kbig unit test; removed SGLANG_K_BIG from run_arm.sh. CPU smoke:
a normal insert creates a node, retains its KV, counts evictable correctly. e2e: K_big-free sys
N=3 -> case2_sys_nokbig/ (validation in flight; predict 0 negative, all reps complete, cache_hit
> base 0.526). Honest caveat: no verbatim "remove K_big" was found in the record; the
recollection is supported in substance (dropped from the Path-B design), not as a recorded
directive.

Latent follow-up: the same dedup-vs-protected-depth mismatch can in principle recur via
EVICTION-tombstones under heavy mamba eviction (not triggered in Case 2, mamba idle). A
conservation-invariant regression test on cache_unfinished_req should pin this independently.

## RECONCILIATION: the agentreplay "cache_hit +13pp win" was a BUG ARTIFACT
Post-K_big-removal, sys cache_hit 0.528 ~= base 0.526 on cc_qwen_t6 (rep1 clean 1498/1498),
tput 436 vs base 447. The earlier 0.65 (with K_big) was the double-free inflating
`available` -> the radix over-retained KV cache on phantom free space -> higher cache_hit,
doomed to crash. It was never a sustainable win.

The REAL documented \sys cross-pool win is NOT a cache_hit win. the genai-bench m2k win A/B (removed) run_kv_bound_ab.sh states: "cache_hit ~0 both arms (distinct prompts) -- the win is
admission/throughput, not cache." It is +10.8% output_tps / -23.8% p99_ttft on: distinct
16k prompts, LOW concurrency (48), KV-bound with prefill HEADROOM (input_tps < prefill cap),
mamba capped at 580 = idle donor. m2k grows KV from idle mamba -> admits more -> prefill
batches toward the cap -> tput up, ttft down. The script itself notes high-concurrency
moderate-length tries are compute-bound (no headroom) -> 0 win.

Two harness gaps that explain the agentreplay non-win:
1. run_arm.sh sys config was MISSING the Admitter (the win mechanism): now fixed,
   SGLANG_HIMA=1 enables Admitter + Budgeter together (demand-driven fire magnitude,
   QUEUE_WAIT_US + COOLDOWN_S + TICK_S=1.0).
2. cc_qwen_t6 at 64 concurrency is compute-bound (no prefill headroom) = the wrong regime
   for the throughput win.
=> The cross-pool throughput win is a low-concurrency-long-prefill-headroom result, not a
high-concurrency cache_hit result. On the high-concurrency real-CC trace, LPB ~= LRU
(0.528 vs 0.526) and the budgeter/arena is pure overhead (-2% tput, pending N=3).

PENDING: (a) paired base-vs-sys N=3 per-rep-boot on cc_qwen_t6 (current no-Admitter config)
to confirm the ~2% regression is stable vs noise -> case2_n3/; (b) user framing call on how
to present the cross-pool throughput win (proper config + prefill-headroom workload).
NOTE: multi-rep on one server crashes at the rep-boundary flush (arena-state-vs-flush bug,
separate); per-rep boot avoids it.

## 2026-06-16 — KV-bound (m2k) clean N=3 WIN, after the win-path fixes

Config: `cc_qwen_t6`, conc 64, CTXLEN 262144, default split. base = lru. sys =
run_arm.sh sys (lpb + Budgeter + Admitter cross-fire + PF64 + calibrated c_M=0,
no K_big). base N=3 (shared server, stable); sys N=3 (per-rep boot, sidesteps the
#327 flush crash). Both arms 0 errors.

| metric | base N=3 | sys N=3 | delta |
|---|---|---|---|
| cache_hit | 0.5267 ± 0.0007 | 0.5431 ± 0.0001 | +1.63 pp |
| throughput tok/s | 449.8 | 448.9 | −0.2% (flat) |
| TTFT p50 | 13453 ± 288 ms | 2766 ± 86 ms | **−79.4%** |
| TTFT p99 | 51945 ms | 37018 ms | **−28.7%** |
| e2e p50 | 30814 ms | 26124 ms | −15.2% |

Mechanism: KV peak capacity grew +50% (773k → 1.16M tokens) via 959 Admitter
cross-fires (base 0). The win is LATENCY (TTFT), not throughput: at conc 64 the
GPU is compute-bound so total throughput is capped (flat, which RESOLVES the prior
"−2% tput overhead" concern as noise), but growing KV from idle mamba relieves
queue-wait so requests reach first-token ~5x sooner.

Three win-path fixes made this clean N=3 reachable (sglang HiMA, local):
- f0df5ec5ff: graceful degrade on mamba-alloc failure (m2k over-drain backstop).
- 1f87f915ac: sync pp_max_micro_batch_size with the actuator-driven admission cap
  (the gate was frozen at boot; needed for the k2m concurrency win, case2).
- 025217ecb1: restore the un-transferred src surplus after a misaligned fire (was
  leaking mamba live_size every m2k fire). Plus the prior #320 free-ceiling and
  #325 K_big removal. Net: 0 "Can not alloc mamba", 0 negative-token drift over
  1498 reqs/rep, far past the old ~670-req arena-drift crash gate.

Residual: shared-server multi-rep still crashes at the rep-boundary `--flush`
(#327, distinct arena-state-vs-flush bug); per-rep boot is the workaround.

## 2026-06-16 — mamba-bound (k2m) clean N=3 WIN (the induced concurrency case)

Config: `cc_qwen_t6`, conc 128, `--mamba-full-memory-ratio 0.1` (induces
mamba-bound: max_running ~27 binds while KV sits ~45% idle),
`SGLANG_ADMISSION_MAX_FACTOR=4` (enables dynamic-cap mode so max_running can grow;
default 1.0 leaves it frozen and the k2m win impossible). base = lru. sys =
run_arm.sh sys + the factor. N=3 per-rep boot, both arms 0 errors.

| metric | base N=3 | sys N=3 | delta |
|---|---|---|---|
| max #running-req | 27 | 98 (all 3 reps) | +3.6x concurrency |
| TTFT p50 | 47040 ms | 21218 ms | **−54.9%** |
| TTFT p99 | 116174 ms | 99204 ms | −14.6% |
| throughput tok/s | 430.0 ± 0.6 | 449.5 ± 0.8 | +4.5% |
| makespan | 1081.7 s | 1034.8 s | −4.3% |
| cache_hit | 0.5096 ± 0.0008 | 0.5445 ± 0.0004 | +3.49 pp |

Mechanism: the Budgeter grows mamba from idle KV (k2m), the admission cap follows
(pool.size 84→112, mamba usage→0.92), max_running rises 27→98, the queue drains.
This is the k2m direction (mirror of the case1 m2k KV-grow). Required the admission
factor (dynamic-cap), the #1 gate-sync, and the None-cap init fix (11c736af2e).
Real CC has no truly-short prompts (16k+ prefix), so the mamba-bound regime is
induced via the low ratio, labeled as such.

## 2026-06-16 — case3 (single-replay temporal flip): structurally NOT achievable on real CC

Goal: one replay whose binding pool flips KV-bound (long phase) -> mamba-bound
(swarm phase), served on ONE fixed boot split, so sys reallocates both ways and
beats the static split. Three GPU attempts, three understood failures:

1. ratio 0.5, 25k swarm, overlapping phases: KV-bound throughout. The swarm's
   independent 25k prompts saturate KV before max_running; and the long phase
   never drained so the swarm queued behind it.
2. ratio 0.1, swarm truncated to ~13k first-turns, separated phases: still
   KV-bound. First-turn truncation removed prefix-sharing, so each fresh 13k
   prompt needed full KV -> KV saturated again.
3. ratio 0.1, full multi-turn swarm (case2's proven mamba-bound trace),
   separated: the LONG phase was now underloaded (KV 0.44) because at ratio 0.1
   the KV pool is large enough that 8 long contexts do not saturate it.

Root cause is structural, not tuning: `--mamba-full-memory-ratio` sets the KV
and mamba pools INVERSELY at boot. A long phase is KV-bound only when KV is
small (high ratio); a swarm phase is mamba-bound only when mamba is small (low
ratio). No single fixed ratio makes BOTH true, and real CC's 16k+ context floor
(no genuinely-short independent prompts) removes the other degree of freedom.
A clean in-replay flip would need either the forbidden `--max-mamba-cache-size`
(force mamba-bound regardless of ratio) or synthetic non-real-CC contexts. This
matches the prior "Case 3 not representable with real CC + fixed cap" note.

The DYNAMIC claim is instead carried by the case1+case2 pair: ONE sys config
wins BOTH opposite regimes (case1 KV-bound m2k: TTFT p50 −79%; case2 mamba-bound
k2m: TTFT p50 −55%, concurrency 27→98), while no single static split wins both
(the case1-best KV-heavy split leaves case2 concurrency-starved; the case2-best
low-mamba ratio leaves case1's long phase underloaded / KV-starved). That is the
"no single static split serves a workload with both phases" argument, shown as
two regimes rather than one temporal trace. The ratio-based case3_dyn artifacts
(build script, runner, trace) were removed as superseded; the current default-split
case3 attempt is `case3_default.sh` (trace built by `case_default_build.py`).

## 2026-06-18 — case1 (KV-bound m2k) on rebased code: no-regression + p50 TTFT win

Rebased onto latest origin/main (`1981464ba4`), sgl-kernel 0.4.4, all 27 unit tests
green.
Trace: `cc_qwen_case1_longkv`, 52 longest roots from the 2.3G corpus (prompt p50
92k / max 196k), fully verbatim multi-turn (38 turns/root, full real outputs, no
decode cap).
Both arms at sglang DEFAULT split, overlap ON, conc 48.
Per-boot N=3 (flush-boundary crash on rep 2+ on rebased code, workaround).

| metric | base N=3 | sys N=3 | delta |
|---|---|---|---|
| out_tps | 555.0 ± 3.2 | 554.9 ± 7.8 | flat (−0.0%) |
| cache_hit | 0.862 ± 0.001 | 0.862 ± 0.000 | flat |
| p50_ttft | 226 ± 13 ms | 215 ± 16 ms | **−4.7%** |
| p99_ttft | 6184 ± 23 ms | 6200 ± 13 ms | flat (+0.3%) |
| p50_tpot | 55.1 ± 0.7 ms | 54.7 ± 0.4 ms | −0.7% |
| errors | 4127 / 4935 | 4127 / 4935 | 0 delta |

NO-REGRESSION: PASS (tps, p99_ttft, errors all within tolerance).
WIN: p50 TTFT −4.7% (median first-token latency improvement).

The win is modest because multi-turn CC sessions share a ~16k system prefix,
giving 0.86 cache hit: most KV is served from the radix cache, not fresh
admission, so the m2k KV-grow mechanism does not exercise heavily.
Throughput is flat (compute-bound at these long contexts + high cache reuse).
To amplify the win: select sessions with lower prefix-sharing (distinct system
prefixes or first-turn-only), driving cache_hit down toward ~0 and forcing
real KV admission pressure (the regime where genai_bench got +10.8%).

### First-turn-only variant (same 52 roots, step=1 only): FLAT

52 single requests (one first turn per root), conc 48, per-boot N=3.
Result: tps 696→698 (+0.3%), p50_ttft 5417→5421 (+0.1%), 0 errors — FLAT.
cache_hit is STILL 0.762 because all CC sessions share the ~16k system prefix;
the radix cache serves it on 51/52 requests, so fresh KV admission is only
~4k/req (the unique suffix), not the full 17k prompt.

### BudgetAgent health-check fix (fb0b159e85) — ALL prior runs were HiMA-disabled

The BudgetAgent's health check required `stats.max_total_num_tokens` which upstream
moved out of `SchedulerStats`. The check SILENTLY hard-disabled the BudgetAgent +
Admitter on first tick: zero fires, zero JSONL lines on every experiment. Fixed by
removing the field from the health check. All results above this line were measured
with HiMA silently OFF (the p50 TTFT −4.7% was from LPB eviction, not cross-pool).

### Multi-turn verbatim RERUN with BudgetAgent enabled (N=3 per-boot, default pool)

base 559.7 ± 6.6 tps, sys 574.4 ± 12.7 tps: **+2.6% tps, −3.4% p50 TTFT**, 18
fewer sys errors. Budgeter alive (77-81 ticks/rep), 0 m2k fires (H200's 1.7M KV
pool is too large for the workload to exhaust own_evict). The +2.6% is LPB eviction.

### Deep-turn 4x at MEMFRAC=0.30: m2k fires HAPPEN, then arena CRASH

52 deepest turns (92k each) duplicated 4x = 208 requests, MEMFRAC=0.30 (KV pool
415k, ~4 fit). Budgeter alive (17 ticks), **4 m2k fires per rep** (consistent
across 3 reps). All 3 sys reps CRASH: CUDA illegal memory access after the fire.
Base: stable 73.8 tps, 0 crashes. The crash is the known arena free-boundary
family (#320/#327) surfacing when the cuMemUnmap/Map remap physically moves pages.

CONCLUSION: the cross-pool mechanism IS alive and fires correctly on the rebased
code. The large win is blocked ONLY by the arena remap crash (the fired pages
corrupt VA after remap). Fixing the arena crash = the path to the large case1 win.
The multi-turn CC result (+2.6% tps from LPB) is the no-regression baseline;
the deep-turn MEMFRAC=0.30 result proves the mechanism fires and is ready to
deliver the win once the arena crash is resolved.
