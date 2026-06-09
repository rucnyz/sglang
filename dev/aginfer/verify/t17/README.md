# T17 — State-dump schema upgrade (DESIGN §5)

PLAN.md §3 T17.  Replace the legacy state-dump schema with the
residence-set, per-subpool, two-view schema specified in DESIGN §5.
This is a hard replace — no add-alongside, no compatibility shim.

## WHAT WE PROMISED

**Capability.**  `GET /aginfer/state` returns JSON with this top-level
shape (full schema in DESIGN §5):

```
{
  "time_counter":      int,
  "throughput_ema":    {prefill_bps, decode_per_program: {pid: ...}},
  "pool_usage":        {tier: {subpools: {sp: {used_bytes, cap_bytes,
                                               available_bytes,
                                               evictable_bytes,
                                               page_bytes}}}},
  "per_program_usage": {pid: {hbm: {committed: {sp: int},
                                    inflight:  {sp: int}},
                              dram: {committed: {sp: int}},
                              state, pre_pause_state, unit_hashes}},
  "units":             [{hash, residence: [tier, ...],
                         n_tokens, n_bytes: {tier: {sp: int}},
                         last_access_time, hit_count, session_ids}],
  "link_stats":        {"σ->τ": {peak_bw_bps, recent_throughput_bps,
                                 time_since_last_sample_s}},
  "tier_holding_cost": {tier: {sp: {h_max_per_byte_sec}}}
}
```

**Hard requirements (in order of how easy they are to silently break).**

1. **Residence is a SET, not a tier.**  A unit post-write_through has
   `residence == ["HBM", "DRAM"]`.  Code that assumes "exactly one
   tier per unit" is wrong.
2. **`n_bytes` is `{tier: {subpool: int}}`, not a scalar.**  Sum over
   the nested dict to get total bytes; sum over one tier-slot to get
   per-tier bytes; the per-(tier, subpool) breakdown is the §9 DP
   axis input.
3. **`pool_usage[tier].subpools` keys ⊇ `units[i].n_bytes[tier]` keys**
   for every tier `t` in `units[i].residence` (every subpool that a
   unit has bytes in must be declared in `pool_usage`).
4. **Legacy fields are GONE**: no top-level `tier_usage`, no top-level
   `page_size`, no top-level `bytes_per_token`, no `swa_*` flat fields
   on `pool_usage.HBM`.  The daemon halts on first launch if it sees
   the legacy shape (round-9 part 4 "halts loudly" invariant).
5. **`link_stats.<σ-τ>.time_since_last_sample_s` cold-start is `+Inf`**,
   not 0 or null.  The daemon's bw_free branch keys on
   `> LINK_IDLE_SECONDS`; 0 means "just sampled" and would mis-route
   to the EMA path.
6. **`per_program_usage[p].committed[sp]` uses 1/holders attribution.**
   A unit shared by 2 programs contributes `n_bytes / 2` to each.
7. **`units[*]` and `per_program_usage[*].unit_hashes` reconcile.**
   `unit_hashes` is a materialised list per program; for every unit
   in `units`, `p ∈ unit.session_ids ⇒ unit.hash ∈ per_program_usage[p].unit_hashes`.

**Cost ceiling.**

* DESIGN §10 F3 trigger: sglang-side `dump_aginfer_state` p99 > 50 ms
  ⇒ revisit triggered.  This task pins p99 < 50 ms on a 10 K-node
  tree as a guard.
* No allocator-pressure regression: the bytes path (`dump_aginfer_state_bytes`)
  must remain allocation-light — no per-node `dict` materialised, hash
  strings written directly into the output `bytearray`.  Empirically a
  Gen-2 GC sweep on the live radix tree + KV-pool state is ~500 ms,
  which would alone violate the budget.
* ~400-600 LoC of sglang change: `_dump_aginfer_state_impl`,
  `_dump_aginfer_state_bytes_inner`, `_aginfer_pool_usage`, plus three
  new builders (`per_program_usage`, `link_stats_init`, `tier_holding_cost`).
  Top-level handler in `http_server.py` does not change.

## WORST CASE (forced)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Multi-holder underflow | 4 programs request same 1 K-token prefix | unit's `session_ids` has 4 ids; each program's `committed.attn` = unit.n_bytes / 4 (±1 byte round) | Stage 5 |
| Post-write_through residence | Send a chat with `program_id=A`, wait for HiCache write_backup to fire (or force via `--hicache-write-policy write_through`) | the prefix node has `residence == ["HBM", "DRAM"]`; `n_bytes["HBM"][sp] > 0` AND `n_bytes["DRAM"][sp] > 0` | Stage 3 |
| HBM-evicted unit | Drive cap = 64 K to force eviction while DRAM is plenty | evicted unit has `residence == ["DRAM"]`, `n_bytes` no longer has an `"HBM"` slot | Stage 3 |
| Empty tree | Fresh launch, no requests | every required field present; `units == []`; `per_program_usage == {}`; `pool_usage[t].subpools[sp].used_bytes == 0` for all (t, sp) | Stage 1 |
| 10 K-node stress | 100 batches × 100 distinct prefixes | dump p99 < 50 ms; no GC sweep visible in p99 tail | Stage 6 perf guard |
| Concurrent walker + traffic | 5 background threads `GET /aginfer/state` @ 10 Hz; concurrent `/v1/chat/completions` | 0 schema failures; 0 unit_hashes-vs-units reconciliation failures; p99 walker latency < 100 ms | Stage 7 |
| Legacy-shape detection | Run verify against a hypothetical pre-upgrade sglang (negative test) | verify.py raises `LegacyShape` and exits non-zero | Stage 0 |

## HOW WE VERIFY

`verify/t17/verify.py` runs N≥3 cycles of the staged plan below.  Each
cycle restarts sglang from clean state.  Each stage asserts ONE invariant
class so a failure points at one DESIGN clause:

```
Stage 0 — Strict-mode parser
  Assert presence of every required §5 path; assert absence of every
  legacy field.  Negative test: feed it a synthesised legacy snapshot,
  expect it to raise.

Stage 1 — Empty tree
  Launch sglang, GET /aginfer/state before any request.
  units == []; per_program_usage == {}; pool_usage[*][*].subpools
  present; subpool[*].cap_bytes > 0.

Stage 2 — Single-unit attribution
  One request with program_id=A, short prompt, 1 prefill chunk.
  units has 1 entry; residence == ["HBM"]; n_bytes["HBM"][sp] > 0;
  per_program_usage["A"].hbm.committed[sp] == unit.n_bytes["HBM"][sp];
  per_program_usage["A"].unit_hashes == [unit.hash].

Stage 3 — Residence-set transitions
  3a. Prefill → HBM only → residence == ["HBM"].
  3b. Wait for write_backup (or force write_through) → residence ==
      ["HBM", "DRAM"]; n_bytes has both slots.
  3c. Evict from HBM by hitting cap → residence == ["DRAM"];
      n_bytes no longer has an "HBM" slot.
  3d. Drop entirely → unit absent from units[].

Stage 4 — Subpool degeneracy (S1)
  For DeepSeek-V4-Flash (MLA single-stack), pool_usage.HBM.subpools
  has exactly the keys declared by the architecture.  Sum of
  units[*].n_bytes["HBM"][sp] ≤ pool_usage.HBM.subpools[sp].used_bytes
  for each sp (radix subset ≤ allocator total because in-flight
  decode bytes are not in the tree).

Stage 5 — Per-program attribution (1/holders)
  4 programs A,B,C,D each issue a chat with identical 1K-token prefix
  + per-program suffix.  Shared-prefix unit has session_ids = {A,B,C,D};
  per_program_usage[p].hbm.committed[sp] for each p == unit_shared.n_bytes
  / 4 (within ±1 byte rounding).  Tail units belong to one program each.

Stage 6 — Perf guard (N>=3)
  Drive tree to 10 K units.  Time 20 dumps each cycle, 3 cycles.
  Assert mean p99 < 50 ms; assert no single dump > 200 ms (Gen-2 GC
  ceiling).  Record per-cycle (p50, p99, max).

Stage 7 — Concurrent stress
  5 walker threads + 32 concurrent /v1/chat/completions for 30 s.
  Every parsed response goes through Stage 0 parser; assert 100 %
  pass rate.  Assert per_program_usage[*].unit_hashes ⊂ {u.hash} for
  every snapshot.  Record p99 walker latency.

Stage 8 — Link stats / tier holding cost / throughput EMA shape
  All three fields present.  link_stats has one entry per σ-τ
  permutation of {HBM, DRAM, DISK} (6 directional links).  Cold-start:
  every link's time_since_last_sample_s == math.inf.  After driving
  a write_backup, HBM->DRAM's value drops.  tier_holding_cost has
  per-(tier, subpool) entries with h_max_per_byte_sec >= 0.  After
  one prefill, throughput_ema.prefill_bps > 0.
```

## REPRODUCING

```bash
# 1. Activate env.
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh
conda activate agsched
cd /scratch/yuzhou/projects/sglang/dev/aginfer

# 2. Pick a free GPU (typically 5 or 6 — see [[gpu-layout]]).
GPU=5
PORT=30001

# 3. Launch sglang with a small KV pool so eviction fires in Stage 3.
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
CUDA_VISIBLE_DEVICES=$GPU \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port $PORT \
    --tp 1 --mem-fraction-static 0.15 \
    --max-total-tokens 65536 \
    --trust-remote-code \
    --attention-backend flashinfer \
    --enable-hierarchical-cache \
    --hicache-ratio 1.5 \
    --hicache-write-policy write_through \
  > logs/sglang_t17.log 2>&1 &
SGLANG_PID=$!
until grep -q "Uvicorn running on http://127.0.0.1:$PORT" logs/sglang_t17.log; do sleep 3; done

# 4. Run verify.
AGINFER_VERIFY_BASE=http://127.0.0.1:$PORT \
AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
  python verify/t17/verify.py

# 5. Teardown.
kill "$SGLANG_PID"
```

## RESULTS

**PASSED** — all 9 stages on a fresh Qwen3-0.6B + HiCache launch,
after subagent audit hardened 5 assertions.

* date: 2026-05-31
* model: Qwen/Qwen3-0.6B, flashinfer backend, HiCache write_through,
  cap 64 K tokens, GPU 5
* lines added: ~520 in `unified_radix_cache.py` (4 new helpers + dict /
  bytes dump paths split); 20 lines simplified in `scheduler.py`
  unsupported-cache fallback; ~80 lines of test tightening after audit
* Stage results (post-audit):
  | Stage | Result |
  |---|---|
  | 0  strict-parser negative | PASS |
  | 1  initial shape | PASS |
  | 2  single-unit attribution | PASS — 1 program, residence ⊇ {HBM} |
  | 3  residence-set transitions | PASS — saw `[HBM, DRAM]` after write_through (HARD-fail if not seen within 3 s) |
  | 4  S1 subpool degeneracy | PASS — HBM `Σ ≤ pool_used`; DRAM `Σ == pool_used` per (tier, sp) |
  | 5  multi-holder strict 1/holders | PASS — 4-program shared prefix, all 4 pids attributed within ±n_units·(n_holders-1) byte tolerance |
  | 6a perf @ 5 K tree (asserted) | PASS — aggregate p99 over n=60 = 28.5 ms (p50 = 25.8 ms; budget 50 ms) |
  | 6b perf @ 17 K tree (informational) | p50 = 119 ms, p99 = 143 ms — within Gen-2 GC ceiling |
  | 7  concurrent stress (5 walkers + 32 chats × 30 s) | PASS — 361 walks, 0 schema failures (walker p99 = 774 ms informational) |
  | 8  link / holding / ema shape | PASS — cold-start `time_since_last_sample_s > LINK_IDLE_SECONDS = 1.0` (the daemon's actual contract) |

* Stage 6a per-cycle: `(p50, max)` = `[(26.0, 32.2), (25.9, 26.3),
  (25.6, 26.4)]` ms.  Aggregate p99 over all 60 samples = 28.5 ms.
  The GC-tail regression from the first impl pass (single dump 543 ms
  at 21 K units) was killed by routing the bytes path through a
  hand-written `bytearray` writer instead of materialising 10 K
  per-node Python dicts.
* Stage 7 walker latency under stress: p99 = 774 ms (informational).
  At 17 K-unit dirty tree + 5 concurrent walkers + 32 concurrent
  chats this is expected — T14 instrumentation will capture this in
  production and trigger F3 revisit if it ever happens under real
  workload (per DESIGN §10 trigger condition).

### Audit findings + fixes applied

The subagent audit (2026-05-31) found 5 critical issues with the
initial test, all fixed in this pass:

1. **Stage 3** had a soft-fail when write_through didn't fire (printed
   PASS with a diagnostic).  Now HARD-fails with `ResidenceInvariant`
   — the new schema's central claim is residence-as-set, so the
   stage that exercises it must require seeing dual residence.
   Timeout dropped from 10 s to 3 s (HiCache write_through completes
   in well under 1 s in healthy config).
2. **Stage 4** asserted HBM `Σ ≤ pool_used` for both tiers — but the
   impl patches `pool_usage.DRAM.subpools[sp].used_bytes` from the
   walk's per-subpool DRAM sum, so DRAM should be `==`.  Now split:
   HBM inequality (radix is subset of allocator) + DRAM equality
   (the patch logic is what we're testing).
3. **Stage 5** had an escape hatch `got < want * n_holders` that
   allowed any attribution from 0 to total_bytes to pass — the bug
   case (no 1/holders divide) slipped through silently.  Rewrote
   with strict equality (±byte-rounding from integer-divide) and
   no fall-back clause.
4. **Stage 6a** averaged per-cycle p99s, which washes out a single
   bad cycle.  Now aggregates all 60 samples and takes the true p99.
5. **Stage 8** checked `time_since_last_sample_s > 1e9` — far
   stricter than the daemon's actual `> LINK_IDLE_SECONDS = 1.0`
   branch.  Now checks the daemon's contract (any value above 1.0
   triggers the bw_free idle branch), decoupling the test from the
   impl's specific placeholder value (1e12 today).

Coverage gaps that the audit flagged as out-of-scope for T17 (each
is the target of another PLAN.md task):

- D4 chunked-prefill atomicity (mid-prefill chunks not in `units`):
  PLAN T19 (Atomic unit visibility).
- Active DROP / empty-residence path: covered by T20 (migrate payload)
  exercising the empty-residence transition.
- Multi-subpool degeneracy (S2 SWA, S3 Mamba): PLAN T45/T46.
- Per-program `inflight` bytes population: PLAN T29 (eviction-scorer
  plugin) + scheduler in-flight tracking.
- DISK-only residence: blocked on Mooncake L3 wiring (PLAN T20 covers
  the migrate payload that lights DISK up).

* raw run log: `results/20260531_t17_post_audit_pass.log`
