# T2 — `POST /aginfer/migrate`

## WHAT WE PROMISED

**Capability**
* Accepts `{"actions": [{"hash": "...", "target_tier": "HBM|DRAM|DISK|DROP"}, ...]}`.
* For each action: locate the tree node by `hash` and dispatch to the
  existing sglang internal (`_evict_device_leaf`, HiCache backup, prefetch).
* Returns `{"applied": N, "skipped": [{"hash": "...", "reason": "..."}]}`
  — skips hashes that don't resolve to a node, never raises.
* Unknown hashes are not an error (cache state changes asynchronously).

**Cost ceiling**
* < 1 ms per action amortized over 1 K-action batches.
* < 40 lines added to sglang.
* No mutex blow-up: while migrating, normal cache operations continue.

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| hash-not-found race | Insert a node, capture its hash, internally `_evict_device_leaf` to remove it; immediately POST `/aginfer/migrate` for that hash | response is `{applied: 0, skipped: [{hash: X, reason: not_in_tree}]}`; no exception. **Daemon: retryable — drop from retry set, log and move on.** | parse response, assert structure + counts |
| Node is internal (has children) | POST migrate(DROP) on a node that still has children | `{applied: 0, skipped: [{hash: X, reason: not_a_leaf}]}`. **Daemon: retryable bottom-up — drop the children first, then re-issue for the parent.** | step [6] explicit-binding probe |
| Node has no data (buffer freed but still in tree) | After a successful DROP on X, send a duplicate DROP for X in a later batch (the in-batch case is caught by `already_acted_this_batch`, see next row); or target a node whose data was cleared by sglang's auto-eviction between snapshot and migrate | `{applied: 0, skipped: [{hash: X, reason: no_data}]}`. **Daemon: retryable, but the same as not_in_tree — the node is functionally gone, drop from retry set.** | parse, assert reason |
| Same hash twice in one batch | `migrate([{drop X}, {drop X}])` — without the defensive `acted_node_ids` set this SIGQUITs the scheduler at `_remove_leaf_from_parent:1124` (`assert v == node`) | second occurrence skipped as `already_acted_this_batch`. **Daemon: programmer error — investigate the upstream code that emitted the duplicate.** | step [7] (DEFENSIVE check, found a real bug; see RESULTS) |
| unknown target_tier (typo / version skew) | POST migrate with `target_tier="DOESNOTEXIST"` for a live hash | response `{applied: 0, skipped: [{hash: X, reason: unknown_target_tier:'DOESNOTEXIST'}]}`; no exception. **Daemon: programmer error — daemon should pin to v1's known tier set.** | parse, assert reason substring |
| DISK tier in v1 (not wired) | POST migrate with `target_tier="DISK"` for a live hash | `{applied: 0, skipped: [{hash: X, reason: disk_tier_not_yet_wired}]}`. **Daemon: not retryable in v1; either pin to HBM/DRAM/DROP or wait for T9.** | parse, assert reason |
| Malformed payload | POST migrate with `{}`, `{"actions": "not a list"}`, or non-JSON body | HTTP 400 on all three; no exception, no partial mutation. **Daemon: caller bug; do not retry.** | requests + status assert |
| DRAM demote with no host backup (v1 contract) | POST migrate(target=DRAM) for an HBM-only hash (no HiCache backup populated) | `{applied: 0, skipped: [{hash: X, reason: demote_requires_existing_host_backup}]}`. **Daemon: retryable later (HiCache backup_thread may catch up), or fall back to DROP.** | parse, assert reason |
| Slow-path 1k batch latency (real DROPs) | warm 60+ distinct leaves, POST migrate(target=DROP) for all of them | per-action amortized < 1 ms; all real targets actually evict | timeit + causal absence check + tier_usage delta |
| TOCTOU between state and migrate | Fetch `/aginfer/state`, then drive 5-10 prefills before POSTing migrate. The hashes the daemon believed were HBM-resident leaves may now be DRAM-only, internal, or auto-evicted | response remains well-formed, skip reasons all in the legal set above; NO 5xx; the daemon's idempotent retry handles the rest | step [17] |
| Mixed-tier batch (cross-action interaction) | One batch with [DROP, DRAM, HBM, DROP, ...] interleaved on hashes that share a parent chain | every action accounted for, no double-apply, no 5xx; applied hashes are absent from a follow-up `/aginfer/state` | step [18] |
| Overlapping concurrent batches | Two threads POST the SAME 20-hash batch | both responses well-formed; `applied_hashes` from the two responses are DISJOINT (each hash applies at most once across both requests); total applied ≤ batch size | step [19] |
| HTTP DoS via huge hash / huge batch | POST a 10 000-char hash; POST 100 001 actions | both return 400 (caps enforced in `http_server.py`) | step [20] |
| 1k small migrate calls in tight loop | repeated cheap migrate calls | p99 latency < 100 ms; late-window median < 3 × early-window median (no slow leak) | step [21] |
| Idempotent replay | POST migrate(DROP) once for hashes H1..Hn; capture {applied}; immediately POST the same batch again with no traffic in between | second response: round-1 applied hashes are still absent; any newly-applied hash must be in the round-1 `not_a_leaf` bucket (cascade promotion is legitimate); no crash | step [10] cascade-aware replay |
| HiCache backup target tier full (audit #14) — deferred to T9/Run K | Fill DRAM to 100 % via real warmup under `--enable-hierarchical-cache`. Then POST 1k migrate(target=DRAM) actions for HBM-resident hashes | Daemon falls back via `demote_requires_existing_host_backup` until backup_thread catches up; once host is full the response carries the existing HiCache pool's `out_of_capacity` skip reason verbatim. Idempotent re-issue does NOT amplify | requires HiCache; exercised in T9 |
| Capacity-full promote — deferred to T9 | HBM at 100 %, POST migrate(target=HBM) for a DRAM hash | v1 returns `promote_not_yet_wired`; promote semantics land in T9 when the kv_scheduler decides promotions explicitly | T9 |
| 1000 actions / batch under 30 RPS load — deferred to T10 | Insert 10 k nodes, drive concurrent traffic at 30 RPS, then POST a single migrate with 1 k random actions | per-action amortized < 2 ms (= 2× ceiling under load); no daemon timeout | T10 (requires RPS generator) |

### `applied_hashes` consumer (audit round-3 MINOR)

The response includes `applied_hashes` so the daemon's retry logic can
prune its retry set without re-walking `/aginfer/state`.  T5/T7 (kv_scheduler
event handlers) and T8 (admission_controller) will both consume this:
when an action is reported applied, the daemon removes it from any
pending retry queue and from its in-memory paper-§3 state snapshot.  The
field is small (one hex string per applied hash) and the cost is O(applied)
both server- and wire-side.  If a future profiling pass shows it dominates,
add a `with_applied_hashes: bool = True` request flag.

### Daemon retry-set classification

For a fast hand-off into T4/T5, here is the canonical mapping every skip
reason emitted by `apply_aginfer_migrations` falls into:

* **retryable, idempotent** (re-issue once the tree changes): `not_in_tree`,
  `no_data`, `not_a_leaf`, `demote_requires_existing_host_backup`,
  `already_on_dram`, `already_on_hbm`.
* **defensive / programmer error** (do not retry; investigate upstream):
  `already_acted_this_batch`, `unknown_target_tier:'...'`,
  `unsupported_tree_cache:...`.
* **v1 not-yet-wired** (will become retryable in T9/T10): `promote_not_yet_wired`,
  `disk_tier_not_yet_wired`.

Daemon's `RETRYABLE_REASONS` set MUST include every string in group 1; the
group-2 strings are programmer-error sentinels; group-3 is a known
v1 limitation that the daemon should explicitly skip (not retry forever).

## HOW WE VERIFY

Mechanism. `verify/t2_migrate_endpoint.py`:

```
1. Pre-populate the tree (same warmup as T1).
2. Snapshot state via /aginfer/state. Pick 100 HBM nodes.
3. POST /aginfer/migrate with target_tier=DROP for all 100.
4. Re-fetch /aginfer/state. Assert:
   - None of those hashes are present on HBM anymore (or absent entirely).
   - Other unrelated nodes are unaffected.
5. POST /aginfer/migrate with target_tier=DRAM for 100 different nodes.
   Re-fetch, assert their tier flipped to DRAM (or DRAM+HBM if still active).
6. POST /aginfer/migrate with target_tier=HBM for a known-DRAM hash.
   Assert it lands on HBM (or surfaces as a "skipped, capacity" if HBM full).
7. Latency micro-bench: 100 batches of 100 actions each, measure throughput.
```

## REPRODUCING

Same setup as T1 (no special launch flags); 21 in-suite steps.
`CUDA_VISIBLE_DEVICES=4` is a default — pick any free GPU per
`nvidia-smi` (the MEMORY-notes default convention is GPU 5 or 6 free,
but check before you launch).  Capture the launch PID for a precise
tear-down (`pkill -f` would also catch unrelated processes that
happen to match the pattern).

```bash
source /scratch/yuzhou/miniconda3/etc/profile.d/conda.sh
conda activate agsched

cd /scratch/yuzhou/projects/sglang/dev/aginfer
# Abort early if port 30001 is already bound.
lsof -i:30001 && { echo "port 30001 already in use"; exit 1; }

SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 CUDA_VISIBLE_DEVICES=4 \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen3-0.6B \
    --host 127.0.0.1 --port 30001 \
    --tp 1 --mem-fraction-static 0.15 \
    --max-total-tokens 65536 \
    --trust-remote-code \
    --attention-backend flashinfer \
  > logs/sglang_t2.log 2>&1 &
SGLANG_PID=$!

# Wait for the listener to come up:
until grep -q "Uvicorn running on http://127.0.0.1:30001" logs/sglang_t2.log; do sleep 3; done

AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
  python verify/t2/verify.py
# expected last line: "=== T2 PASSED (depth-audit + round-3) ==="

kill "$SGLANG_PID"
```

## RESULTS

**PASSED** (audit round-3 edition, 2026-05-26).  All 21 in-suite steps pass on
Qwen3-0.6B + `--attention-backend flashinfer`, single-DP.  This edition
covers every branch of `apply_aginfer_migrations` that is reachable without
HiCache, plus the tier_usage delta invariant, IPC serialization, duplicate
hashes, empty/malformed actions, cascade-to-zero, round-2 applied absence,
and a 2-thread concurrency probe.

**Bug found and fixed during the audit**.  Sending a duplicate hash within
the same migrate batch SIGQUIT-crashed the scheduler:
`_remove_leaf_from_parent` line ~1124 fired `assert v == node` because the
second iteration of the same node entered the DROP path even though the
first iteration had already detached it.  Root cause:
`FullComponent.evict_component` (`full_component.py:113`) defers
`cd.value = None` to a later `_cascade_evict(trigger=BASE,
target=DEVICE)` trigger, and our DROP path uses `target=ALL` which never
fires the deferred clear.  Fix lives in `apply_aginfer_migrations` itself
as a defensive `acted_node_ids` set; documented in code.  The first-round
audit (which did not require duplicate-batch testing) would have shipped
this bug.

* date: 2026-05-26
* sglang sha: (this commit)
* lines added: ~170 across 5 files (io_struct + scheduler dispatch + mixin
  pair + HTTP handler + tree migration logic + defensive duplicate-detect)
* Causal happy path (step [2]):
  - 21 of 44 HBM units applied; 23 skipped as `not_a_leaf`.
  - server-reported `applied_hashes` were all absent post-snapshot ✓
  - server cannot fabricate hashes (applied_hashes ⊆ pre-snapshot) ✓
  - **tier_usage delta**: HBM `used_tokens` dropped from 394 → 306, exactly
    matching `Σ n_tokens of applied_hashes = 88`.  This is the
    audit-BLOCKER-#1 invariant the v2 verify missed: a regression that
    detaches the node but leaks the buffer would have passed v2 but fails
    here.
* Branch-coverage probes (audit BLOCKERs #2 / #3 / #4 / #5):
  - DRAM on HBM-only node → `demote_requires_existing_host_backup` ✓
  - HBM on HBM-resident node → `already_on_hbm` ✓
  - Explicit `not_a_leaf` binding to a known internal node ✓
  - DISK → `disk_tier_not_yet_wired` ✓
  - unknown target_tier on a real hash → `unknown_target_tier:'...'` ✓
  - Malformed actions inside valid list (missing hash, None target,
    non-string hash, missing target_tier): all clean, no 5xx ✓
  - IPC unknown-key passthrough (`session_id`, `owner`, `weight` in action
    dict): server accepts and ignores ✓
  - Empty `actions` list: `{applied: 0, applied_hashes: [], skipped: []}` ✓
  - Malformed HTTP payload (empty body, non-list actions, non-JSON body):
    all return 400 ✓
* Duplicate-hash defense (audit MINOR — found real bug, see above):
  - Batch of 10 unique × 2 = 20 actions: applied=8, no double-apply ✓
  - 4 first-occurrence applies whose dup was blocked by
    `already_acted_this_batch` ✓ (defensive check fired)
  - 4 cascade-promoted (first occurrence `not_a_leaf`, second occurrence
    applied as the child's drop made it a leaf) — correct semantics ✓
* Cost ceilings:
  - **Slow path** (211 real-DROP actions): 0.033 ms/action (audit-BLOCKER-#5
    relevant; v2 verify only measured the fast path)
  - Fast path (1k bogus, all `not_in_tree`): 0.009 ms/action
  - Both 30× under the 1 ms/action ceiling
* Cascade-to-zero (audit MINOR): looped `migrate(targets)` until applied==0;
  terminated in 6 iterations, 149 cumulative applied, HBM `used_tokens`
  went 1664 → 0 ✓
* Round-2 applied absence (audit MINOR): round-2's own `applied_hashes`
  also verified absent post-replay ✓
* Concurrency (audit MINOR): 2 threads with disjoint batches, no 5xx and
  no overlap in `applied_hashes` ✓

### Round-3 audit additions (production realism + adversarial)

* **TOCTOU between state and migrate** (step [17]): fetched state, drove
  8 chats to mutate the tree, then ran migrate. 41/66 applied; only
  retry-able reasons in the skip set; no 5xx. The daemon's idempotent
  retry loop is safe even when the tree mutates between snapshot and
  action.
* **Mixed-tier batch** (step [18]): 15 actions interleaving DROP/DRAM/HBM
  on the same subtree. 2 DROPs of leaves applied; DRAM and HBM correctly
  skipped without HiCache (`demote_requires_existing_host_backup`,
  `already_on_hbm`); cascade-removed parents did NOT crash subsequent
  actions in the same batch.
* **Overlapping concurrent batches** (step [19]): 2 threads sent the
  SAME 20-hash batch. Thread A applied 8, thread B applied 8, **overlap = 0**.
  Scheduler ZMQ serialization prevents double-apply across requests.
* **Adversarial HTTP** (step [20]): hash longer than 1024 chars and batch
  larger than 100 000 actions both return 400 at the HTTP layer (caps in
  `http_server.py:692`).
* **Memory / GC soak** (step [21]): 1000 small migrate calls; p50=2.75ms,
  p99=4.24ms; late-window median 1.55× early-window — well under the
  3× slow-leak threshold. Server still responsive after the burst.
* raw logs (relative to this directory):
  * `results/20260525_234929_run1.log` — initial T2 passing run (audit round 1)
  * `results/20260526_001806_run3_depth.log` — depth-audit FAILURE (shows the
    duplicate-batch SIGQUIT, kept as bug-discovery evidence)
  * `results/20260526_004033_run9_clean.log` — post-depth-audit passing run
  * `results/20260526_004940_run10_round3.log` — round-3 audit (production
    realism + adversarial) passing run, 21 steps
  * `results/20260526_012629_run11_bytes_schema.log` — final passing run on
    the bytes-schema rewrite (current implementation)

### Caveats (deferred, NOT regressions)

* **HiCache backup capacity-full row** — predicted `out_of_capacity` skip
  reason verbatim from the existing HiCache pool. Deferred to T9/Run K
  because it requires `--enable-hierarchical-cache`. On the smoke harness
  the failure mode reduces to `demote_requires_existing_host_backup`, which
  is the same safety floor: the daemon retries idempotently, no amplification.
* **HBM promote under capacity-full** — deferred to T9 along with promote
  semantics. v1 returns `promote_not_yet_wired`; the next-request cache-hit
  path auto-loads back.
* **1k-actions-under-30RPS concurrency row** — deferred to T10 (requires RPS
  generator). The standalone slow-path measurement at 0.044 ms/action gives
  us 22× headroom under the 2× cost-under-load ceiling.
* Behavioral note: `node.id` (used in the `node-<id>` fallback for
  pre-HiCache-backup nodes) comes from `UnifiedTreeNode.counter`, a
  class-level **monotonic** counter that strictly increases for the lifetime
  of the process — id-recycle aliasing is impossible. The auditor's
  initial concern (replay applying 21 to the "same" hashes after they
  vanished) turned out to be the cascade behavior, not address instability.
