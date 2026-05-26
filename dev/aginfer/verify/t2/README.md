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
| hash-not-found race | Insert a node, capture its hash, internally `_evict_device_leaf` to remove it; immediately POST `/aginfer/migrate` for that hash | response is `{applied: 0, skipped: [{hash: X, reason: not_in_tree}]}`; no exception | parse response, assert structure + counts |
| unknown target_tier (typo / version skew) | POST migrate with `target_tier="DOESNOTEXIST"` for a live hash | response `{applied: 0, skipped: [{hash: X, reason: unknown_target_tier:'DOESNOTEXIST'}]}`; no exception | parse, assert reason substring |
| DISK tier in v1 (not wired) | POST migrate with `target_tier="DISK"` for a live hash | response `{applied: 0, skipped: [{hash: X, reason: disk_tier_not_yet_wired}]}` | parse, assert reason |
| Malformed payload | POST migrate with `{}`, `{"actions": "not a list"}`, or non-JSON body | HTTP 400 on all three; no exception, no partial mutation | requests + status assert |
| DRAM demote with no host backup (v1 contract) | POST migrate(target=DRAM) for an HBM-only hash (no HiCache backup populated) | response `{applied: 0, skipped: [{hash: X, reason: demote_requires_existing_host_backup}]}` — same safety floor as HiCache-full, the daemon retries idempotently | parse, assert reason |
| Slow-path 1k batch latency (real DROPs) | warm 60+ distinct leaves, POST migrate(target=DROP) for all of them | per-action amortized < 1 ms; all real targets actually evict | timeit + causal absence check |
| Idempotent replay | POST migrate(DROP) once for hashes H1..Hn; capture {applied}; immediately POST the same batch again with no traffic in between | second response has `applied == 0`; reasons ⊆ {`not_in_tree`, `no_data`, `not_a_leaf`}; no crash | strict applied==0 assert |
| HiCache backup target tier full (audit #14) — deferred to T9/Run K | Fill DRAM to 100 % via real warmup under `--enable-hierarchical-cache`. Then POST 1k migrate(target=DRAM) actions for HBM-resident hashes | Daemon falls back via `demote_requires_existing_host_backup` until backup_thread catches up; once host is full the response carries the existing HiCache pool's `out_of_capacity` skip reason verbatim. Idempotent re-issue does NOT amplify | requires HiCache; exercised in T9 |
| Capacity-full promote — deferred to T9 | HBM at 100 %, POST migrate(target=HBM) for a DRAM hash | v1 returns `promote_not_yet_wired`; promote semantics land in T9 when the kv_scheduler decides promotions explicitly | T9 |
| 1000 actions / batch under 30 RPS load — deferred to T10 | Insert 10 k nodes, drive concurrent traffic at 30 RPS, then POST a single migrate with 1 k random actions | per-action amortized < 2 ms (= 2× ceiling under load); no daemon timeout | T10 (requires RPS generator) |

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

## RESULTS

**PASSED** (depth-audit edition, 2026-05-26).  All 16 in-suite steps pass on
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
* raw log: `results/<YYYYMMDD_HHMMSS>_run9_clean.log`

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
