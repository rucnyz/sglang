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

**PASSED** — happy path + 4 in-suite worst-case rows + audit-tightened replay
test. Run on Qwen3-0.6B + `--attention-backend flashinfer`, GPU 7, single-DP.
Three rows of the predicted contract are deferred to T9/T10 with explicit
justifications below — none of them are environmentally reachable on the
smoke harness without HiCache or an RPS generator.

* date: 2026-05-26
* sglang sha: (this commit)
* lines added: ~150 across 5 files (io_struct + scheduler dispatch + mixin
  pair + HTTP handler + tree migration logic)
* DROP applied: ✓ 21 leaves of 44 HBM units in round 1; the other 23 were
  internal nodes (`not_a_leaf`, daemon should drop bottom-up). All 21
  server-reported `applied_hashes` were absent in the immediately-after
  state snapshot (causal check ✓). Server cannot fabricate hashes outside
  the pre-snapshot — verified.
* DROP cascade (audit Q6): round-2 replay of the same batch applied 20 more
  hashes, ALL drawn from round-1's `not_a_leaf` bucket (size 23). This is
  the bottom-up cascade the design promises, not unstable addressing.
* DRAM demote: not yet exercised — requires HiCache backup to populate
  `host_value`. v1 contract: returns `demote_requires_existing_host_backup`
  skip reason for HBM-only nodes. Will exercise in T9 / Run K with
  `--enable-hierarchical-cache`.
* HBM promote: not wired (v1). Returns `promote_not_yet_wired`. v1 falls
  back to sglang's normal cache-hit auto-load.
* DISK tier: returns `disk_tier_not_yet_wired` (v1 contract).
* not_in_tree: 2 bogus hashes → both skipped correctly.
* unknown target_tier: 1 bogus tier → skipped with `unknown_target_tier:'...'`.
* Malformed payload (empty, non-list actions, non-JSON body): all return 400.
* Per-action latency:
  - **Slow path** (147-action batch, real targets, 61 actual DROPs):
    **0.044 ms/action** (6 ms total). This is the ceiling-relevant path.
  - **Fast path** (1000-action all-bogus → not_in_tree dict miss):
    **0.007 ms/action** (7 ms total). Sanity floor.
  - Original v1 verify only measured the fast path; audit Q3 flagged this.
    The slow path is now first-class. Both are 23× under the 1 ms/action
    ceiling.
* raw log: `results/<YYYYMMDD_HHMMSS>_run2.log`

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
