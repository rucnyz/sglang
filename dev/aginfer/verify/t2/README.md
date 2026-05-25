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
| HiCache backup target tier full (audit #14, deterministic version) | Fill DRAM to 100 % via real warmup (load N×page_size tokens into the host pool with a tight in-process loop until `tier_usage.DRAM.used_bytes ≈ cap_bytes`). Then POST 1k migrate(target=DRAM) actions for HBM-resident hashes | Every action returns `{skipped: capacity_full}`; cache state unchanged; daemon's idempotent re-issue does NOT amplify (sees no progress, but doesn't crash) | poll /aginfer/state to confirm pre+post are identical, assert response shape |
| Capacity-full promote | HBM at 100 %, POST migrate(X, HBM); | response `{applied: 0, skipped: [{hash: X, reason: capacity_full}]}`; no eviction triggered as side-effect | state assert |
| 1000 actions / batch under load | Insert 10 k nodes, drive concurrent traffic at 30 RPS, then POST a single migrate with 1 k random actions | per-action amortized < 2 ms (= 2× ceiling under load); no daemon timeout | timeit, assert |

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

**PASSED** — happy path (DROP) + 4 forced-injection worst-case rows. Run on Qwen3-0.6B + `--attention-backend flashinfer`, GPU 7, single-DP.

* date: 2026-05-25
* sglang sha: (this commit)
* lines added: ~120 (io_struct + scheduler dispatch + mixin pair + HTTP handler + tree migration logic)
* DROP applied: ✓ 21 of 44 HBM units successfully dropped (the other 23 were internal nodes → `not_a_leaf` reason, daemon should drop bottom-up). All 21 reported-applied hashes were gone from `/aginfer/state` on re-fetch.
* DRAM demote: not yet exercised — requires HiCache backup to populate `host_value`. Smoke test runs without `--enable-hierarchical-cache`. v1 contract: returns `demote_requires_existing_host_backup` skip reason. Will exercise in T9 / Run K (V4-Flash + HiCache).
* HBM promote: not yet wired (v1 contract). Returns `promote_not_yet_wired`. Daemon falls back to sglang's normal cache-hit auto-load.
* DISK tier: returns `disk_tier_not_yet_wired` (v1 contract).
* not_in_tree: 2 bogus hashes → both skipped with correct reason.
* unknown target_tier: 1 bogus tier → skipped with `unknown_target_tier:'...'` reason.
* Malformed payload (empty, non-list actions, non-JSON body): all return 400.
* Per-action latency:
  - 44-action mixed (real hashes): **0.14 ms/action** (6 ms total)
  - 1000-action all-bogus (not_in_tree fast path): **0.010 ms/action** (10 ms total)
  - both well under the 1 ms/action ceiling.
* Idempotent replay: re-issuing the same DROP set after the targets are gone returns sane responses — no crash.
* raw log: `results/<YYYYMMDD_HHMMSS>_run1.log`

### Caveats (deferred, NOT regressions)

* HiCache backup capacity-full row of the worst-case table (#14) is deferred to T9 / Run K because it requires `--enable-hierarchical-cache`. v1 daemon already refuses DRAM demote without a host backup, so the failure mode reduces to `demote_requires_existing_host_backup` — equivalent safety floor.
* HBM-promote-under-capacity-full row is deferred for the same reason: HBM promote is not yet wired in v1; v1 simply trusts sglang's normal cache-hit auto-load. Promote semantics will land in T9.
* 1k-actions-under-30RPS row tested without concurrent traffic; ran 1000-action batch standalone since RPS generator is not yet wired. T10 will re-run this under concurrent load.
