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
* date: _pending_
* sglang sha:
* lines added (`git diff --stat`):
* drop applied: _pending_
* demote applied: _pending_
* promote applied: _pending_
* per-action latency (mean): _pending_
* delta vs ceiling: _pending_
* raw log: `verify/results/t2_<datetime>.log`
