# T1 — `GET /aginfer/state`

## WHAT WE PROMISED

**Capability**
* Returns JSON with two top-level keys: `tier_usage` (per-tier
  used_bytes + cap_bytes) and `units` (list of all evictable nodes).
* Each unit object has at minimum: `hash, tier, n_tokens, n_bytes,
  last_access_time, hit_count, session_ids`.
* Read-only — does not change cache state.

**Cost ceiling** (revised after measuring on Qwen3-0.6B + flashinfer)
* Operating point: Run F peaked at #cached-token 121 856 / page_size 256
  ≈ 470 leaves; steady-state 100-1 000 leaves is the regime that matters.
* Linear in node count: ~100 μs/node + 200 ms baseline (= Python dict
  construction + ZMQ pickle from scheduler subproc + HTTP orjson).
* Practical bound: < 300 ms p99 at 1 000 nodes; < 1.2 s p99 at 10 000.
  At daemon's 5 s tick this stays under 5 % overhead.
* < ~150 lines added to sglang (io_struct + scheduler dispatch +
  tokenizer mixin pair + HTTP handler + tree-walk helper).

## WORST CASE (forced, must actually run)

| Failure mode | How to force | Predicted floor | Assertion |
|---|---|---|---|
| Torn snapshot under live mutation | Drive 50 concurrent /v1/chat/completions while 5 background threads loop `GET /aginfer/state` at 10 Hz, with cap=64 K so eviction runs hot. | No exceptions; every response is valid JSON; no `hash` duplicated within a single response; `Σ unit.n_bytes per tier ≈ tier_usage.used_bytes` within page-rounding. | parse 500 responses, run schema validator + invariant checker, assert 0 failures |
| Endpoint latency spike (lock contention) | Inject `__post_init__` sleep(0.05) in the walk path (test build flag) | p99 ≤ 200 ms; daemon event-to-migrate p99 ≤ 280 ms (= +200 ms over baseline) | timeit 50 calls under traffic; assert p99 < 200 ms |
| Concurrent walk + concurrent migrate | Same as row 1 + 5 daemon threads issuing POST /aginfer/migrate to random hashes | No deadlock (< 1 s p99); no stale `tier` reported (= migrate completed before state read) | 60-s stress, expect ≥ 95 % of migrates to reflect in next state poll |

## HOW WE VERIFY

Mechanism (no harbor needed). `verify/t1_state_endpoint.py`:

```
1. Launch sglang with a small KV pool (cap 64 K so trees stay tractable).
2. Send ~50 distinct chat-completion warmups with shared + per-req prefixes
   so the radix tree has a known structure (≥ 50 nodes).
3. GET /aginfer/state and assert:
   - 200 OK, schema matches
   - len(units) >= 50
   - sum(u.n_bytes for u in units if u.tier=="HBM")
        ≈ tier_usage.HBM.used_bytes  (within page rounding)
4. Crank prefix variation up to ~10 000 distinct units, measure latency
   of `/aginfer/state` over 20 calls, report p50/p99.
```

## RESULTS
* date: _pending_
* sglang sha:
* lines added (`git diff --stat`):
* schema valid: _pending_
* p50 latency (1 K nodes):
* p99 latency (10 K nodes):
* delta vs ceiling: _pending_
* raw log: `verify/results/t1_<datetime>.log`
