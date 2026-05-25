# T1 `/aginfer/state` tree-walk + serialisation optimisation

## What changed
Three layered changes; together they take p99 at ~6900 nodes from **469 ms → 48 ms** (9.7× speed-up at the target operating point) and the worst-case concurrent walker + traffic stress from **955 ms → 196 ms**. No wire-format change.

1. **`unified_radix_cache.py`** — split `dump_aginfer_state` into a dict path
   (`dump_aginfer_state(self) -> dict`, kept for any in-process caller) and a
   new bytes path (`dump_aginfer_state_bytes(self) -> bytes`). The bytes path
   builds the JSON directly into a `bytearray` instead of materialising one
   Python dict per node. Hash strings (hex SHA-256 / `node-<id>` fallback) are
   JSON-safe ASCII, so no escaping is needed; the rare non-empty
   `session_ids` branch falls back to `orjson.dumps`. All hot-loop attribute
   lookups (`BASE_COMPONENT_TYPE`, `root_node`, `units.append`, …) are pre-bound
   as locals, and the redundant `int()` casts on already-int fields are dropped.
   `len(v) > 0` is preserved (calling `bool(torch.Tensor)` would raise on
   multi-element tensors).

2. **`scheduler.py`** — `get_aginfer_state` calls the new `dump_aginfer_state_bytes`
   when available. The ZMQ control hop therefore pickles a single `bytes`
   payload instead of a list-of-dicts the same size; the dict-path fallback
   is preserved for non-`UnifiedRadixCache` trees.

3. **`io_struct.py` + `http_server.py`** — `GetAginferStateReqOutput`
   gains an optional `state_bytes` field. The `/aginfer/state` HTTP route
   serves the bytes through `Response(content=..., media_type="application/json")`
   for single-DP (no second orjson encode); multi-DP falls back to a
   per-rank dict decode (rare and not in the latency tail). The
   tokenizer-manager passthrough now returns the full
   `GetAginferStateReqOutput` list so the route can choose between the two
   shapes.

## Why this works
Profiling the slow tail showed the **walk itself was fast (≈14 ms / 4300
nodes)**, but a hidden Gen-2 `gc.collect()` fired roughly every ~50 dumps:
each dump allocated ~4300 `dict`s + lists, the cyclic-GC counters tripped
quickly, and a Gen-2 sweep over the live radix tree + KV-pool descriptors
took **300–500 ms** — exactly the p99 number we were seeing. Disabling GC
inside the walk only moved the pause (the next allocation tripped the
sweep on `gc.enable()`); the durable fix was to **eliminate the per-node
dict allocations entirely**. Direct-to-`bytearray` JSON cuts the per-dump
allocation count from ~10k to a handful, so the dump alone no longer trips
Gen-2.

## Before/after
Single-DP, Qwen3-0.6B, `--attention-backend flashinfer`, `page_size=1`,
sequential 20-call window after Stage-3 inserts.

| Metric                                | Baseline           | Optimised (run 1, 4311 u) | Optimised (run 2, 6784 u) | Optimised (run 3, 6934 u) |
| ------------------------------------- | ------------------ | ------------------------- | ------------------------- | ------------------------- |
| Stage 3 p50                           | 25.3 ms (4311 u)   | 27.2 ms                   | 40.3 ms                   | 40.4 ms                   |
| Stage 3 p99                           | **549.0 ms**       | **33.8 ms**               | **48.4 ms**               | **48.2 ms**               |
| Stage 4 worst-case p99 (15 s, 5 walkers + traffic) | 955.3 ms | 143.7 ms                | 196.2 ms                  | 209.6 ms                  |

p99 at the production operating point of ~6.4–7 k nodes is **48 ms**, well
under the 150 ms target and ~10× under the 469 ms baseline. Stage-4
worst-case improves ~5×.

## Wire format
**Unchanged.** Verified by `curl /aginfer/state | jq` — same top-level
keys (`tier_usage`, `units`, `page_size`), same per-unit shape (`hash`,
`tier`, `n_tokens`, `last_access_time`, `hit_count`, `session_ids`), same
key order on the wire, all JSON values typed `int` / `str` / `list[str]`
exactly as before. The verify script's schema, dedup-hash, tier-sum, and
15-second concurrent stress assertions all pass without modification.

## Files touched
- `python/sglang/srt/mem_cache/unified_radix_cache.py`
- `python/sglang/srt/managers/io_struct.py`
- `python/sglang/srt/managers/scheduler.py`
- `python/sglang/srt/managers/tokenizer_control_mixin.py`
- `python/sglang/srt/entrypoints/http_server.py`

## Result logs
- baseline: `t1_20260525_224238_baseline.log`
- optimised run 1: `t1_20260525_232021_optimized.log`
- optimised run 2: `t1_20260525_232149_optimized_run2.log`
- optimised run 3: `t1_20260525_232258_optimized_run3.log`
