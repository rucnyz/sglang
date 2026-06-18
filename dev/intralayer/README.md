# intralayer — per-pool eviction policy

Per-pool (intra-pool) eviction policy. Replaces sglang's default
recency-LRU with **LPB** (loss-per-byte) on the radix tree: each
evictable node scores `hits-per-byte`, lowest score evicts first.
Bigger and more-frequently-hit prefixes survive longer.

The full multi-engine writeup (design + per-engine implementation
review + cross-engine results) lives in the companion repo at
`vllm-songyang:dev/intralayer/`. This README is the sglang-side
quick-reference.

## Drivers + workloads

| file | role |
|---|---|
| `compare_lru_lpb.py` | Phase A → H pipeline driver (engine-agnostic phase design from `vllm-songyang:intralayer/scenarios.md`). Compares recency-LRU vs LPB on the same workload trace. Multi-trial; emits jsonl + `.out` per trial. |
| `skewed_bench.py` | Skewed-popularity stress workload that exposes the **LPB-favorable** regime: 12 groups × Zipf(α=1.5) traffic × `--max-mamba-cache-size 8` (forced tight mamba pool). Out-of-the-box driver for the headline LPB-vs-recency comparison on sglang. |
| `skewed_run.sh` | Convenience wrapper around `skewed_bench.py` (boot server + run bench × N trials + tear down). |
| `gsp_bench.sh` | GSP (generated-shared-prefix) bench driver from sglang's built-in `bench_serving --dataset-name generated-shared-prefix`. Predecessor of `skewed_bench.py`; kept for reproduction of historical results. |
| `runs/`               | Local intralayer run artefacts. The cross-engine result hub (with vLLM results, sglang multi-version sweeps, figures) lives in `vllm-songyang:dev/intralayer/runs/`. |

## Source

| layer | location |
|---|---|
| LPB scoring (mamba) | `mamba_radix_cache.TreeNode.eviction_priority` — `ℓ(b) = n_b · (c_kv + c_m) / B_b` |
| LPB scoring (KV-only) | `radix_cache.TreeNode.lpb_priority` — `ℓ(b) = n_b · c_kv / B_b`; `radix_cache.LPBStrategy.get_priority` returns `(lpb_priority, last_access_time)` |
| Eviction selectors (mamba) | `mamba_radix_cache`: `_iter_mamba_victims` (single victim-ordering source: LPB cold-tail run then lazy Phase-2 heap, or LRU prev-chain; consumed by both `evict_mamba` and `_plan_mamba_eviction`), reusing `_lpb_build_eviction_heap` / `_lpb_pop_eviction_victim`; KV side: `_plan_full_eviction`, `_lpb_build_full_eviction_heap`, `_lpb_pop_full_eviction_victim` |
| Policy gate | `mamba_radix_cache.MambaRadixCache._should_use_lpb` (reads `params.eviction_policy`); `evict_full` / `evict_mamba` consult it. Plain `RadixCache.evict` gates on `isinstance(eviction_strategy, LPBStrategy)`. |
| Hit recording (`n_b`) | `record_hit` is called in `_match_prefix_helper` of `RadixCache` + `HiRadixCache` (gated on `LPBStrategy`) and `MambaRadixCache` + `HiMambaRadixCache` (always). `_split_node` carries `_hit_times` (and `hit_count` on the mamba side) onto the shared-prefix node. **`SWARadixCache` has no LPB plumbing — its `__init__` raises `NotImplementedError` on `--radix-eviction-policy lpb` (fail-loud, not silent LRU); SWA support tracked in #261.** |
| Engine knob | `--radix-eviction-policy lpb` (default `lru` → recency-LRU). Single source of truth across plain / hybrid / hierarchical caches (#181); the old `SGLANG_LPB_LRU` env toggle was removed. |
| Window | `SGLANG_LPB_WINDOW_S=60.0` (driver uses 3600 to avoid expiration) |
| `_hit_times` deque cap | `SGLANG_LPB_HIT_DEQUE_MAXLEN=4096` |
| Per-mamba-slot bytes | read at `MambaRadixCache.__init__` from `mamba_pool.mamba_cache.mem_usage_bytes() / mamba_pool.size` (e.g. 32 MB on Qwen3.5-35B-A3B util=0.9) |

## Scoring formula (current)

```python
def eviction_priority(self) -> float:
    n_hits = self.hits_in_window()  # windowed deque length, capped at maxlen
    size_bytes = 0
    if self.value is not None:
        size_bytes += int(self.value.numel())
    if self.mamba_value is not None:
        size_bytes += int(self.mamba_value.numel()) * TreeNode.lpb_bytes_per_mamba_slot
    if size_bytes == 0:
        return float("inf") if n_hits > 0 else 0.0
    return n_hits / size_bytes
```

Eviction selector: O(n) heap build + O(log n) heappop per victim,
total `O(n + K log n)` for K evictions per `evict_mamba` /
`evict_full` call. Applied to **both** the mamba-snapshot path and
the KV-page path.

## Result summary (full data in vllm-songyang)

| workload variant | LPB vs recency (mean req TTFT) | source |
|---|---|---|
| Path A (`compare_lru_lpb.py`, 4 optimization rounds, n=3) | tied within noise on the canonical pipeline | `vllm-songyang:dev/intralayer/runs/sglang/` |
| GSP (8 groups uniform Zipf) | tied (LPB's `−19.77 %` prelude headline was n=1; n=3 mean = `−0.86 %`) | `vllm-songyang:dev/intralayer/runs/sglang_gsp/` |
| **Skewed (12 groups Zipf α=1.5, mamba-cache-size=8)** | **−17.5 % mean / −27 % median / +23 pp cache hit** (LPB wins decisively) | `vllm-songyang:dev/intralayer/runs/sglang_skewed/` |

For the **skewed-popularity** workload the conditions LPB needs are
met (free-leaf snapshots + skewed hit counts + forced mamba pressure).
For default Path A / GSP, sglang's per-node radix-tree LRU already
encodes prefix-level recency well enough that LPB's hit-count signal
adds nothing on top — see `vllm-songyang:dev/intralayer/sglang.md`
§"Why eviction outcomes converge on sglang" for the full structural
explanation.

## Reproduce — skewed-popularity (the headline LPB win)

```bash
cd /scratch/yuzhou/projects/sglang
CUDA_VISIBLE_DEVICES=<gpu> NUM_TRIALS=3 \
    bash dev/intralayer/skewed_run.sh
```

Results land in `vllm-songyang:dev/intralayer/runs/sglang_skewed/`.

## Cross-engine references

| topic | document |
|---|---|
| Engine-agnostic phase design (workload pipeline A → H) | `vllm-songyang:dev/intralayer/scenarios.md` |
| vLLM-side LPB review (driver, results, Phase H win) | `vllm-songyang:dev/intralayer/vllm.md` |
| sglang-side LPB review (4-round optimization journey + 7-variant measurements + skewed-popularity headline) | `vllm-songyang:dev/intralayer/sglang.md` |
| Cross-engine result hub (vLLM + sglang runs, figures) | `vllm-songyang:dev/intralayer/` |

This README is intentionally short — the deep design + measurement
work is upstream of both engines and the right home for it is the
cross-engine hub. Anything you'd write here that's not pointer-to-
upstream belongs there.
