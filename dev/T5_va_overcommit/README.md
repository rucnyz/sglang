# T5: VA overcommit at boot

Task #77. Paper §3.2.1: "For each pool $i$ we call
`cuMemAddressReserve(c_i^cap)` where $c_i^{cap}$ is generously oversized
— typically ~80–90% of total HBM each, so $\sum_i c_i^{cap} >
M_{total}$. VA reservation is virtual and consumes no physical bytes."

## What's broken without T5

T1 dropped chunk granularity from 64 MiB to 2 MiB. The pool's growable
headroom (the VA range past `init_chunks` that the cross-pool actuator
can `cuMemMap` into) is computed as `*_HEADROOM_CHUNKS` (default 4):

```
mamba_max_tokens = tot_aligned + mamba_growth_chunks * tokens_per_chunk
                                       ↑
                           4 chunks @ chunk_size
```

At 64 MiB chunks: 4 × 64 MiB = 256 MiB headroom — plenty to grow into.
At 2 MiB pages: 4 × 2 MiB = **8 MiB headroom** — basically nothing. T1's
notes called this out as "growth budget shrunk; T5 will fix by re-deriving
growth from a byte budget".

When the inter-pool actuator wants to grow KV (e.g., for the long-horizon
M1 workload), it can only map up to 8 MiB of new physical handles into
the KV VA range. The other ~85 GiB of free HBM the actuator could
hypothetically pull from the mamba side has nowhere to go in KV's VA
range — `cuMemMap` would fail because the VA past `init + 8 MiB` was
never reserved.

## Fix

Add two env vars that take precedence over the chunk-count form:

- `SGLANG_ARENA_KV_HEADROOM_BYTES` (default: 80 GiB)
- `SGLANG_ARENA_MAMBA_HEADROOM_BYTES` (default: 80 GiB)

When set, headroom = bytes / chunk_size chunks. Default ensures each
pool's VA range is generously oversized regardless of chunk granularity.
80 GiB is approximately 56% of H200's 143 GB — under T5 the sum of two
pools' VA ranges (each pool init + headroom) can comfortably exceed
total HBM, which is the whole point of overcommit.

Example post-T5 sizing on Qwen3.5-35B-A3B / H200:

| | T1 (no T5) | T5 |
|---|---|---|
| KV chunk_size | 2 MiB | 2 MiB |
| KV init mapped | 1.27 M tokens | 1.27 M tokens |
| KV growth headroom | **8 MiB** (4 chunks) | **80 GiB** (40 960 chunks) |
| KV max_tokens | 1.27 M tokens | ~80 M tokens |
| Mamba init mapped | 366 slots (~700 MiB) | 366 slots |
| Mamba growth headroom | 8 MiB | 80 GiB |
| Mamba max slots | 370 | ~40 K |
| Sum VA reserved | ~26 GiB | ~187 GiB (> 143 GiB phys) |

VA reservation is virtual; the 187 GiB doesn't cost any physical HBM.
Physical handles totaling 143 GiB at boot are mapped into one of the
two ranges; cross-pool transfer remaps them.

## Flag

Two precedence-ordered env vars per pool:
1. `SGLANG_ARENA_*_HEADROOM_BYTES` (T5, new) — overrides chunk count
2. `SGLANG_ARENA_*_HEADROOM_CHUNKS` (legacy) — used when bytes unset

Default behavior changes (T5 default = 80 GiB instead of 4 chunks).
Smoke-friendly opt-out: `SGLANG_ARENA_KV_HEADROOM_BYTES=0` reverts to
legacy chunks.

## How to verify

1. **Boot log inspection**: server log shows
   `MultiTensorArena initialized: ... max_tokens=N` where N reflects
   the new big headroom. Compare on/off via:
   ```
   SGLANG_ARENA_KV_HEADROOM_BYTES=80GB → max_tokens >> 1.27 M
   SGLANG_ARENA_KV_HEADROOM_BYTES=0  → max_tokens = init (legacy 4-chunk)
   ```
2. **Smoke** (`test/test_smoke.sh`): boot with all 5 flags on
   (T1+T2+T3+T4+T5), 5 generates pass, max_tokens log line shows the
   bigger headroom.

End-to-end fire-with-large-grow verification is T7's job (only matters
when actuator actually fires a multi-GB grow).

## Status

- [x] Design note
- [x] Code change: `SGLANG_ARENA_{KV,MAMBA}_HEADROOM_BYTES` env vars
  (`python/sglang/srt/mem_cache/memory_pool.py`); legacy `_CHUNKS` env
  retained for explicit override
- [x] Smoke under T1+T2+T3+T4+T5: PASS, boot 110 s, 5 generates clean
- [x] Boot log shows expanded max_tokens: KV 1.27M → 85.1M (67×),
  mamba 362 → 41,322 (114×). Headroom = 80 GiB per pool by default.
  Sum VA reserved ≈ 174 GiB > 143 GiB physical (overcommit confirmed,
  no extra physical bytes consumed).
- [x] **Env precedence verified** (`test_env_override.sh`, 4 boots):
  default (BYTES & CHUNKS unset) → 83,886,080 ✓; CHUNKS=4 (BYTES unset)
  → 8,192 ✓; BYTES=1 GiB → 1,048,576 ✓; BYTES=0 → 0 ✓ (BYTES wins
  precedence over CHUNKS). Byte→chunks→tokens math byte-exact.

## Followups

T6 (admission-time cost evaluation) and T7 (M2 swarm validation) are
the remaining milestones. T5 is a prerequisite for T7's "real big
grow" to actually succeed (otherwise actuator fires would max out at
the 8 MiB headroom regardless).
