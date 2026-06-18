"""Microbench: MambaPool per-request hot-path ops, per-layer-list vs stacked.

Pins (and re-confirms the fix of) the accidental per-step overhead the arena
MambaPool layout adds over the stacked baseline on a high-concurrency
short-swarm. All timed ops are ~0 in the stacked layout; the per-layer-list
layout pays a Python loop over `num_layers` tensors plus, in `free`, two GPU
ops that sync the device on every call.

Builds a REAL `MambaPool` both ways (no hand-rolled imitation): set
`SGLANG_MAMBA_PERLAYER=1` before construction for the per-layer-list layout,
unset for the stacked single-tensor layout. The shape matches production
(Qwen3-Next on H200): num_layers=24, temporal_state (32, 128, 128) fp32,
conv (8192, 3) bf16, size=443.

The `_capped_slots`-EMPTY case (no cross-fire shrink/drain ever) is the
common no-fire path the default split runs in, and is what `free`/`alloc`
must be free in.

Run: CUDA_VISIBLE_DEVICES=0 .venv/bin/python bench_mamba_pool_ops.py
"""
import os
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

import torch  # noqa: E402

DEVICE = "cuda:0"

# Production Qwen3-Next-on-H200 hot-path geometry.
NUM_LAYERS = 24
SIZE = 443
WARMUP = 30
ITERS = 200

# Batch sizes timed. The pinned regression is LAUNCH overhead, not byte
# bandwidth: a high-concurrency short-swarm fires thousands of COW hits, each
# copying a SMALL number of slots (often a single matched prefix => B=1). At
# small B the per-layer loop's ~48 kernel launches dominate the wall time; at
# large B the actual byte movement (bandwidth-bound, layout-independent)
# masks it. We report both so the launch overhead is visible (small B) and
# the large-B case confirms the fix adds no byte cost.
BATCHES = (1, 4, 147)


def _cache_params(num_layers: int):
    """Real `Mamba2CacheParams` whose shape is exactly the production target:
    temporal (32, 128, 128), conv [(8192, 3)]. dtype defaults to the pool
    defaults (conv bf16, temporal fp32)."""
    from sglang.srt.configs.mamba_utils import (
        Mamba2CacheParams,
        Mamba2StateShape,
    )

    shape = Mamba2StateShape.create(
        tp_world_size=1,
        intermediate_size=7936,   # 7936 + 2*1*128 = 8192 = conv_dim
        n_groups=1,
        num_heads=32,
        head_dim=128,
        state_size=128,
        conv_kernel=4,            # conv_kernel - 1 = 3
    )
    return Mamba2CacheParams(shape=shape, layers=list(range(num_layers)))


def _build_pool(per_layer: bool):
    """Real `MambaPool` in the requested layout. The layout toggle is read at
    construction time, so set/unset the env BEFORE building. `_capped_slots`
    is empty (no cross-fire): size == max_size."""
    if per_layer:
        os.environ["SGLANG_MAMBA_PERLAYER"] = "1"
    else:
        os.environ.pop("SGLANG_MAMBA_PERLAYER", None)
    os.environ.pop("SGLANG_MAMBA_ARENA", None)
    os.environ.pop("SGLANG_ARENA_SHARED", None)

    from sglang.srt.mem_cache.memory_pool import MambaPool

    cache_params = _cache_params(NUM_LAYERS)
    return MambaPool(
        size=SIZE,
        spec_state_size=SIZE,
        cache_params=cache_params,
        mamba_layer_ids=list(range(NUM_LAYERS)),
        device=DEVICE,
        enable_memory_saver=False,
        speculative_num_draft_tokens=None,
        max_size=SIZE,   # max_size == size => _capped_slots empty (no-fire)
    )


def _time_us(fn, warmup=WARMUP, iters=ITERS):
    """Mean microseconds per call, device-synced around the timed region."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def _bench_copy_from(pool, b):
    """copy_from(src, dst) over a batch of b disjoint slot pairs."""
    src = torch.arange(1, b + 1, dtype=torch.int64, device=DEVICE)
    dst = torch.arange(b + 1, 2 * b + 1, dtype=torch.int64, device=DEVICE)
    return _time_us(lambda: pool.copy_from(src, dst))


def _bench_alloc_free(pool, b):
    """alloc(b) then free(b), the per-request churn. `_capped_slots` is empty
    so `free` should take the plain free-list path (no isin, no sync)."""
    def cycle():
        idx = pool.alloc(b)
        pool.free(idx)

    return _time_us(cycle)


def _bench_free(pool, b):
    """free(b) in isolation: isolates the free-side isin + mask_above.any()
    GPU sync (the `_capped_slots`-empty waste). free mutates free_slots by
    appending the batch, so snapshot/restore free_slots OUTSIDE the timed
    region; the timed call is purely `free`."""
    idx = torch.arange(1, b + 1, dtype=torch.int64, device=DEVICE)
    saved = pool.free_slots

    def fn():
        pool.free_slots = saved      # restore (cheap rebind, not timed-bound)
        pool.free(idx)

    us = _time_us(fn)
    pool.free_slots = saved
    return us


def _bench_alloc(pool, b):
    """alloc(b) in isolation: the per-slot zero-init loop. alloc consumes
    free_slots, so snapshot/restore OUTSIDE the timed call."""
    saved = pool.free_slots

    def fn():
        pool.free_slots = saved
        pool.alloc(b)

    us = _time_us(fn)
    pool.free_slots = saved
    return us


def _bench_live_size(pool):
    """live_size property read (used in the cap-accounting hot path)."""
    def read():
        _ = pool.live_size

    return _time_us(read, iters=ITERS * 5)


def main():
    assert torch.cuda.is_available(), "needs a GPU (CUDA_VISIBLE_DEVICES=0)"
    results = {}
    for label, per_layer in (("stacked", False), ("per-layer-list", True)):
        pool = _build_pool(per_layer)
        assert pool._capped_slots.numel() == 0, "bench requires no-fire (empty capped)"
        r = {"layout": "per-layer-list" if pool._mamba_perlayer else "stacked"}
        for b in BATCHES:
            r[("copy_from", b)] = _bench_copy_from(pool, b)
            r[("alloc", b)] = _bench_alloc(pool, b)
            r[("free", b)] = _bench_free(pool, b)
            r[("alloc+free", b)] = _bench_alloc_free(pool, b)
        r["live_size"] = _bench_live_size(pool)
        results[label] = r
        del pool
        torch.cuda.empty_cache()

    st, pl = results["stacked"], results["per-layer-list"]
    print()
    print(f"MambaPool hot-path microbench (num_layers={NUM_LAYERS}, size={SIZE}, "
          f"iters={ITERS})")
    print(f"  temporal (32,128,128) fp32, conv (8192,3) bf16  [device={DEVICE}]")
    print(f"  stacked layout reports as: {st['layout']!r};  "
          f"per-layer reports as: {pl['layout']!r}")
    print()
    hdr = f"  {'op':<16}{'B':>5}{'stacked us':>14}{'per-layer us':>16}{'delta us':>14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for b in BATCHES:
        for op in ("copy_from", "alloc", "free", "alloc+free"):
            d = pl[(op, b)] - st[(op, b)]
            print(f"  {op:<16}{b:>5}{st[(op, b)]:>14.2f}"
                  f"{pl[(op, b)]:>16.2f}{d:>14.2f}")
    d = pl["live_size"] - st["live_size"]
    print(f"  {'live_size':<16}{'-':>5}{st['live_size']:>14.3f}"
          f"{pl['live_size']:>16.3f}{d:>14.3f}")
    print()


if __name__ == "__main__":
    main()
