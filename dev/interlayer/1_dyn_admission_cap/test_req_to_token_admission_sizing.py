"""ReqToTokenPool admission sizing: verify max_size = (1+r)/r × max_num_reqs.

Tests the _resolve_max_admission_size formula across mamba_full_memory_ratio
values, confirming:
  1. The formula produces correct multipliers.
  2. ReqToTokenPool boots with the right size/max_size.
  3. VA arena is used iff max_size > size (dynamic-cap mode).
  4. grow/shrink works within the reserved range.
  5. Real GPU memory overhead is bounded (only the pre-created handles for
     the extra rows, not the full max_size × row_bytes).
"""
import os
import sys
import subprocess

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    _resolve_max_admission_size,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTEXT_LEN_SMALL = 4096
CONTEXT_LEN_REAL = 262144
MAX_NUM_REQS = 244


def gpu_used_mb(gpu_id=0):
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits", "-i", str(gpu_id)],
        capture_output=True, text=True,
    )
    return int(r.stdout.strip())


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0])
def test_formula_multiplier(ratio):
    max_size = _resolve_max_admission_size(MAX_NUM_REQS, ratio)
    expected = int(MAX_NUM_REQS * (1 + ratio) / ratio)
    assert max_size == expected, f"ratio={ratio}: got {max_size}, expected {expected}"
    assert max_size >= MAX_NUM_REQS


def test_formula_zero_ratio():
    assert _resolve_max_admission_size(MAX_NUM_REQS, 0.0) == MAX_NUM_REQS


@pytest.mark.parametrize("ratio", [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0, 0.0])
def test_pool_boot_and_grow(ratio):
    max_size = _resolve_max_admission_size(MAX_NUM_REQS, ratio)
    pool = ReqToTokenPool(
        size=MAX_NUM_REQS,
        max_context_len=CONTEXT_LEN_SMALL,
        device=DEVICE,
        enable_memory_saver=False,
        max_size=max_size,
    )
    assert pool.size == MAX_NUM_REQS
    assert pool.max_size == max_size
    assert len(pool.free_slots) == MAX_NUM_REQS
    assert pool.req_to_token.shape[0] >= max_size + 1

    expect_arena = max_size > MAX_NUM_REQS
    assert (pool._va_arena is not None) == expect_arena

    if max_size > MAX_NUM_REQS:
        grow_target = min(MAX_NUM_REQS + 10, max_size)
        pool.grow(grow_target)
        assert pool.size == grow_target
        assert len(pool.free_slots) == grow_target

        pool.shrink(MAX_NUM_REQS)
        assert pool.size == MAX_NUM_REQS


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("ratio,max_extra_mb", [
    (0.9, 300),
    (0.5, 600),
])
def test_gpu_memory_overhead(ratio, max_extra_mb):
    gpu_id = torch.cuda.current_device()
    max_size = _resolve_max_admission_size(MAX_NUM_REQS, ratio)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    m0 = gpu_used_mb(gpu_id)

    pool_default = ReqToTokenPool(
        size=MAX_NUM_REQS, max_context_len=CONTEXT_LEN_REAL,
        device="cuda", enable_memory_saver=False,
    )
    torch.cuda.synchronize()
    m1 = gpu_used_mb(gpu_id)
    default_cost = m1 - m0
    del pool_default
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    m2 = gpu_used_mb(gpu_id)
    pool_hima = ReqToTokenPool(
        size=MAX_NUM_REQS, max_context_len=CONTEXT_LEN_REAL,
        device="cuda", enable_memory_saver=False,
        max_size=max_size,
    )
    torch.cuda.synchronize()
    m3 = gpu_used_mb(gpu_id)
    hima_cost = m3 - m2
    del pool_hima
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    extra = hima_cost - default_cost
    assert extra <= max_extra_mb, (
        f"ratio={ratio}: extra GPU memory {extra} MB > {max_extra_mb} MB cap. "
        f"default={default_cost} MB, HiMA={hima_cost} MB"
    )


if __name__ == "__main__":
    passed = failed = skipped = 0

    for ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0]:
        test_formula_multiplier(ratio)
        passed += 1
    test_formula_zero_ratio()
    passed += 1
    print(f"formula: {passed} passed")

    for ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0, 0.0]:
        test_pool_boot_and_grow(ratio)
        passed += 1
    print(f"boot+grow: {passed} passed")

    if torch.cuda.is_available():
        for ratio, cap in [(0.9, 300), (0.5, 600)]:
            test_gpu_memory_overhead(ratio, cap)
            passed += 1
        print(f"gpu memory: {passed} passed")
    else:
        skipped += 2
        print(f"gpu memory: skipped (no CUDA)")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
