"""Compare GPU kernel perf on torch.zeros tensor vs cuMem-mapped tensor.

In sglang, when SGLANG_ARENA_SHARED=1 the KV cache tensor is backed by
cuMem-mapped pages (via CUDAPluggableAllocator + MemPool path) instead
of torch.zeros (PyTorch's caching allocator).

From the KERNEL's perspective, both are just GPU pointers — `.data_ptr()`
returns a valid CUDA VA in both cases, and reads/writes go through the
same hardware path. The only physical difference is the page table:
- torch.zeros: PyTorch caching allocator block (typically 2MB+ aligned)
- cuMem-mapped: 2 MiB physical handles mapped via cuMemMap

If TLB / coalescing / etc. don't differ, kernel perf on both should be
identical. If they DO differ, that's a candidate for the +3% TTFT.

This test directly compares a representative kernel (gather-scatter,
mimicking attention KV access) on both tensor types.
"""
import ctypes
import statistics
import sys
import time

import torch

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

DEVICE = "cuda:0"
torch.cuda.set_device(0)

# Tensor shape mimicking KV cache: (max_tokens, num_heads, head_dim)
N_TOKENS = 100_000
N_HEADS = 4
HEAD_DIM = 256
DTYPE = torch.bfloat16
BYTES = N_TOKENS * N_HEADS * HEAD_DIM * 2  # bf16
print(f"Tensor: ({N_TOKENS}, {N_HEADS}, {HEAD_DIM}) bf16, total={BYTES/2**20:.0f} MiB")


def make_torch_zeros():
    """Standard PyTorch caching-allocator-backed tensor."""
    return torch.zeros(N_TOKENS, N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)


def make_cumem_backed():
    """Use sglang's from_blob_ext to wrap a cudaMalloc'd region as a
    PyTorch tensor — directly mimics SGLANG_ARENA_FROM_BLOB=1 path,
    which is what the arena does when from_blob mode is on.

    The point: a tensor whose memory is OUTSIDE PyTorch's caching
    allocator. If kernel perf on this tensor matches torch.zeros, then
    arena tensor backing has zero kernel-level cost."""
    from sglang.srt.arena.from_blob_ext import tensor_from_va
    cudart = ctypes.CDLL("libcudart.so")
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    ptr = ctypes.c_void_p(0)
    rc = cudart.cudaMalloc(ctypes.byref(ptr), BYTES)
    assert rc == 0, f"cudaMalloc failed: {rc}"
    cudart.cudaMemset(ptr.value, 0, BYTES)
    t = tensor_from_va(
        va=ptr.value,
        sizes=(N_TOKENS, N_HEADS, HEAD_DIM),
        dtype=DTYPE,
        device_index=0,
    )
    # Keep the raw ptr alive on the tensor so it isn't freed
    t._cudamalloc_keepalive_ptr = ptr
    return t


def bench_kernel(tensor, n_iters=200, seed=0):
    """Mimic attention KV-write + KV-read: gather/scatter with random
    indices over the tensor's first dim."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    # Indices into the KV tokens
    idx = torch.randint(0, N_TOKENS, (1024,), generator=g_cpu, device="cpu").cuda()
    payload = torch.randn(1024, N_HEADS, HEAD_DIM, dtype=DTYPE, device=DEVICE)

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        # KV-write-like scatter
        tensor[idx] = payload
        # KV-read-like gather
        _ = tensor[idx]
        # Mimic attention matmul touching the same memory
        _ = (tensor[idx].to(torch.float32) * payload.to(torch.float32)).sum()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main():
    # Warm
    _ = torch.randn(1024, 1024, device=DEVICE) @ torch.randn(1024, 1024, device=DEVICE)
    torch.cuda.synchronize()

    # Discard first iteration of each phase (warmup); take last 10 of 11
    def bench_phase(label, make_fn, n_reps=11):
        times = []
        for i in range(n_reps):
            t = make_fn()
            elapsed = bench_kernel(t, seed=i)
            times.append(elapsed)
            del t
            torch.cuda.empty_cache()
        # Drop the first (warmup)
        return times[1:]

    # Phase A: torch.zeros, N=10 reps (after warmup)
    print("\n--- Phase A: torch.zeros backing (PyTorch caching allocator) ---")
    times_a = bench_phase("A", make_torch_zeros)
    a_mean = statistics.mean(times_a)
    a_std = statistics.stdev(times_a)
    print(f"  N=10 (warmup discarded), mean={a_mean*1000:.2f}±{a_std*1000:.2f}ms")

    # Phase B: from_blob-wrapped cudaMalloc (closest analog to arena from_blob)
    print("\n--- Phase B: from_blob(cudaMalloc) backing (arena-like, outside caching alloc) ---")
    try:
        times_b = bench_phase("B", make_cumem_backed)
        b_mean = statistics.mean(times_b)
        b_std = statistics.stdev(times_b)
        print(f"  N=10 (warmup discarded), mean={b_mean*1000:.2f}±{b_std*1000:.2f}ms")
    except Exception as e:
        print(f"  Phase B SKIPPED: {e}")
        return

    import math
    delta = (b_mean - a_mean) / a_mean * 100
    se = math.sqrt(a_std**2/10 + b_std**2/10) / a_mean * 100
    print(f"\n{'='*70}")
    print(f"VERDICT — arena (from_blob) vs torch.zeros kernel perf")
    print(f"{'='*70}")
    print(f"  Phase A (torch.zeros):       {a_mean*1000:.2f}±{a_std*1000:.2f}ms")
    print(f"  Phase B (from_blob+cuMalloc): {b_mean*1000:.2f}±{b_std*1000:.2f}ms")
    print(f"  Δ = {delta:+.2f}% ± {se:.2f} SE  (|Δ|/SE = {abs(delta)/max(se,1e-9):.2f})")
    print()
    if abs(delta) < 2 * se:
        print(f"  → No significant kernel-level cost from arena tensor backing")
        print(f"    (|Δ|/SE = {abs(delta)/max(se,1e-9):.2f} < 2).")
        print(f"    The idle_no_regression +3% TTFT is NOT from this — must be either noise or")
        print(f"    something else (e.g. CUDA-graph capture path with arena tensors).")
    else:
        print(f"  → Arena tensor backing IS slower by {delta:+.2f}% on this kernel")
        print(f"    pattern. Plausible source of idle_no_regression's +3% TTFT.")


if __name__ == "__main__":
    main()
