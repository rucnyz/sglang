"""Isolated demo: does PyTorch's user-MemPool disable expandable_segments
and slow down general allocation?

Hypothesis (per multi_tensor_arena.py + pytorch issue 165419): when a
`torch.cuda.MemPool` is in scope, PyTorch silently disables
`expandable_segments` → general allocs (kernel intermediates, matmul
buffers) pay a measurable TTFT penalty. The from_blob arena path
avoids this by not creating a MemPool.

Three phases, N=10 reps each (take median to suppress noise):

  A: baseline — no MemPool, expandable_segments on
  B: with user MemPool active (mimics SGLANG_ARENA_FROM_BLOB=0 path)
  C: NO MemPool (mimics SGLANG_ARENA_FROM_BLOB=1 path) — should
     equal A if hypothesis is right (from_blob doesn't create MemPool,
     so no penalty)

Note: this demo doesn't actually USE the from_blob VA mechanism; it
just tests whether AVOIDING a MemPool restores baseline alloc speed.
That's the exact claim from_blob makes: "no MemPool registered →
expandable_segments stays on → baseline-equivalent alloc speed".

Cost: ~1 min wall on a single GPU; pure-Python; no sglang.
"""
import os
import statistics
import time

import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
print(f"PYTORCH_CUDA_ALLOC_CONF = {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
N_REPS = 10
N_ITERS = 200


def bench(seed=42):
    """One trial: varying-size matmul allocs + rolling cache."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    g_gpu = torch.Generator(device="cuda").manual_seed(seed)
    cache = []
    start = time.perf_counter()
    for _ in range(N_ITERS):
        size = 512 + int(torch.randint(0, 1024, (1,), generator=g_cpu).item())
        x = torch.randn(size, size, device="cuda", generator=g_gpu)
        y = x @ x
        cache.append(y)
        if len(cache) > 30:
            cache.pop(0)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    del cache
    return elapsed


def bench_n(label, n, ctx_factory=None):
    """Run bench n times under an optional context, return list of times."""
    times = []
    for i in range(n):
        if ctx_factory is None:
            t = bench(seed=i)
        else:
            with ctx_factory():
                t = bench(seed=i)
        times.append(t)
    return times


def stats(label, times):
    mean = statistics.mean(times)
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    median = statistics.median(times)
    print(f"  {label:30s} mean={mean*1000:6.2f}±{stdev*1000:4.2f} ms  "
          f"median={median*1000:6.2f} ms  (N={len(times)})")
    return mean, stdev, median


def main():
    # Warm CUDA context
    _ = torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()
    # One warmup bench (discarded)
    _ = bench(seed=0)

    print(f"\nN_REPS={N_REPS}, N_ITERS={N_ITERS}\n")

    # ---- Phase A: baseline ----
    print("--- Phase A: baseline (NO MemPool, expandable_segments on) ---")
    times_a = bench_n("baseline", N_REPS)
    a_mean, a_std, a_med = stats("A baseline", times_a)

    # ---- Phase B: with user MemPool active ----
    print("\n--- Phase B: WITH user MemPool active (expandable_segments forcibly off) ---")
    try:
        from torch.cuda.memory import MemPool
    except ImportError:
        from torch.cuda import MemPool
    mp = MemPool()

    def _mp_ctx():
        return torch.cuda.use_mem_pool(mp)
    times_b = bench_n("with-MemPool", N_REPS, ctx_factory=_mp_ctx)
    b_mean, b_std, b_med = stats("B with-MemPool", times_b)

    # ---- Phase C: no MemPool (proves Phase B's slowdown comes FROM the MemPool) ----
    # MemPool object `mp` still exists in process. Question: does just
    # EXISTING (without `use_mem_pool` scope) cause penalty? Per PyTorch
    # behavior, the disable triggers on use_mem_pool entry. Outside scope
    # PyTorch should resume expandable_segments.
    print("\n--- Phase C: MemPool exists but NOT active (= from_blob mode: no use_mem_pool scope) ---")
    times_c = bench_n("no-active-MemPool", N_REPS)
    c_mean, c_std, c_med = stats("C no-active-MemPool", times_c)

    # ---- Stats ----
    import math
    def pct_diff(x_mean, x_std, x_n, y_mean, y_std, y_n):
        pp = (x_mean - y_mean) / y_mean * 100
        se = math.sqrt(x_std**2 / x_n + y_std**2 / y_n) / y_mean * 100
        return pp, se

    pp_b, se_b = pct_diff(b_mean, b_std, len(times_b), a_mean, a_std, len(times_a))
    pp_c, se_c = pct_diff(c_mean, c_std, len(times_c), a_mean, a_std, len(times_a))

    print(f"\n{'='*70}")
    print(f"VERDICT (N={N_REPS} reps, mean±SE)")
    print(f"{'='*70}")
    print(f"  A baseline                  : {a_mean*1000:6.2f}±{a_std*1000:.2f} ms")
    print(f"  B inside use_mem_pool scope : {b_mean*1000:6.2f}±{b_std*1000:.2f} ms  "
          f"Δ={pp_b:+5.2f}±{se_b:.2f} pp")
    print(f"  C outside use_mem_pool scope: {c_mean*1000:6.2f}±{c_std*1000:.2f} ms  "
          f"Δ={pp_c:+5.2f}±{se_c:.2f} pp")
    print()
    # Significance: need |Δ| > 2 × SE
    b_sig = abs(pp_b) > 2 * se_b
    c_sig = abs(pp_c) > 2 * se_c
    print(f"  B vs A significant?  {b_sig}  (|Δ|/SE = {abs(pp_b)/max(se_b,1e-9):.2f})")
    print(f"  C vs A significant?  {c_sig}  (|Δ|/SE = {abs(pp_c)/max(se_c,1e-9):.2f})")
    print()
    if b_sig and pp_b > 0 and not c_sig:
        print(f"  → CONFIRMED: MemPool ACTIVE causes slowdown ({pp_b:+.2f}%); ")
        print(f"    outside-scope (== from_blob mode) is statistically ")
        print(f"    indistinguishable from baseline.")
        print(f"  → from_blob path SHOULD eliminate the ~3% idle_no_regression TTFT regression.")
    elif b_sig and pp_b > 0 and c_sig and pp_c > 0:
        print(f"  → MemPool active is slower ({pp_b:+.2f}%) AND just having")
        print(f"    the MemPool object also regresses ({pp_c:+.2f}%) — investigate.")
    else:
        print(f"  → No significant MemPool penalty detected on this PyTorch/GPU.")


if __name__ == "__main__":
    main()
