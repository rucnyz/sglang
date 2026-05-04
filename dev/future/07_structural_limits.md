# Structural Limits (cannot be removed)

These are constraints that even the ideal-mode implementation cannot cross
without changes outside the engine's control. Documented for honest scope.

## VA range fixed at boot

`cuMemAddressReserve(c_i^cap)` fixes each pool's virtual-address upper bound
at boot. The tensor wraps `[V_i, V_i + c_i^cap)` and exposes that as
`tensor.data_ptr()`; captured CUDA graphs hard-code this pointer. Re-reserving
a larger VA range produces a different pointer, invalidating every captured
graph (re-capture is several minutes wall on a 35B model).

We mitigate by **VA overcommit at boot**: reserve $\sum_i c_i^{\text{cap}} >>
M_{\text{total}}$, so each pool's individual maximum is generous even though
they cannot all be at maximum simultaneously. This pushes the limit out of the
common range but does not eliminate it. A workload that wants pool A to grow
beyond `c_A^cap` is structurally impossible without a server restart.

## HBM outside the two pools is invisible

Model weights (~70 GB on Qwen3.5-35B-A3B / TP=1), activation scratch buffers
(~10–20 GB), and CUDA-graph workspace are not in the inter-pool layer's
visibility. Even if a workload would benefit from "borrowing" weight or
scratch HBM into the cache pool, our budgeter cannot do this.

To touch these, you'd need:
- **Weight offload**: streaming weights from host DRAM during decode (e.g.,
  ZeRO-Offload-style). Expert mixture models partly do this with
  per-expert offload, but inference engines don't.
- **Activation streaming**: similarly, but activations are live and
  hot-path.

Both are different sub-systems. Our paper scope is fixed at the
"pool-reallocatable" subset of HBM.

## CUDA VMM minimum granularity 2 MiB on H200

`cuMemCreate` allocates physical handles in multiples of 2 MiB on Hopper
(the OS-level page size for GPU memory). Sub-2-MiB granularity is not
exposed. If a future hardware/driver supported, e.g., 4 KiB pages, the
random-pin-collision probability for live blocks would drop another 500×;
drain success rate would be effectively 100% even without smart selection.

This is a hardware/driver limit. We have no path to push past it.

## CUDA graph hard binding to tensor pointer + shape

Captured CUDA graphs compile down to tensor data pointers and tensor shapes.
Both are fixed at capture time. Our entire VMM design is engineered around
this constraint: the tensor pointer never moves, the tensor shape never
changes; only the physical memory under the pointer flips. Any approach that
needs to change tensor shape (e.g., to grow a pool past its VA cap) forces
re-capture.

This is a CUDA constraint, not a Prelude constraint, but it shapes every
choice in §3.2.

## Cache cannot be promoted to fill gaps in compute pressure

Our cost model is purely memory-aware: it ranks candidates by
compute-saved-per-byte. If the bottleneck shifts to compute (e.g., a workload
where the attention all-reduce is the binding constraint), reallocating
memory bytes does nothing for throughput. The budgeter's $\hat V_i'$
estimate would correctly fall to zero in this regime, and the gate would
correctly suppress fires; but Prelude has no positive action here.
Compute-aware scheduling is a different sub-system (out of paper scope).
