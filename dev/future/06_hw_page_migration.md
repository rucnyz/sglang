# Hardware GPU page migration (UVM-style)

## Motivation
Our atomic migration path (design.tex §3.2.3) uses software-driven D2D copy
plus atomic VA pointer swap. CUDA Unified Memory (UVM) on Hopper supports
hardware page migration via the GPU MMU's page-fault handler: when a kernel
touches an unmapped VA, the driver migrates the page in transparently.

If the cross-pool reallocation could leverage hardware page migration
instead of software copy + atomic swap, the migration wall would drop from
~1 ms/block to driver-internal page-walk time (~10s of μs).

## Why we don't use it currently
1. **UVM page faults are on the kernel hot path**. Every kernel access to an
   unmapped page triggers a fault, ~10-100 μs latency. KV cache access is a
   tight inner loop (every attention layer reads every block); even one
   fault per request can push P99 latency by 100 μs+.
2. **No programmable demote/promote**. UVM's placement is heuristic
   (LRU-ish in driver), not under engine control. Production deployments need
   deterministic latency, which UVM cannot guarantee.
3. **vAttention's reasoning**. The vAttention paper explicitly rejects UVM
   for inference for both reasons above.

## Why this might change
- A future driver could expose a "controlled migration" API that lets the
  engine pre-stage pages without depending on page faults. Effectively,
  what we already do in software (cuMemMap before kernel launch) but with
  hardware-accelerated migration semantics.
- New GPU generations (Blackwell?) may have better TLB / page-walk
  hardware that makes the cold-page penalty negligible.

## Approach (if/when it's viable)
- Replace our software D2D copy with a `cuMemRangeMigrate` (hypothetical)
  call that hardware-migrates the live block's bytes during the cuMemUnmap
  preparation.
- Atomic VA pointer swap becomes implicit (driver handles).
- All hot-path semantics unchanged: kernels still see fully-mapped pages
  because admission gate completes the migration before launch.

## When to revisit
After the next-gen NVIDIA driver or hardware ships with better hardware
migration support that doesn't depend on page-fault hot-path semantics.
Not actionable today.
