# Reservation-style allocator (pre-empty pool tail)

## Motivation
The atomic-migration path (design.tex §3.2.3) handles the case where smart
over-cap selection cannot find N free pages because some target pages hold live
blocks. If we make this case structurally impossible, we can eliminate the
migration path entirely — fewer code paths, simpler liveness story.

## Approach
Each pool's allocator maintains an always-empty *reservation tail* of size
N_max pages, where N_max is the largest single fire we expect (e.g., 30 pages
≈ 60 MiB). The allocator's first-fit lowest-address policy is augmented with a
hard rule: never allocate within the reservation tail. When the active region
gets close to the tail boundary, the reservation slides leftward, freeing
new space at the tail and forcing live blocks out via natural completion.

## Trade-offs
- **Eliminates atomic migration**: the fire path becomes pure cuMemUnmap +
  cuMemMap, no D2D copy ever. Drain success rate trivially 100%.
- **Costs N_max pages × 2 MiB ≈ 60 MiB per pool**: physical HBM permanently
  reserved-but-empty. Conflicts with the "fill HBM" motivation directly:
  ~120 MiB across both pools is bytes that could be holding cache.
- **Reservation slide is non-trivial**: when the reservation moves, it
  effectively forces eviction of any live block in its new tail. This
  re-introduces the same drain problem at a smaller scale (just farther in the
  future).

## When to do it
Only if engine telemetry shows atomic-migration overhead is a measurable cost
fraction (the per-fire ~5 ms for 5 live blocks averaged over 30s control tick
is ~0.02% — far below noise). Currently better to keep migration path for
flexibility; reservation is a brittle optimization.
