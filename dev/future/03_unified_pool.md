# Fully unified pool with type-aware eviction

## Motivation
The current design keeps two physically separate pools (KV, mamba) with
native layouts and routes byte movement between them via VMM remap. A
fundamentally different design: a single physical pool that holds both KV
blocks and recurrent-state snapshots in shared address space, with a unified
priority queue for eviction across types.

If realized, this avoids the actuator entirely — no fire path, no per-fire
wall cost, no drain protocol. Allocation simply pulls from the global free
pool, and eviction picks the lowest-value block of any type.

## Approach
- Single `cuMemAddressReserve` for the entire pool budget.
- Either uniform page size (vLLM v1 style) with padding inside each block, or
  variable-size allocation with a slab/buddy allocator handling fragmentation.
- Eviction priority key encodes per-block recovery cost: HPB(b) × c_σ(b)(L̄)
  where σ(b) is the block's type (KV vs snapshot). Crossover at L* is implicit
  in the c_σ(L) values.
- Per-layer kernel access pattern needs to handle the unified layout. KV's
  [layer × block × token] vs recurrent's [layer × request × hidden] cannot
  share a single physical layout cleanly; one must scatter or pad.

## Trade-offs vs. our 2-pool + VMM design
- **Pro**: no actuator wall, no drain protocol, no migration. Simpler.
- **Con (i)**: layout. KV per-token vs recurrent per-request scaling. Either
  pay padding cost in every block (vLLM v1) or kernel scatter cost.
- **Con (ii)**: CUDA graph capture. Resizing a single pool changes tensor
  shapes; captured graphs invalidate. Either pre-capture at max size with
  unmapped tail (essentially what VMM does, just less explicit) or pay
  re-capture cost (~minutes wall) on every resize.
- **Con (iii)**: eviction value asymmetry. Single LRU key cannot encode
  c_σ(L) cleanly without prefix-length awareness. Either re-implement our
  c_i(L) gate inside the eviction policy, or accept type-blind eviction.

## Why we didn't do it for the paper
All three con's are real engine surgery. Our 2-pool + page-grain VMM design
gets the same byte-fungibility (any byte can serve any pool over time)
without paying any of the three. Unifying is a different paper, not a
strictly better one.

## When to revisit
If a future engine ships from scratch (not retrofitting on SGLang/vLLM), the
unified design might be cleaner to implement than the VMM remap path. For
incrementally extending an existing 2-pool engine, our design wins.
