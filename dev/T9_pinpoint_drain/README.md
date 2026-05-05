# T9 — pinpoint drain (RadixCache `evict_pages_in_range` + cap-barrier sweep)

**Status:** code + unit tests done; awaiting live engine smoke (M2 swarm).
**Triggered by:** post-T8 audit of paper §3.2 vs implementation. The drain step
of the atomic transfer protocol needed to free *exactly* the pages the
planner picked, but the existing wiring called `tree_cache.evict(num_tokens=N)`
which is LRU — semantically misaligned. Two follow-on bugs:

1. **LRU/pinpoint mismatch.** Planner says "drain pages [57,58,59,60]"; LRU
   evict picks any N coldest cached tokens, often outside the cap range.
   Tail tree pages stay tree-referenced. Verify only checks `free_pages`
   overlap with cap range; tree-still-references-cap-page slips through →
   `cuMemUnmap` leaves dangling tree refs → next cache hit on those pages
   reads unmapped VA → CUDA illegal-address. (Exactly the T7 v3 crash.)
2. **Cap-barrier vs `allocator.free`.** Even when LRU happens to pick a
   cap-range page, evicting the tree node calls `allocator.free(value)`,
   which routes the page back into `free_pages` rather than `_capped_pages`.
   Verify catches this and aborts — but the abort wastes a fire opportunity
   in the steady-state path.

## What changed

### `python/sglang/srt/mem_cache/radix_cache.py` — new method

```python
def evict_pages_in_range(self, low: int, high: int) -> int:
    """Evict every evictable tree node whose value page-ids overlap
    [low, high). Returns the number of pages freed. Skips locked nodes
    (lock_ref > 0). Bottom-up: iterates leaves, evicts overlapping ones,
    re-fetches the leaf set, repeats until stable."""
```

Properties:
- **Pinpoint by construction.** Only nodes with `value ∩ [low, high) ≠ ∅`
  are evicted. The set the cross-pool planner asks to drain is exactly
  what gets drained.
- **Whole-node eviction at the boundary.** A node spanning the cap edge
  (e.g., value [55,56,57,58], cap [57,70)) is evicted as a whole.
  Pages 55-56 are released along with 57-58 (collateral cache loss, but
  bounded under T2 placement bias since boundary nodes are rare).
- **Locked-node safety.** `lock_ref > 0` nodes are skipped. Under
  T8's OwnerMap walker, "active wins over cached", so any
  prefix-locked-by-active page is in `pages_to_migrate`, not
  `pages_to_drain` — drain never sees locked nodes in normal operation.

### `python/sglang/srt/budgeter/agent.py` — drain_callback rewrite

```python
def drain_callback(pages):
    low, high = min(pages), max(pages) + 1
    n_freed = tree_cache.evict_pages_in_range(low, high)  # pinpoint
    cap_t = torch.tensor(list(range(low, high)), ...)
    kv_alloc.mark_pages_capped(cap_t)  # sweep freed pages from free → capped
    return n_freed
```

Two-step:
1. **Pinpoint evict** — replaces the prior LRU-by-count call.
2. **Cap-barrier sweep** — `mark_pages_capped` on the cap range catches
   pages that just returned through `allocator.free` during evict. After
   this, every cap-range page is in `_capped_pages`, never in `free_pages`.
   Verify becomes a true sanity check: it should never trip in steady state.

## Tests

```
dev/T9_pinpoint_drain/test/test_evict_pages_in_range.py     6 cases
dev/T9_pinpoint_drain/test/test_drain_callback.py           2 cases
```

All pass. T8's 8 tests still pass after the wiring change.

## What this fixes vs the paper claim

§3.2.5 (transfer execution) claims atomicity-by-construction:
> "every committed transfer moves exactly $n$ pages"

Pre-T9: claim was contingent on `verify` catching residual cache references,
but the verify check only inspected `free_pages` — it could not detect
cap-range pages still owned by tree nodes (since tree-owned pages aren't
in `free_pages`). T9 closes this gap by making drain semantically match
the planner's pages_to_drain set; cap-range pages can no longer end up
tree-referenced after drain.

## Out of scope

- The §3.2.3 admission `argmin{c_1, c_2, c_q}` framing: the engine's
  existing layered logic (own-free → tree-evict → cross-pool fire →
  retract/queue) realizes the same three-way comparison implicitly.
  Implementing the explicit argmin would be a large addition with no
  observable behavior change for the M2 swarm experiments. Deferred.
- Lagrange-bisection vs water-level threshold: equivalent at $|\mathcal{I}| = 2$.
  Paper notes this in a §3.2.4 footnote; no code change.
