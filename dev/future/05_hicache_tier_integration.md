# Joint optimization with HiCache tier hierarchy

## Motivation
SGLang already ships `HiMambaRadixCache`, a multi-tier prefix cache that
demotes / promotes both KV and recurrent-state snapshots between HBM, host
DRAM, and (optionally) NVMe disk. This works orthogonally to our inter-pool
budgeter: tier eviction is purely a function of per-tier capacity and
recency, with no coupling to the cross-pool reallocation decision.

Two opportunities for joint optimization:

1. **Demoting from HBM to host DRAM is cheaper than firing across pools.**
   When the HBM is tight and the cost-comparison rule (Eq. nb-direction) is
   choosing between cross-pool transfer and within-pool eviction, a third
   candidate becomes available: demote the chosen cached block to host DRAM
   instead of evicting it. The block is still recoverable on hit (PCIe
   transfer back), so the recovery cost is much lower than re-prefill.
2. **The tier policy itself can consume the cost-aware signal.** Currently
   HiMambaRadixCache's eviction-to-host follows recency. If it consumes the
   same HPB × c_i(L) cost we already compute for inter-pool decisions, the
   tier eviction queue stays consistent with the in-HBM eviction queue.

## Approach
Extend the unified cost model from §3.1 to N tiers, adding a third action:

```
c3(X) = (block.demote-to-host cost)
      = c_i(L) × tier_recovery_factor(host)
      + c_actuator_PCIe
```

`tier_recovery_factor(host)` is the ratio of "PCIe-fetch-back time" to
"re-prefill-time" — typically <<1, so demote-to-host is cheaper than evict
when the block has any chance of being hit. The cost comparison then ranks
{c1, c2, c3, queue} and picks the cheapest.

## Challenges
- HiMambaRadixCache's demote/promote machinery is independent of the
  budgeter; integrating it means adding a callback / signal path.
- The tier wall cost (PCIe latency, ~10s of μs per page) is in a different
  regime than HBM-internal moves (μs). Cost comparison needs careful
  unit handling.
- Multi-tier eviction adds state that the inter-pool gate must consult; the
  control loop's per-tick complexity grows.

## When to do it
After the inter-pool design lands stably. The current paper claims
"orthogonal to HiCache; we adopt it wholesale" — that scopes us out of the
joint optimization. A follow-up paper "Three-tier hybrid memory budgeting"
could combine HBM cross-pool reallocation with host-DRAM demotion under a
single cost model.
