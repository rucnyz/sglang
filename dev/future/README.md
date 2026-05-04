# Future Work — Beyond the Ideal-Mode Design

Index of items deliberately scoped out of the current paper design but worth
revisiting. Each file describes motivation, approach, trade-offs, and when it
becomes worth doing.

## In-architecture micro-optimizations
Items that tighten the current 2-pool / single-rank / page-grain VMM design
without architectural change. Marginal returns; do only if a measured workload
needs it.

- [01_lockfree_rcu_migration.md](01_lockfree_rcu_migration.md) — replace cuStreamSynchronize barrier with RCU-style reference counting
- [02_reservation_allocator.md](02_reservation_allocator.md) — pre-empty pool tail to eliminate atomic migration path entirely

## Cross-architecture extensions
Items that change the design space. Not strictly "more ideal" — different
trade-offs, often a new contribution on its own.

- [03_unified_pool.md](03_unified_pool.md) — collapse the two pools into one, type-aware eviction
- [04_multi_rank_tp.md](04_multi_rank_tp.md) — cross-TP-rank coordinated budgeter
- [05_hicache_tier_integration.md](05_hicache_tier_integration.md) — joint optimization with HBM ↔ host DRAM ↔ NVMe tiers
- [06_hw_page_migration.md](06_hw_page_migration.md) — driver-level UVM-style page migration

## Structural boundaries (cannot be removed)
Limits that even ideal-mode cannot cross within current CUDA / hardware
constraints. Documented for honest scope.

- [07_structural_limits.md](07_structural_limits.md) — VA range fixed at boot, HBM outside pools, 2 MiB hardware minimum
