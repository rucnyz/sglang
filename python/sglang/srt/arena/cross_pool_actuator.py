"""
Phase 2e.5.6 — CrossPoolTransferActuator: KV ↔ mamba physical-handle migration.

Sits on top of two MultiTensorArena instances that share one
SharedHandlePool (Phase 2e.5.6.1). Exposes:

  - kv_to_mamba_chunks(n_per_kv_subpool):
        Shrinks each KV sub-pool by `n` chunks (frees `n * n_kv_subpools`
        handles into the shared pool), then grows each mamba sub-pool
        by `floor(n_freed / n_mamba_subpools)` chunks. Any remainder
        stays in the shared pool's free list and is available for the
        next call (or the reverse direction).

  - mamba_to_kv_chunks(n_per_mamba_subpool):
        Symmetric.

The asymmetry — KV has `n_kv_layers * 2` sub-pools (k, v per layer),
mamba has `n_mamba_layers * 1` (temporal per layer) — is handled by
keeping per-call grow rounding to the floor. Engineering rationale:
the planner's "budget" is in tokens-of-capacity per pool, which maps
to "live capacity = min mapped chunks * tokens_per_chunk" inside each
MultiTensorArena. The min-across-subpools requirement is what forces
us to grow (or shrink) all sub-pools by the same amount.

The planner is policy-side; this actuator only handles the mechanical
"move these many chunks from arena A to arena B" operation.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.arena.chunk_arena import cross_arena_transfer

if TYPE_CHECKING:
    from sglang.srt.arena.multi_tensor_arena import MultiTensorArena
    from sglang.srt.arena.chunk_arena import SharedHandlePool


logger = logging.getLogger(__name__)


class CrossPoolTransferActuator:
    """KV ↔ mamba chunk migration over a shared handle pool."""

    def __init__(
        self,
        kv_arena: "MultiTensorArena",
        mamba_arena: "MultiTensorArena",
        shared_pool: "SharedHandlePool",
    ) -> None:
        self.kv = kv_arena
        self.mamba = mamba_arena
        self.shared = shared_pool

        if kv_arena._arena._external_pool is not shared_pool:
            raise ValueError("kv_arena does not use the provided shared_pool")
        if mamba_arena._arena._external_pool is not shared_pool:
            raise ValueError("mamba_arena does not use the provided shared_pool")

        self.n_kv_subpools = kv_arena.n_layers * kv_arena.n_kinds
        self.n_mamba_subpools = mamba_arena.n_layers * mamba_arena.n_kinds

        logger.info(
            "CrossPoolTransferActuator: kv_subpools=%d, mamba_subpools=%d, "
            "shared_handles=%d, free=%d",
            self.n_kv_subpools, self.n_mamba_subpools,
            self.shared.total_count(), self.shared.free_count(),
        )

    # ------------------------------------------------------------------

    def _all_subpool_names(self, mta: "MultiTensorArena") -> list[str]:
        n = mta.n_layers * mta.n_kinds
        return [mta._pool_name(i) for i in range(n)]

    def _do_transfer(
        self,
        src: "MultiTensorArena",
        dst: "MultiTensorArena",
        n_per_dst_subpool: int,
        direction_label: str,
    ) -> dict:
        """Grow every dst sub-pool by `n_per_dst_subpool` chunks; this
        requires shrinking each src sub-pool by
        `ceil(n_per_dst_subpool * n_dst_subpools / n_src_subpools)`. Any
        leftover unmapped handles stay in the shared pool's free list and
        are available for the next call.

        Why dst-anchored (not src-anchored):
          live capacity of an MTA is min mapped chunks across its
          sub-pools. If we shrank src by 1 chunk per src-subpool but
          src has more sub-pools than dst, dst would only grow by
          floor(n_src/n_dst) per dst-subpool, which is 0 when n_src <
          n_dst (e.g., KV's 20 sub-pools < mamba's 30). Making the
          caller specify dst-side guarantees the transfer always
          actually grows dst.

        Returns: stats dict.
        """
        if n_per_dst_subpool <= 0:
            raise ValueError(
                f"n_per_dst_subpool={n_per_dst_subpool} must be > 0"
            )

        src_names = self._all_subpool_names(src)
        dst_names = self._all_subpool_names(dst)
        n_src = len(src_names)
        n_dst = len(dst_names)

        # How many chunks must each src sub-pool shed to free enough
        # handles for the dst grow? ceil(needed / n_src).
        needed = n_per_dst_subpool * n_dst
        n_per_src_subpool = (needed + n_src - 1) // n_src

        free_before = self.shared.free_count()
        unmapped_total = 0
        for name in src_names:
            unmapped_total += src._arena.shrink(name, n_per_src_subpool)
        free_after_shrink = self.shared.free_count()

        # Grow every dst sub-pool by exactly n_per_dst_subpool. Anything
        # we couldn't grant (because src didn't free enough handles) is
        # tracked in the stats; the leftover stays in the shared free
        # list for the next call.
        granted_total = 0
        for name in dst_names:
            granted_total += dst._arena.grow(name, n_per_dst_subpool)

        free_after_grow = self.shared.free_count()

        stats = {
            "direction": direction_label,
            "n_per_src_subpool": n_per_src_subpool,
            "n_per_dst_subpool": n_per_dst_subpool,
            "src_subpools": n_src,
            "dst_subpools": n_dst,
            "unmapped_total": unmapped_total,
            "granted_total": granted_total,
            "free_before": free_before,
            "free_after_shrink": free_after_shrink,
            "free_after_grow": free_after_grow,
            "kv_capacity_tokens": self.kv.current_capacity_tokens(),
            "mamba_capacity_tokens": self.mamba.current_capacity_tokens(),
        }
        logger.info(
            "CrossPoolTransferActuator.%s: shrank %d/src=%d → freed %d, "
            "grew %d/dst=%d → consumed %d, leftover free %d → KV cap=%d tok, "
            "mamba cap=%d tok",
            direction_label,
            n_per_src_subpool, n_src, unmapped_total,
            n_per_dst_subpool, n_dst, granted_total,
            free_after_grow,
            stats["kv_capacity_tokens"], stats["mamba_capacity_tokens"],
        )
        return stats

    # ------------------------------------------------------------------

    def kv_to_mamba_chunks(self, n_per_mamba_subpool: int) -> dict:
        """Grow mamba by `n` chunks per mamba sub-pool, sourcing handles
        from KV via the shared pool. KV sheds
        `ceil(n * n_mamba_subpools / n_kv_subpools)` chunks per KV
        sub-pool (rounded up so dst grows fully). Any leftover handles
        stay in the shared free list.
        """
        return self._do_transfer(
            src=self.kv, dst=self.mamba,
            n_per_dst_subpool=n_per_mamba_subpool,
            direction_label="kv_to_mamba",
        )

    def mamba_to_kv_chunks(self, n_per_kv_subpool: int) -> dict:
        """Symmetric: grow KV by `n` chunks per KV sub-pool, sourcing from
        mamba. See `kv_to_mamba_chunks`.
        """
        return self._do_transfer(
            src=self.mamba, dst=self.kv,
            n_per_dst_subpool=n_per_kv_subpool,
            direction_label="mamba_to_kv",
        )

    # ------------------------------------------------------------------

    def state(self) -> dict:
        return {
            "kv_capacity_tokens": self.kv.current_capacity_tokens(),
            "mamba_capacity_tokens": self.mamba.current_capacity_tokens(),
            "shared_total_handles": self.shared.total_count(),
            "shared_free_handles": self.shared.free_count(),
        }
