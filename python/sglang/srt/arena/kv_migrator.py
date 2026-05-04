"""T8 step 5 — KV page migrator.

Implements the migrate side of the FirePlan: copies KV-cache slices from
to-be-unmapped pages to fresh pages reserved by the planner, atomically
updates `req_to_token_pool.req_to_token` so the req keeps reading the
right data, and pulls dst pages out of `allocator.free_pages` so they
can't be re-handed-out.

Lives under `arena/` (not `budgeter/`) because it touches K/V buffers
and the allocator — same physical layer as the actuator. The scheduler
constructs one at startup and binds its `migrate` method as the
`migrate_callback` argument to `CrossPoolTransferActuator.execute(plan)`.

Runs synchronously, between forward steps. Cuda-graph rebuild is the
caller's concern (T6 already handles this around fires).
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from sglang.srt.arena.fire_plan import MigrateOp

logger = logging.getLogger(__name__)


class KVPageMigrator:
    """One-shot atomic migrator for KV pages.

    Construction:
      mig = KVPageMigrator(kv_pool, req_to_token_pool, allocator)

    Use as a callback into execute(plan):
      result = actuator.execute(plan, migrate_callback=mig.migrate)
    """

    def __init__(self, kv_pool, req_to_token_pool, allocator) -> None:
        self.kv_pool = kv_pool
        self.req_to_token_pool = req_to_token_pool
        self.allocator = allocator
        # Cache buffer refs once — re-resolving them every fire is needless
        # work since the underlying tensor view is stable across resizes
        # (the arena swaps the *backing storage*, not the Python object).
        self._k_buffers = kv_pool.k_buffer
        self._v_buffers = kv_pool.v_buffer
        self._n_layers = len(self._k_buffers)
        logger.info(
            "KVPageMigrator init: layers=%d k0_shape=%s allocator.size=%d",
            self._n_layers, tuple(self._k_buffers[0].shape), allocator.size,
        )

    # ------------------------------------------------------------------

    def migrate(self, ops: Sequence[MigrateOp]) -> int:
        """Carry out every migration in `ops`. Returns count of pages
        successfully migrated.

        Steps per op:
          1. D2D copy K[src] -> K[dst] and V[src] -> V[dst] across every layer.
          2. Update req_to_token[req_pool_idx, slot_in_req] = dst_page.
          3. After the loop, remove all dst_pages from allocator.free_pages
             (they're now owned by their respective reqs, not the free list).

        We sync once at the end so the GPU sees a consistent KV state
        before the next forward step. The executor caller already does
        a sync around the unmap step; this one is for inside-migrate
        ordering against the upcoming kernels.
        """
        if not ops:
            return 0

        # Vectorize the gather/scatter: dst_t[i] should receive src_t[i].
        # Using two index tensors lets torch issue a single advanced-indexing
        # copy per buffer instead of a Python loop over `len(ops)` ops.
        device = self._k_buffers[0].device
        src_t = torch.tensor(
            [op.src_page for op in ops], dtype=torch.int64, device=device
        )
        dst_t = torch.tensor(
            [op.dst_page for op in ops], dtype=torch.int64, device=device
        )

        # Step 1: D2D copy across all layers.
        for layer in range(self._n_layers):
            kbuf = self._k_buffers[layer]
            vbuf = self._v_buffers[layer]
            kbuf[dst_t] = kbuf[src_t]
            vbuf[dst_t] = vbuf[src_t]

        # Step 2: update req_to_token. Build CPU-side index tensors and
        # write in one shot. req_pool_idx and slot_in_req are CPU-side
        # ints from the planner; the writes target a CUDA tensor.
        rt = self.req_to_token_pool.req_to_token
        rt_device = rt.device
        req_idx = torch.tensor(
            [op.req_pool_idx for op in ops], dtype=torch.long, device=rt_device
        )
        slot_idx = torch.tensor(
            [op.slot_in_req for op in ops], dtype=torch.long, device=rt_device
        )
        # advanced indexing assignment — safe because (req_idx, slot_idx)
        # tuples are unique by construction (planner only emits one
        # MigrateOp per src_page, and one src_page maps to exactly one
        # (req, slot) per the OwnerMap walk).
        rt[req_idx, slot_idx] = dst_t.to(dtype=rt.dtype, device=rt_device)

        # Step 3: claim dst pages from allocator.free_pages so subsequent
        # alloc() can't hand them out. We hold the scheduler lock here, so
        # there's no race against a concurrent alloc.
        self._claim_dst_pages(dst_t)

        # Single sync so the migrate is visible to the upcoming
        # shrink_explicit + grow + post-fire kernels.
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass

        logger.info(
            "KVPageMigrator.migrate: ops=%d layers=%d (D2D + req_to_token + claim)",
            len(ops), self._n_layers,
        )
        return len(ops)

    # ------------------------------------------------------------------

    def _claim_dst_pages(self, dst_t: torch.Tensor) -> None:
        """Remove `dst_t` from allocator.free_pages and release_pages.

        These pages were chosen by the planner FROM the free-pages set,
        so they should be in there. Fail loudly if any aren't — that's
        a planner bug we want surfaced, not silently degraded into a
        double-handed-out page later.
        """
        alloc = self.allocator
        free_t = getattr(alloc, "free_pages", None)
        rel_t = getattr(alloc, "release_pages", None)

        # Build a lookup set on the dst pages for fast isin.
        # (dst_t is already on the allocator's device.)
        n_to_claim = int(dst_t.numel())
        n_claimed = 0

        if free_t is not None and free_t.numel() > 0:
            mask = torch.isin(free_t, dst_t)
            n_claimed += int(mask.sum().item())
            alloc.free_pages = free_t[~mask]
        if rel_t is not None and rel_t.numel() > 0:
            mask = torch.isin(rel_t, dst_t)
            n_claimed += int(mask.sum().item())
            alloc.release_pages = rel_t[~mask]

        if n_claimed != n_to_claim:
            raise RuntimeError(
                f"KVPageMigrator: planner claimed {n_to_claim} dst pages but "
                f"only {n_claimed} were actually in allocator.free_pages∪release_pages. "
                f"Planner reserved a page that wasn't free. Refusing to "
                f"continue — would result in a double-handed-out page."
            )
