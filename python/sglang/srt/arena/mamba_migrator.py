"""T8 step 6.4 — Mamba slot migrator.

Same role as `KVPageMigrator`, but for the mamba pool. Mamba slots
hold per-sequence recurrent state (conv + temporal + speculative-decode
intermediates), one slot per active req. Migrating a slot copies the
full per-sequence state via `MambaPool.migrate_slot` (T4) and then
updates the owning req's `mamba_pool_idx` so the next forward step
addresses the new slot.

Note: mamba uses `MigrateOp.slot_in_req == 0` because each req owns at
most one mamba slot, stored as `req.mamba_pool_idx` (a 1-element tensor).
Planner emits `slot_in_req=0` for mamba ops; we ignore the field at
write time but keep the contract uniform.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch

from sglang.srt.arena.fire_plan import MigrateOp

logger = logging.getLogger(__name__)


class MambaPageMigrator:
    """One-shot atomic mamba-slot migrator.

    Construction:
      mig = MambaPageMigrator(mamba_pool, scheduler)

    Use as a callback into execute(plan):
      result = actuator.execute(plan, migrate_callback=mig.migrate)

    Needs the scheduler (not just req_to_token_pool) because mamba
    routing is via `req.mamba_pool_idx` — a tensor stored on the Req,
    not in a central pool table. The migrator walks running_batch +
    waiting_queue to find the req owning each src slot.
    """

    def __init__(self, mamba_pool, scheduler) -> None:
        self.mamba_pool = mamba_pool
        self.scheduler = scheduler
        logger.info(
            "MambaPageMigrator init: mamba_pool.size=%d", mamba_pool.size
        )

    # ------------------------------------------------------------------

    def migrate(self, ops: Sequence[MigrateOp]) -> int:
        if not ops:
            return 0
        # Build a quick map of src_slot → req for fast lookup.
        src_to_req: dict[int, object] = {}
        sched = self.scheduler
        candidates = list(getattr(sched.running_batch, "reqs", []) or [])
        for r in getattr(sched, "waiting_queue", []) or []:
            if getattr(r, "mamba_pool_idx", None) is not None:
                candidates.append(r)
        for r in candidates:
            mp = getattr(r, "mamba_pool_idx", None)
            if mp is None:
                continue
            try:
                src_slot = int(mp[0]) if hasattr(mp, "__getitem__") else int(mp)
            except (TypeError, IndexError):
                continue
            src_to_req[src_slot] = r

        n_done = 0
        for op in ops:
            req = src_to_req.get(int(op.src_page))
            if req is None:
                raise RuntimeError(
                    f"MambaPageMigrator: no req owns src_slot={op.src_page} — "
                    f"OwnerMap walker disagrees with current scheduler state. "
                    f"Refusing to continue."
                )
            ok = self.mamba_pool.migrate_slot(int(op.src_page), int(op.dst_page))
            if not ok:
                raise RuntimeError(
                    f"MambaPageMigrator: migrate_slot({op.src_page}→{op.dst_page}) "
                    f"returned False. dst slot may not be free, or src==dst. "
                    f"Refusing to continue (would leave req routed to a stale slot)."
                )
            # Update the req's slot pointer. mamba_pool_idx is typically a
            # 1-element CPU tensor; do an in-place write so any captured
            # references (e.g. graph baked-in tensor address) see the new id.
            mp = req.mamba_pool_idx
            if hasattr(mp, "__setitem__"):
                mp[0] = int(op.dst_page)
            else:
                req.mamba_pool_idx = torch.tensor([int(op.dst_page)], dtype=torch.int32)
            n_done += 1

        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        logger.info("MambaPageMigrator.migrate: ops=%d", n_done)
        return n_done
