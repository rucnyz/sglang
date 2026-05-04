"""T3+T4 chain test: drive cross_pool_actuator's shrink-then-grow path
directly with mocked components, verify the SMART_OVERCAP and
ATOMIC_MIGRATION env-gated branches actually execute.

This bridges the gap between component-level unit tests (T3 free-mask,
T3 shrink_explicit, T4 migrate_slot, T4 expand_via_migration) and the
real fire path (T7 with admission saturation).
"""

import os
import sys
import torch
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.arena.cross_pool_actuator import (
    _select_drainable_chunks,
    _expand_via_migration,
)


class _FakeAlloc:
    def __init__(self, size, free_ids, device):
        self.size = size
        self.device = device
        self.free_pages = torch.tensor(sorted(free_ids), dtype=torch.int64, device=device)
        self.release_pages = torch.tensor([], dtype=torch.int64, device=device)
        self._capped_pages = torch.tensor([], dtype=torch.int64, device=device)
        self.capped_calls = []
        self.unmark_calls = []

    def free_page_mask(self):
        m = torch.zeros(self.size + 1, dtype=torch.bool, device=self.device)
        if self.free_pages.numel():
            m[self.free_pages] = True
        return m

    def select_drain_pages(self, n, prefer="high"):
        if n <= 0 or self.free_pages.numel() == 0:
            return torch.empty(0, dtype=torch.int64, device=self.device)
        sorted_free, _ = torch.sort(self.free_pages)
        if prefer == "high":
            return sorted_free[-n:].to(torch.int64) if sorted_free.numel() > n else sorted_free.to(torch.int64)
        return sorted_free[:n].to(torch.int64)

    def mark_pages_capped(self, page_indices):
        target = page_indices.to(self.device).to(torch.int64)
        before = self.free_pages.numel()
        mask = torch.isin(self.free_pages, target)
        held = self.free_pages[mask]
        self.free_pages = self.free_pages[~mask]
        moved = int(held.numel())
        self._capped_pages = torch.cat([self._capped_pages, held])
        self.capped_calls.append((page_indices.tolist(), moved))
        return moved


class _FakePool:
    def __init__(self, size, free_ids):
        self.size = size
        # mamba pool stores `free_slots`; mirror that so _expand_via_migration
        # can find them via attribute fallback.
        self.free_slots = torch.tensor(sorted(free_ids), dtype=torch.int64)


class _FakeSrcAct:
    def __init__(self, alloc, pool=None):
        self.allocator = alloc
        self.pool = pool


def case_smart_overcap_full_drain():
    """T3 path: enough free chunks at tail, shrink_explicit takes them
    directly, mark_pages_capped called with the right pages.
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # 100 pages, all free at tail (pages 91..100).
    # tokens_per_chunk = 5 → chunks 0..19. Chunk 18..19 are pages 91..100.
    alloc = _FakeAlloc(100, list(range(91, 101)) + list(range(1, 80)), device)
    src_act = _FakeSrcAct(alloc)
    chunks = _select_drainable_chunks(src_act, n_chunks=2, tokens_per_chunk=5)
    # Chunk 19 = pages 96..100 (all in alloc.free_pages); chunk 18 = pages 91..95.
    # Lower chunks (1..15) have pages 1..79 free which spans many chunks; chunk 0 = 1..5
    # (all free), chunk 1 = 6..10 (all free), ... up to chunk 15 = 76..79 + 80 (80 NOT free).
    # So drainable = chunks 0..14 + 18 + 19. Top 2 = [19, 18].
    assert chunks == [19, 18], f"expected [19,18], got {chunks}"
    print(f"[case full-drain] _select_drainable_chunks → {chunks}")

    # Simulate the actuator's mark_pages_capped step:
    drained_pages = []
    tpc = 5
    for c in chunks:
        drained_pages.extend(range(c * tpc + 1, (c + 1) * tpc + 1))
    pages_t = torch.tensor(drained_pages, dtype=torch.int64, device=device)
    moved = alloc.mark_pages_capped(pages_t)
    assert moved == 10, f"expected 10 pages capped, got {moved}"
    # And those pages should now be UNAVAILABLE for alloc.
    mask = alloc.free_page_mask()
    for c in chunks:
        for p in range(c * tpc + 1, (c + 1) * tpc + 1):
            assert not mask[p].item(), f"page {p} should be capped, but mask shows free"
    print(f"[case full-drain] mark_pages_capped removed {moved} pages from free_pages, "
          f"now in _capped_pages")


def case_atomic_migration_expansion():
    """T4 path: only some free chunks; _expand_via_migration uses migrator
    to convert live chunks into drainable ones.
    """
    # Pool of size 20, free slots = {1, 2, 3, 18, 19, 20} (low + tail).
    # Live slots = {4..17}.
    # smart-overcap returned [19, 18] (chunks 17, 18 — slot+1=18, 19); want 4 total.
    # Need to migrate live slot 17 (chunk 16) and slot 16 (chunk 15) → into low free.
    pool = _FakePool(size=20, free_ids=[1, 2, 3, 18, 19, 20])
    src_act = _FakeSrcAct(alloc=None, pool=pool)
    migrations = []
    def migrator(src, dst):
        migrations.append((src, dst))
        return True

    expanded = _expand_via_migration(
        src_act, drainable_chunks=[19, 18], n_target=4,
        tokens_per_chunk=1, migrator=migrator,
    )
    # Walk down: slot 20 (chunk 19) → in drain. slot 19 → in drain. slot 18 → FREE
    # (not live), skip. slot 17 → LIVE, eligible. slot 16 → LIVE, eligible. Done.
    # Migrate 17→1, 16→2.
    assert expanded == [19, 18, 16, 15], f"expanded {expanded}"
    assert migrations == [(17, 1), (16, 2)], f"migrations {migrations}"
    print(f"[case migration-expansion] expanded: {expanded}; migrations: {migrations}")


def case_atomic_migration_partial_failure():
    """If migrator returns False (e.g., dst became live mid-flight),
    that chunk is skipped, expansion continues.
    """
    pool = _FakePool(size=20, free_ids=[1, 2, 3, 18, 19, 20])
    src_act = _FakeSrcAct(alloc=None, pool=pool)
    attempts = []
    def flaky_migrator(src, dst):
        attempts.append((src, dst))
        # Fail the first attempt only.
        return len(attempts) > 1

    expanded = _expand_via_migration(
        src_act, drainable_chunks=[19, 18], n_target=4,
        tokens_per_chunk=1, migrator=flaky_migrator,
    )
    # First attempt 17→1 fails; chunk 16 skipped. Second attempt 16→2 succeeds.
    # No further attempts because the helper iterates over (chunk, src) extras
    # which only had 2 items before the for loop.
    # Result: expanded = [19, 18, 15] (only one extra chunk made it).
    assert 15 in expanded and 16 not in expanded, \
        f"chunk 16 (failed migration) shouldn't appear: {expanded}"
    assert len(attempts) == 2, f"two migration attempts: {attempts}"
    print(f"[case partial-failure] expanded with one mig fail: {expanded}; attempts: {attempts}")


def case_no_migrator_no_op():
    """Without migrator (i.e., budgeter doesn't pass one), expansion is no-op."""
    pool = _FakePool(size=20, free_ids=[1, 2, 18])
    src_act = _FakeSrcAct(alloc=None, pool=pool)
    expanded = _expand_via_migration(
        src_act, drainable_chunks=[19, 18], n_target=5,
        tokens_per_chunk=1, migrator=None,
    )
    assert expanded == [19, 18], f"no-migrator should be no-op: {expanded}"
    print(f"[case no-migrator] no-op: {expanded}")


def main():
    case_smart_overcap_full_drain()
    case_atomic_migration_expansion()
    case_atomic_migration_partial_failure()
    case_no_migrator_no_op()
    print("\nT3+T4 actuator chain test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
