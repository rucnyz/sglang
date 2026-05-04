"""T4 unit test: _expand_via_migration extends a drainable_chunks list
to size n_target by calling the migrator on partial-free chunks.

Pure-Python unit test with a fake src_act + counting migrator, no
CUDA / SGLang server.
"""

import sys
import torch
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

from sglang.srt.arena.cross_pool_actuator import _expand_via_migration


class _FakePool:
    def __init__(self, size, free_slot_ids):
        self.size = size
        self.free_slots = torch.tensor(sorted(free_slot_ids), dtype=torch.int64)


class _FakeSrcAct:
    def __init__(self, pool):
        self.pool = pool
        self.allocator = None


def main():
    # Pool of size 20. Free slots: {1,2,3,4,5, 18,19,20}.
    # Live slots: {6,7,...,17}.
    # smart_overcap already returned [19, 18] (top-2 free chunks via
    # _select_drainable_chunks; chunk c == slot c+1, so chunks
    # {17,18,19} = slots {18,19,20}; only 18 and 19 chosen here for the
    # test). We want 4 total.
    pool = _FakePool(size=20, free_slot_ids=[1, 2, 3, 4, 5, 18, 19, 20])
    src_act = _FakeSrcAct(pool)

    migrations_called = []

    def fake_migrator(src, dst):
        migrations_called.append((src, dst))
        return True  # always succeed

    drainable_initial = [19, 18]  # chunks 18, 19 (slot 19, 20)
    extended = _expand_via_migration(
        src_act,
        drainable_initial,
        n_target=4,
        tokens_per_chunk=1,
        migrator=fake_migrator,
    )

    # Expected: walk down from slot 20, skip in-drain (chunk 19, 18),
    # next is slot 18 (chunk 17), but slot 18 is FREE → skip. Hmm wait
    # let me re-check. Free slots are {1,2,3,4,5, 18,19,20}. So:
    #   slot 20 (chunk 19) — already in drain, skip
    #   slot 19 (chunk 18) — already in drain, skip
    #   slot 18 (chunk 17) — FREE, skip (not live)
    #   slot 17 (chunk 16) — live, eligible
    #   slot 16 (chunk 15) — live, eligible
    # We want 2 more (n_target=4, have 2). So extras = chunks 16, 15.
    # Migrate slot 17 → some free in low (1,2,3,4,5). Pop 1.
    # Migrate slot 16 → 2.
    # Final extended (sorted desc): [19, 18, 16, 15]
    assert extended == [19, 18, 16, 15], \
        f"expected [19,18,16,15], got {extended}"
    print(f"[expand 2→4] extended chunks: {extended}")
    assert migrations_called == [(17, 1), (16, 2)], \
        f"unexpected migrations: {migrations_called}"
    print(f"[migrations called] {migrations_called}")

    # Edge: tpc != 1 → no expansion (KV out of scope).
    pool2 = _FakePool(size=20, free_slot_ids=[1, 2, 3])
    extended2 = _expand_via_migration(
        _FakeSrcAct(pool2),
        [19, 18],
        n_target=4,
        tokens_per_chunk=2,  # KV-grain
        migrator=fake_migrator,
    )
    assert extended2 == [19, 18], \
        f"tpc>1 should be no-op: got {extended2}"
    print(f"[edge: tpc=2] no expansion (KV out of scope): {extended2}")

    # Edge: migrator None → no-op.
    extended3 = _expand_via_migration(
        src_act, [19, 18], n_target=5, tokens_per_chunk=1, migrator=None
    )
    assert extended3 == [19, 18]
    print(f"[edge: no migrator] {extended3}")

    # Edge: already at n_target → no-op.
    extended4 = _expand_via_migration(
        src_act, [19, 18, 16, 15], n_target=4, tokens_per_chunk=1,
        migrator=fake_migrator,
    )
    assert extended4 == [19, 18, 16, 15]
    print(f"[edge: already at target] {extended4}")

    # Edge: migrator fails (returns False) — chunks not added.
    pool5 = _FakePool(size=20, free_slot_ids=[1, 2, 3])
    src_act5 = _FakeSrcAct(pool5)
    failures = []
    def failing_migrator(src, dst):
        failures.append((src, dst))
        return False
    extended5 = _expand_via_migration(
        src_act5, [], n_target=2, tokens_per_chunk=1,
        migrator=failing_migrator,
    )
    assert extended5 == [], f"migration-fail should not add chunks: got {extended5}"
    print(f"[edge: migrator fails] tried {len(failures)} times, got {extended5}")

    print("\nT4 _expand_via_migration unit test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
