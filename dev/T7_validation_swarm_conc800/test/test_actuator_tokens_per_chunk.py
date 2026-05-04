"""T7-discovered bug fix: KVArenaActuator + MambaArenaActuator now
expose tokens_per_chunk via property, matching what cross_pool_actuator's
_select_drainable_chunks helper expects.

Without this, the helper falls back to default tpc=1, computes chunk
indices in mamba-page space (1..size), passes them to shrink_explicit
which silently skips them (slot indices way > pool.n_slots), and the
fire is a no-op. T7 first run showed unmapped=0 / granted=0 on every
fire because of this.

This test confirms the property reads from the underlying arena.
"""

import sys
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")


class _FakeArena:
    def __init__(self, tpc):
        self.tokens_per_chunk = tpc


class _FakePoolKV:
    def __init__(self, tpc):
        self._kv_arena = _FakeArena(tpc)
        self.size = 1_263_070
        self.page_size = 1


class _FakePoolMamba:
    def __init__(self, tpc):
        self._mamba_temporal_arena = _FakeArena(tpc)
        self.size = 366
        self.page_size = 1


def main():
    from sglang.srt.arena.kv_actuator import KVArenaActuator
    from sglang.srt.arena.mamba_actuator import MambaArenaActuator

    # KV pool with tpc=2048 (KV at 2 MiB chunks).
    pool_kv = _FakePoolKV(tpc=2048)
    # Need to bypass the strict __init__ ('_kv_arena' assertion)
    kv = KVArenaActuator.__new__(KVArenaActuator)
    kv.pool = pool_kv
    kv.allocator = None
    kv.max_tokens = pool_kv.size
    kv.live_tokens = pool_kv.size
    assert kv.tokens_per_chunk == 2048, \
        f"KV expected 2048, got {kv.tokens_per_chunk}"
    print(f"[KV actuator] tokens_per_chunk = {kv.tokens_per_chunk} (expected 2048)")

    # Mamba pool with tpc=1 (mamba at 2 MiB pages, 1 slot per chunk).
    pool_m = _FakePoolMamba(tpc=1)
    m = MambaArenaActuator.__new__(MambaArenaActuator)
    m.pool = pool_m
    m.max_slots = pool_m.size
    m.live_slots = pool_m.size
    assert m.tokens_per_chunk == 1, \
        f"mamba expected 1, got {m.tokens_per_chunk}"
    print(f"[mamba actuator] tokens_per_chunk = {m.tokens_per_chunk} (expected 1)")

    # Edge: pool without _kv_arena attribute → MUST raise (no silent fallback).
    # Silent fallback to 1 was the ORIGINAL T7 bug — unmapped chunks got
    # picked at wrong granularity and shrink_explicit silently skipped.
    class _NoArena: pass
    kv2 = KVArenaActuator.__new__(KVArenaActuator)
    kv2.pool = _NoArena()
    kv2.allocator = None
    kv2.max_tokens = 0
    kv2.live_tokens = 0
    raised = False
    try:
        _ = kv2.tokens_per_chunk
    except RuntimeError as e:
        raised = True
        msg = str(e)
    assert raised, "missing _kv_arena MUST raise (no silent fallback)"
    assert "Don't fall back silently" in msg
    print(f"[KV actuator no-arena] correctly raised RuntimeError")

    print("\nT7 actuator tokens_per_chunk fix unit test PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
