"""P4.1-B1 — the k2m grow fire must be sized by the mamba slot need, in mamba
CHUNKS, not by the req's KV input length or in KV pages (Phase 4 audit B1).

Two coupled unit bugs in the mamba-grow firing path:
  1. `_maybe_admitter_fire` passes `x_tokens = len(req.origin_input_ids)` (the
     req's KV input) to `execute_decision`, which sizes `n_pages` from it. For
     a mamba GROW that's the wrong quantity — the fire should add the mamba
     state the arrival needs (active + fork), not transfer the req's KV input
     worth of pages.
  2. Even sized by the mamba need, one transferred page is one arena CHUNK =
     `mamba_tokens_per_chunk` slots. Converting the slot need with the KV
     `tokens_per_page` (1024) instead of the mamba `tokens_per_chunk`
     over-grows mamba by `tokens_per_chunk×` on a fragmentable layout (tps>1).

Correct: grow `ceil(need_slots / mamba_tps)` chunks. With need=2 and mamba
tokens_per_chunk=2 that is exactly 1 chunk — not 2 (KV-token mis-conversion),
not 4 (sized by a 4096-token KV input).

Test-first: drive decide_for_req (mamba-scarce, under queue pressure → grow
mamba) then execute_decision, and assert the planner builds `kv_to_mamba` with
`n_pages = ceil(need/mamba_tps)`. RED today (sized by KV input → 4 pages).
"""
import math
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/dev/interlayer/2_admitter")

from sglang.srt.budgeter.admitter import Admitter  # noqa: E402
from sglang.srt.budgeter.cost_model import reset_cost_model, get_cost_model  # noqa: E402
from test_scheduler_hook import StubReq, StubAllocator, StubTreeCache, StubMambaPool  # noqa: E402
from test_sync_fire import FakePlanner, FakeActuator  # noqa: E402


class _MambaScarceSched:
    """mamba scarce (0 free/evictable), KV slack, queue backlog → the Admitter
    grows mamba from KV. Exposes a mamba tokens_per_chunk seam."""

    def __init__(self, *, mamba_tps):
        self.token_to_kv_pool_allocator = StubAllocator(available=200_000)
        self.tree_cache = StubTreeCache(evictable=50_000)
        self._mamba_pool = StubMambaPool(available=0)
        self._mamba_evictable = 0
        self._mamba_tps = mamba_tps
        self.waiting_queue = [None] * 200  # backlog → defer is costly
        self.disaggregation_mode = "NULL"

    def get_mamba_pool(self):
        return self._mamba_pool

    def get_mamba_evictable(self):
        return self._mamba_evictable

    def get_mamba_tokens_per_chunk(self):
        return self._mamba_tps


def _warm_admitter():
    reset_cost_model()
    cm = get_cost_model()
    for _ in range(5):
        cm.update_xfer(total_us=1000.0, n_chunks=10)
    adm = Admitter(cost_model=cm)
    planner = FakePlanner()
    # lcm=1 (single subpool each) so n_pages reflects the need, not LCM rounding.
    adm.actuator = FakeActuator(n_kv_subpools=1, n_mamba_subpools=1)
    adm.planner = planner
    adm.lcm_pages = 1
    return adm, planner


def _drive(mamba_tps, n_input_tokens=4096):
    adm, planner = _warm_admitter()
    sched = _MambaScarceSched(mamba_tps=mamba_tps)
    req = StubReq(n_input_tokens=n_input_tokens)
    dec = adm.decide_for_req(req, sched, tokens_per_page=1024)
    assert dec.dst_pool == "mamba", (
        f"precondition: mamba-scarce burst should grow mamba; got dst={dec.dst_pool} "
        f"action={dec.action}")
    # Mirror the scheduler hook: it passes the req's KV input length.
    adm.execute_decision(dec, x_tokens=n_input_tokens,
                         src_pool=dec.src_pool, dst_pool=dec.dst_pool,
                         tokens_per_page=1024)
    return dec, planner


def test_grow_sized_by_mamba_need_not_kv_input_atomic():
    """tokens_per_chunk=1 (atomic): need=2 slots → 2 chunks. The fire must be
    2 pages (the mamba need), NOT 4 (the 4096-token KV input)."""
    need_slots = 2
    dec, planner = _drive(mamba_tps=1, n_input_tokens=4096)
    assert len(planner.calls) == 1, f"expected one build; got {planner.calls}"
    direction, n_pages = planner.calls[0][0], planner.calls[0][1]
    assert direction == "kv_to_mamba", f"grow mamba must fire kv_to_mamba; got {direction}"
    assert n_pages == math.ceil(need_slots / 1), (
        f"grow fire must be sized by the mamba need ({need_slots} slots / tps 1 "
        f"= 2 chunks), not the 4096-token KV input (4 pages); got {n_pages}")


def test_grow_sized_in_mamba_chunks_fragmentable():
    """tokens_per_chunk=2 (fragmentable): need=2 slots = 1 chunk. The fire must
    be 1 page, NOT 2 (KV-token mis-conversion) and NOT 4 (KV input)."""
    need_slots = 2
    dec, planner = _drive(mamba_tps=2, n_input_tokens=4096)
    direction, n_pages = planner.calls[0][0], planner.calls[0][1]
    assert direction == "kv_to_mamba"
    assert n_pages == math.ceil(need_slots / 2), (
        f"grow fire must be ceil(need/tps)=1 chunk on a tps=2 layout; "
        f"got {n_pages} (2 = KV-token mis-conversion, 4 = sized by KV input)")


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError:
                failures += 1
                print("FAIL", name)
                traceback.print_exc()
            except Exception:
                failures += 1
                print("ERROR", name)
                traceback.print_exc()
    sys.exit(1 if failures else 0)
