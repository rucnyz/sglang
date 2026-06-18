"""P4.1 — symmetric (bidirectional) Admitter decision (Phase 4).

Today `Admitter.decide_for_req` is hardcoded `dst_pool="kv"`, `src_pool="mamba"`:
it only ever decides cross-fires that GROW KV by draining mamba (m2k). A hybrid
request needs room in BOTH pools — x_tokens of KV *and* one mamba state slot
(plus its caching fork). So admission can be blocked by EITHER pool, and the
burst that crashes #312 is the mamba-scarce one: the Admitter must be able to
GROW MAMBA from KV (k2m) on a mamba-pressured arrival. That direction does not
exist yet.

Symmetric contract this test pins:
  - mamba scarce, KV slack  -> grow mamba: dst_pool="mamba", src_pool="kv",
    action in cross_*.
  - KV scarce, mamba slack  -> grow KV (existing): dst_pool="kv",
    src_pool="mamba".
  - BOTH scarce             -> defer (a hybrid req needs room in both; the two
    grows are opposite directions and cannot both be satisfied by cross-
    transfer, so we fall back to sglang's normal defer/retract — never crash).
  - neither scarce          -> own_free (normal sglang path).

Test-first (bug-workflow): on current code the mamba-dst case returns own_free
(it only looks at KV) or lacks the dst_pool field -> RED. After the symmetric
reparameterization -> GREEN.
"""
import sys

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")
sys.path.insert(0, "/scratch/yuzhou/projects/sglang/dev/interlayer/2_admitter")

# Reuse the faithful scheduler/req stubs + cost-warmed admitter factory.
from test_scheduler_hook import StubReq, StubScheduler, _fresh_admitter  # noqa: E402


def _decide(*, kv_free, kv_evictable, mamba_free, mamba_evictable,
            n_input_tokens=128, queue_len=0):
    """Drive decide_for_req with cross-fire ENABLED (Phase 4) over a stub
    scheduler in the requested pressure regime. tokens_per_page=1024 mirrors
    the hook default."""
    adm = _fresh_admitter()
    sched = StubScheduler(
        kv_free=kv_free, kv_evictable=kv_evictable,
        mamba_free=mamba_free, mamba_evictable=mamba_evictable,
        queue_len=queue_len,
    )
    req = StubReq(n_input_tokens=n_input_tokens)
    return adm.decide_for_req(req, sched, tokens_per_page=1024)


def test_A_mamba_scarce_never_owns():
    """Crash-safety: a mamba-scarce arrival (mamba free=0, evictable=0) must
    NEVER be own_free/own_evict — that admits the req into a pool that can't
    fork its cache slot and crashes (#312). It must either GROW MAMBA (dst=
    mamba cross) or DEFER, regardless of KV slack.

    RED today: decide_for_req only prices dst=kv, so with KV slack it returns
    own_free and never notices mamba is the binding pool. GREEN: not own_*."""
    dec = _decide(kv_free=200000, kv_evictable=50000,
                  mamba_free=0, mamba_evictable=0, queue_len=0)
    assert dec is not None
    assert not dec.action.startswith("own_"), (
        f"mamba-scarce arrival must not own_* into a pool that can't fork "
        f"(#312); got {dec.action} dst={getattr(dec,'dst_pool','<missing>')}")


def test_A2_mamba_scarce_under_queue_pressure_grows_mamba_from_kv():
    """The #312 burst: mamba can't give a slot but KV has plenty AND a queue
    backlog makes deferring expensive -> the Admitter fires k2m, growing mamba
    from KV (dst=mamba, src=kv, cross_*). (Empty queue would correctly prefer
    the free defer; a burst is precisely a backlog.)"""
    dec = _decide(kv_free=200000, kv_evictable=50000,
                  mamba_free=0, mamba_evictable=0, queue_len=200)
    assert dec is not None
    assert getattr(dec, "dst_pool", None) == "mamba", (
        f"mamba-scarce burst must GROW MAMBA (dst=mamba); got "
        f"dst={getattr(dec, 'dst_pool', '<missing>')} action={dec.action}.")
    assert getattr(dec, "src_pool", None) == "kv"
    assert dec.action.startswith("cross_"), (
        f"mamba scarce + KV slack + queue pressure must cross-fire to grow "
        f"mamba; got {dec.action}")


def test_B_both_pools_scarce_defers():
    """Both pools full (no free, no evictable in either). A hybrid req needs
    room in BOTH; the two grows are opposite cross directions and can't both be
    served, so the decision must be defer — sglang's normal back-pressure, not
    a crash or a doomed fire."""
    dec = _decide(kv_free=0, kv_evictable=0, mamba_free=0, mamba_evictable=0,
                  n_input_tokens=4096, queue_len=5)
    assert dec is not None
    assert dec.action == "defer", (
        f"both pools scarce must defer (neither can donate to the other); "
        f"got {dec.action} (dst={getattr(dec,'dst_pool',None)})")


def test_C_kv_scarce_grows_kv_from_mamba_regression():
    """Existing direction must be preserved and now explicitly labeled: KV
    scarce (free+evictable can't cover x_tokens) but mamba has free slots ->
    grow KV from mamba (dst=kv, src=mamba)."""
    # x_tokens = 4096; KV can cover only a sliver, mamba has free pages, and a
    # queue backlog makes deferring costly so the KV grow is chosen.
    dec = _decide(kv_free=8, kv_evictable=0, mamba_free=100000,
                  mamba_evictable=0, n_input_tokens=4096, queue_len=200)
    assert dec is not None
    assert getattr(dec, "dst_pool", None) == "kv", (
        f"KV-scarce arrival grows KV (dst=kv); got "
        f"dst={getattr(dec,'dst_pool','<missing>')} action={dec.action}")
    assert getattr(dec, "src_pool", None) == "mamba"
    assert dec.action.startswith("cross_"), (
        f"KV scarce + mamba slack + queue pressure must cross-fire to grow KV; "
        f"got {dec.action}")


def test_D_neither_scarce_owns():
    """Both pools have ample room -> normal own_free, no cross-fire."""
    dec = _decide(kv_free=200000, kv_evictable=0, mamba_free=100000,
                  mamba_evictable=0, n_input_tokens=128)
    assert dec is not None
    assert dec.action == "own_free", f"expected own_free, got {dec.action}"


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
