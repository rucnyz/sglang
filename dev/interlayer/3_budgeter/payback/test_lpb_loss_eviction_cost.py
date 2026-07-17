"""Reproducing test: the cross-pool eviction-cost signal must be the LPB LOSS
(reuse-weighted recompute cost), NOT the raw evicted-token/slot COUNT.

Swarm k2m/m2k oscillation root (empirical, budgeter.jsonl): every tick the
PaybackPlanner logged `R_kv=50157[evict=50157+admit=0] R_m=0` and fired m2k
(drain mamba). KV was shedding a large COUNT of LOW-REUSE cache (swarm
hit=0.15), which the raw-count signal priced at a full re-prefill EACH. The
Admitter meanwhile grew mamba on demand (k2m) to serve requests -> the two
fought -> mamba never stayed grown -> no throughput win.

The LPB loss of a block is `n_b * c_recompute(s_b)`: a never-reused block
(n_b = hits_in_window() = 0) costs ~0 to evict. With the loss signal,
r_evict_kv ~ 0 for the swarm's low-reuse KV churn, so the planner stops firing
and the Admitter's mamba growth sticks. Accurate cost => no oscillation.
"""
from sglang.srt.budgeter.xpool_planner import PaybackConfig, PaybackPlanner


def _snap(kv_loss=0.0, m_loss=0.0, **kw):
    """Snapshot with the LPB-LOSS eviction signal (microseconds, reuse-weighted),
    the replacement for the raw kv_evicted_tokens_recent / mamba_evicted_slots_recent
    counts."""
    s = {"kv_evicted_lpb_loss_recent": kv_loss,
         "mamba_evicted_lpb_loss_recent": m_loss}
    s.update(kw)
    return s


def _planner():
    # cost_curves unused now: the planner consumes the pre-computed LPB loss
    # (already the reuse-weighted recompute cost), not a raw count it must price.
    return PaybackPlanner(cost_curves=None,
                          config=PaybackConfig(cooldown_s=1.0),
                          fire_cost_us=5000.0)


def test_low_reuse_kv_eviction_does_not_drive_m2k():
    """The oscillation case: a large COUNT of LOW-REUSE KV evictions carries
    ~0 LPB loss, so it must NOT fire m2k. (The old raw-count signal fired every
    tick, draining the mamba the Admitter was growing.)"""
    p = _planner()
    d = None
    for i in range(6):  # warm the EWMA across several ticks of low-reuse churn
        d = p.decide(_snap(kv_loss=0.0, m_loss=0.0, num_running_reqs=200,
                           num_queue_reqs=0), clock_s=10.0 * i, dt=1.0)
    assert d.direction is None, f"low-reuse KV eviction must not fire: {d.reason}"


def test_high_reuse_kv_eviction_still_fires_m2k():
    """The signal is reuse-weighted, not dead: a pool shedding HIGH-REUSE cache
    (real recompute loss) still fires toward it."""
    p = _planner()
    d = p.decide(_snap(kv_loss=5_000_000.0, m_loss=0.0), clock_s=100.0, dt=1.0)
    assert d.direction == "mamba_to_kv", f"real KV loss must fire m2k: {d.reason}"


def test_mamba_reuse_loss_fires_k2m():
    """Symmetric: mamba shedding REUSED snapshots (e.g. the swarm's shared COW
    prefix) carries real loss and fires k2m (grow mamba) — the swarm's wanted
    direction."""
    p = _planner()
    d = p.decide(_snap(kv_loss=0.0, m_loss=5_000_000.0), clock_s=100.0, dt=1.0)
    assert d.direction == "kv_to_mamba", f"real mamba loss must fire k2m: {d.reason}"
