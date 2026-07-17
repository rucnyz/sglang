"""Tests for the PaybackPlanner (cross-pool fire decision).

Key invariants:
  1. No eviction → no fire (idle workload).
  2. Sustained eviction in one pool → fires in that direction until eviction drops.
  3. Direction reversal when eviction pattern shifts (case3).
  4. Cooldown gate: at most 1 fire per cooldown_s.
  5. Self-convergence: if fires reduce eviction rate below payback threshold, fires stop.
"""
import pytest
from sglang.srt.budgeter.xpool_planner import PaybackPlanner, PaybackConfig


def _snap(kv_evict=0, m_evict=0):
    """kv_evict / m_evict are the reuse-weighted LPB LOSS (us) shed per pool
    this tick — the accurate eviction-cost signal the planner consumes (r_evict
    is the EWMA'd loss rate directly, no per-token re-multiply)."""
    return {
        "kv_evicted_lpb_loss_recent": float(kv_evict),
        "mamba_evicted_lpb_loss_recent": float(m_evict),
    }


class TestPaybackIdle:
    """No eviction → no fire."""

    def test_no_eviction_no_fire(self):
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=10.0))
        for t in range(100):
            d = p.decide(_snap(0, 0), clock_s=float(t), dt=1.0)
            assert d.direction is None, f"tick {t}: unexpected fire {d.direction}"


class TestPaybackSteady:
    """Sustained KV eviction → m2k fires; sustained mamba eviction → k2m fires."""

    def test_kv_eviction_fires_m2k(self):
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0), fire_cost_us=100.0)
        fired = []
        for t in range(50):
            d = p.decide(_snap(kv_evict=500, m_evict=0), clock_s=float(t), dt=1.0)
            if d.direction:
                fired.append((t, d.direction))
        assert len(fired) > 0, "should fire at least once with high KV eviction"
        assert all(d == "mamba_to_kv" for _, d in fired), "all fires should be m2k"

    def test_mamba_eviction_fires_k2m(self):
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0), fire_cost_us=100.0)
        fired = []
        for t in range(50):
            d = p.decide(_snap(kv_evict=0, m_evict=500), clock_s=float(t), dt=1.0)
            if d.direction:
                fired.append((t, d.direction))
        assert len(fired) > 0, "should fire at least once with high mamba eviction"
        assert all(d == "kv_to_mamba" for _, d in fired), "all fires should be k2m"


class TestPaybackShift:
    """Direction reversal when eviction pattern shifts (case3)."""

    def test_kv_then_mamba(self):
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0), fire_cost_us=100.0)
        dirs = []
        for t in range(100):
            if t < 50:
                d = p.decide(_snap(kv_evict=500, m_evict=0), clock_s=float(t), dt=1.0)
            else:
                d = p.decide(_snap(kv_evict=0, m_evict=500), clock_s=float(t), dt=1.0)
            if d.direction:
                dirs.append(d.direction)
        m2k = [d for d in dirs if d == "mamba_to_kv"]
        k2m = [d for d in dirs if d == "kv_to_mamba"]
        assert len(m2k) > 0, "should have m2k fires in phase A"
        assert len(k2m) > 0, "should have k2m fires in phase B"


class TestPaybackCooldown:
    """At most 1 fire per cooldown_s."""

    def test_cooldown_gate(self):
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=10.0), fire_cost_us=1.0)
        fired = []
        for t in range(100):
            d = p.decide(_snap(kv_evict=1000), clock_s=float(t), dt=1.0)
            if d.direction:
                fired.append(t)
        if len(fired) >= 2:
            gaps = [fired[i+1] - fired[i] for i in range(len(fired)-1)]
            assert all(g >= 10 for g in gaps), f"fire gap < cooldown: gaps={gaps}"


class TestPaybackConvergence:
    """Eviction drops after fires → fires stop."""

    def test_convergence(self):
        """After eviction stops, EWMA decays and fires cease within a few tau."""
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0), fire_cost_us=5000.0)
        fired_early = 0
        fired_late = 0
        for t in range(200):
            evict = 8000 if t < 30 else 0  # LPB loss (us) > fire_cost 5000
            d = p.decide(_snap(kv_evict=evict), clock_s=float(t), dt=1.0)
            if d.direction:
                if t < 30:
                    fired_early += 1
                elif t >= 60:
                    fired_late += 1
        assert fired_early > 0, "should fire during eviction phase"
        assert fired_late == 0, "should NOT fire well after eviction stops (EWMA decayed)"


class TestPaybackMargin:
    """payback_margin scales the fire threshold: fire iff payback > cost x margin."""

    def test_margin_1_fires(self):
        p = PaybackPlanner(
            config=PaybackConfig(cooldown_s=1.0, payback_margin=1.0),
            fire_cost_us=100.0,
        )
        fired = [
            t for t in range(50)
            if p.decide(_snap(kv_evict=500), clock_s=float(t), dt=1.0).direction
        ]
        assert fired, "margin 1.0 must fire on a signal well above fire_cost"

    def test_high_margin_suppresses_fire(self):
        # Same signal, threshold raised to 100 * 1000 = 100k us -- far above
        # any payback this signal can reach; must never fire.
        p = PaybackPlanner(
            config=PaybackConfig(cooldown_s=1.0, payback_margin=1000.0),
            fire_cost_us=100.0,
        )
        for t in range(50):
            d = p.decide(_snap(kv_evict=500), clock_s=float(t), dt=1.0)
            assert d.direction is None, f"tick {t}: fired despite margin 1000"

    def test_margin_env_roundtrip(self):
        from sglang.srt.budgeter.xpool_planner import _config_from_env
        from sglang.srt.environ import envs

        assert PaybackConfig().payback_margin == 1.0
        with envs.SGLANG_XPOOL_PAYBACK_MARGIN.override(2.5):
            assert _config_from_env().payback_margin == 2.5

    def test_margin_must_be_positive(self):
        with pytest.raises(ValueError):
            PaybackConfig(payback_margin=0.0)
