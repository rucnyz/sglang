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


class TestConvergenceBackoff:
    """When fires do NOT reduce the harm they target — harm is not capacity-
    elastic because the working set exceeds total memory (the 35B-swarm case) —
    the planner must back off instead of firing every tick forever for zero
    gain. When fires DO reduce harm (elastic), it keeps firing until converged.
    """

    def _run(self, evict_feed, ticks=200, cooldown=1.0, fire_cost=1000.0):
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=cooldown), fire_cost_us=fire_cost)
        fires = []
        state = {"fired": False}
        for t in range(ticks):
            m_evict = evict_feed(t, state)
            d = p.decide(_snap(kv_evict=0, m_evict=m_evict), clock_s=float(t), dt=1.0)
            state["fired"] = bool(d.direction)
            if d.direction:
                fires.append(t)
        return fires

    def test_unresponsive_harm_backs_off(self):
        # Harm stays high forever regardless of fires (inelastic working set).
        # Without backoff this fires ~once/cooldown = ~200×. Backoff caps the
        # cadence at cooldown * 2**cap = 16s, so far fewer fires over 200 ticks.
        fires = self._run(lambda t, s: 8000.0)
        assert 0 < len(fires) < 40, f"backoff should throttle inelastic harm: {len(fires)} fires"
        gaps = [fires[i + 1] - fires[i] for i in range(len(fires) - 1)]
        assert gaps and max(gaps) >= 8, f"backoff should widen the cooldown: gaps={gaps}"

    def test_responsive_harm_keeps_firing(self):
        # Elastic: each fire drops the harm (pool grew, eviction fell). The
        # planner should keep firing at the base cooldown until harm decays
        # below the fire threshold (converged) — no premature backoff.
        def feed(t, s):
            h = s.get("h", 40000.0)
            if s.get("fired"):
                h *= 0.4
            s["h"] = h
            return h
        fires = self._run(feed, ticks=80)
        assert len(fires) >= 5, f"elastic harm should keep firing: {len(fires)}"
        early_gaps = [fires[i + 1] - fires[i] for i in range(min(3, len(fires) - 1))]
        assert all(g <= 2 for g in early_gaps), f"effective fires stay at base cooldown: {early_gaps}"

    def test_backoff_beats_no_backoff_on_inelastic(self):
        # Direct contrast: the backoff arm must fire strictly fewer times than a
        # backoff-disabled arm (converge_backoff_cap=0 -> 2**0=1, no widening)
        # on the same inelastic feed.
        def run(cap):
            p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0, converge_backoff_cap=cap),
                               fire_cost_us=1000.0)
            return sum(
                1 for t in range(200)
                if p.decide(_snap(m_evict=8000.0), clock_s=float(t), dt=1.0).direction
            )
        assert run(4) < run(0), "backoff must throttle vs no-backoff on inelastic harm"

    def test_effective_fire_resets_backoff(self):
        # Back off on inelastic harm, then make it elastic: a fire that drops
        # harm must reset the streak so firing resumes at the base cadence.
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0), fire_cost_us=1000.0)
        # Phase 1: inelastic -> backoff engages.
        for t in range(60):
            p.decide(_snap(m_evict=8000.0), clock_s=float(t), dt=1.0)
        assert p._dir_ineffective.get("kv_to_mamba", 0) > 0, "should have accumulated ineffective streak"
        # Phase 2: harm drops sharply (elastic) -> next fire is effective -> reset.
        p.decide(_snap(m_evict=100.0), clock_s=200.0, dt=1.0)
        d = p.decide(_snap(m_evict=8000.0), clock_s=201.0, dt=1.0)
        # after a reset the direction can fire again promptly
        assert p._dir_ineffective.get("kv_to_mamba", 0) == 0 or d.direction is not None


class TestActiveUsageGuard:
    """Never grow a pool by shrinking one with strictly higher active usage.
    r_evict is a cache-reuse signal blind to live-work displacement; on 35B swarm
    the smaller mamba pool sheds more re-prefill cost (tombstoning the hot
    shared-prefix trunk) yet KV carries more live work — growing mamba there
    steals from the bottleneck. The guard blocks exactly that."""

    def _snap_active(self, kv_evict=0, m_evict=0, kv_active=0.0, m_active=0.0):
        return {
            "kv_evicted_lpb_loss_recent": float(kv_evict),
            "mamba_evicted_lpb_loss_recent": float(m_evict),
            "usage_kv_active": float(kv_active),
            "usage_mamba_active": float(m_active),
        }

    def test_blocks_growing_less_active_pool(self):
        # mamba sheds more loss (would fire k2m) but KV is the more live-active
        # pool -> guard blocks k2m (the 35B-swarm pathology).
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0), fire_cost_us=100.0)
        fired = [
            p.decide(self._snap_active(m_evict=5000, kv_active=0.56, m_active=0.50),
                     clock_s=float(t), dt=1.0).direction
            for t in range(50)
        ]
        assert not any(fired), f"guard must block k2m when KV is more active: {[d for d in fired if d]}"

    def test_allows_growing_more_active_pool(self):
        # mamba sheds more loss AND mamba is the more-active pool -> k2m allowed.
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0), fire_cost_us=100.0)
        fired = [
            t for t in range(50)
            if p.decide(self._snap_active(m_evict=5000, kv_active=0.40, m_active=0.60),
                        clock_s=float(t), dt=1.0).direction == "kv_to_mamba"
        ]
        assert fired, "guard must allow k2m when mamba is the more-active pool"

    def test_allows_m2k_toward_more_active_kv(self):
        # KV sheds more loss AND KV is more active -> m2k (grow KV) allowed (9B).
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0), fire_cost_us=100.0)
        fired = [
            t for t in range(50)
            if p.decide(self._snap_active(kv_evict=5000, kv_active=0.56, m_active=0.50),
                        clock_s=float(t), dt=1.0).direction == "mamba_to_kv"
        ]
        assert fired, "guard must allow m2k (grow the more-active KV pool)"

    def test_ablation_guard_off_restores_pathology(self):
        # with the guard disabled, k2m fires even when KV is more active.
        p = PaybackPlanner(config=PaybackConfig(cooldown_s=1.0, active_usage_guard=False),
                           fire_cost_us=100.0)
        fired = [
            t for t in range(50)
            if p.decide(self._snap_active(m_evict=5000, kv_active=0.56, m_active=0.50),
                        clock_s=float(t), dt=1.0).direction
        ]
        assert fired, "guard OFF must restore the unguarded k2m pathology"


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
