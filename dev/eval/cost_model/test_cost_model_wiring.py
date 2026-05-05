"""
Unit test: SGLANG_CSIGMA_* env → CostCurves → SGLangPressureAdapter pipeline.

Run:
    .venv/bin/python -m pytest dev/eval/cost_model/test_cost_model_wiring.py -v

Or as a script:
    .venv/bin/python dev/eval/cost_model/test_cost_model_wiring.py
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "python")
)


def _clear_env(env=None):
    """Strip every SGLANG_CSIGMA_* and SGLANG_XPOOL_* override from env."""
    target = env if env is not None else os.environ
    for k in list(target.keys()):
        if k.startswith("SGLANG_CSIGMA_") or k.startswith("SGLANG_XPOOL_"):
            del target[k]


class TestCostCurves(unittest.TestCase):
    def setUp(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        _clear_env()

    def tearDown(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        _clear_env()

    def test_legacy_default_used_when_no_env(self):
        from sglang.srt.budgeter.cost_model import get_cost_curves, LEGACY_DEFAULT
        c = get_cost_curves()
        self.assertEqual(c.source, "legacy_default")
        self.assertAlmostEqual(c.kv_alpha, LEGACY_DEFAULT.kv_alpha)
        self.assertAlmostEqual(c.m_beta, LEGACY_DEFAULT.m_beta)

    def test_env_loading(self):
        os.environ["SGLANG_CSIGMA_KV_ALPHA"] = "1.5e-7"
        os.environ["SGLANG_CSIGMA_KV_BETA"] = "0"
        os.environ["SGLANG_CSIGMA_KV_GAMMA"] = "0.5"
        os.environ["SGLANG_CSIGMA_M_ALPHA"] = "3e-3"
        os.environ["SGLANG_CSIGMA_M_BETA"] = "8.0"
        os.environ["SGLANG_CSIGMA_LSTAR"] = "20000"

        from sglang.srt.budgeter.cost_model import get_cost_curves
        c = get_cost_curves()
        self.assertEqual(c.source, "env")
        self.assertAlmostEqual(c.kv_alpha, 1.5e-7)
        self.assertAlmostEqual(c.kv_gamma, 0.5)
        self.assertAlmostEqual(c.m_alpha, 3e-3)
        self.assertAlmostEqual(c.m_beta, 8.0)
        self.assertAlmostEqual(c.L_star, 20000)

    def test_json_loading(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({
                "fit": {
                    "c_kv": {
                        "alpha_ms_per_tok2": 2e-7,
                        "beta_ms_per_tok": 0.0,
                        "gamma_ms": 0.6,
                    },
                    "c_m": {
                        "alpha_ms_per_tok": 4e-3,
                        "beta_ms": 9.0,
                    },
                    "crossover_L_star": 18000,
                }
            }, f)
            path = f.name
        try:
            os.environ["SGLANG_CSIGMA_JSON"] = path
            from sglang.srt.budgeter.cost_model import get_cost_curves
            c = get_cost_curves()
            self.assertTrue(c.source.startswith("json:"))
            self.assertAlmostEqual(c.kv_alpha, 2e-7)
            self.assertAlmostEqual(c.m_beta, 9.0)
        finally:
            os.unlink(path)

    def test_curves_arithmetic_at_crossover(self):
        """c_KV(L*) ≈ c_M(L*) at the reported crossover. Tolerance reflects
        rounding in the legacy-default coefficients (≈5% of either value)."""
        from sglang.srt.budgeter.cost_model import get_cost_curves
        c = get_cost_curves()
        L = c.L_star
        kv = c.c_kv_ms(L)
        m = c.c_m_ms(L)
        self.assertLess(abs(kv - m) / max(kv, m), 0.10,
            f"c_KV({L:.0f})={kv:.2f} ms ≠ c_M({L:.0f})={m:.2f} ms (rel err > 10%)")

    def test_short_recovery_mamba_dominates(self):
        """Below crossover: c_M >> c_KV."""
        from sglang.srt.budgeter.cost_model import get_cost_curves
        c = get_cost_curves()
        for L in (128, 512, 2048):
            ratio = c.c_m_ms(L) / c.c_kv_ms(L)
            self.assertGreater(ratio, 5.0,
                f"At L={L}, expected c_M/c_KV > 5, got {ratio:.2f}")

    def test_long_recovery_kv_dominates(self):
        """Above crossover (using a deliberately-aggressive L): c_KV > c_M."""
        from sglang.srt.budgeter.cost_model import get_cost_curves
        c = get_cost_curves()
        L_long = max(2 * c.L_star, 50000)
        self.assertGreater(c.c_kv_ms(L_long), c.c_m_ms(L_long))


class TestPressureAdapterWiring(unittest.TestCase):
    def setUp(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        _clear_env()

    def tearDown(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        _clear_env()

    def test_adapter_default_pulls_from_curves(self):
        from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
        a = SGLangPressureAdapter()
        # legacy default curves give c_KV(2048)/2048 in us/tok ≈ 0.5e3 * 0.5 / 2048
        # Just sanity-check the value is in a plausible range (0.1–50 us/tok)
        self.assertGreater(a.prefill_save_us_per_token, 0.05)
        self.assertLess(a.prefill_save_us_per_token, 100.0)
        # Curves are exposed on the adapter for downstream consumers.
        self.assertIsNotNone(a.cost_curves)

    def test_explicit_env_overrides_curve_default(self):
        os.environ["SGLANG_XPOOL_PREFILL_SAVE_US_PER_TOKEN"] = "42.0"
        from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
        a = SGLangPressureAdapter()
        self.assertAlmostEqual(a.prefill_save_us_per_token, 42.0)

    def test_signals_evict_uses_total_cost_at_evicted_length(self):
        """When calibration is loaded, evict_us = c_KV(N_evicted_tokens) — the
        total wall-clock to re-prefill that batch as one chunked-prefill,
        not a per-token multiplier × N."""
        # Force calibrated path with explicit env vars
        os.environ["SGLANG_CSIGMA_KV_ALPHA"] = "1.19e-7"
        os.environ["SGLANG_CSIGMA_KV_GAMMA"] = "0.44"
        os.environ["SGLANG_CSIGMA_M_ALPHA"] = "2.17e-3"
        os.environ["SGLANG_CSIGMA_M_BETA"] = "6.99"
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
        a = SGLangPressureAdapter()
        s_small = a.signals_from_snapshot(
            {"num_evicted_tokens_recent": 100}, 0, 0)
        s_large = a.signals_from_snapshot(
            {"num_evicted_tokens_recent": 8192}, 0, 0)
        # small batch: c_KV(100) ≈ γ ≈ 0.44 ms = 440 us (L² term negligible)
        self.assertGreater(s_small.evict_us, 100)
        self.assertLess(s_small.evict_us, 1000)
        # large batch: c_KV(8192) ≈ α·8192² + γ ≈ 8.0 ms = ~8000 us
        self.assertGreater(s_large.evict_us, 5000)
        self.assertLess(s_large.evict_us, 20000)

    def test_signals_retract_uses_observed_length(self):
        from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
        a = SGLangPressureAdapter()
        snap_short = {"num_retracted_reqs": 1, "mean_recovery_len_retract": 1024}
        snap_long = {"num_retracted_reqs": 1, "mean_recovery_len_retract": 32768}
        s_short = a.signals_from_snapshot(snap_short, 0, 0)
        s_long = a.signals_from_snapshot(snap_long, 0, 0)
        # 32K-token retract is far more expensive than 1K-token retract
        self.assertGreater(s_long.retract_us, 5 * s_short.retract_us)

    def test_signals_back_compat_no_recovery_len_field(self):
        """Without mean_recovery_len_*, signals fall back to scalar coefficients
        (legacy behavior preserved)."""
        from sglang.srt.budgeter.pressure_adapter import SGLangPressureAdapter
        a = SGLangPressureAdapter(prefill_save_us_per_token=10.0,
                                  full_prefill_us=50000.0)
        s = a.signals_from_snapshot(
            {"num_evicted_tokens_recent": 100, "num_retracted_reqs": 2}, 0, 0)
        self.assertAlmostEqual(s.evict_us, 1000.0)     # 100 * 10
        self.assertAlmostEqual(s.retract_us, 100000.0)  # 2 * 50000


class TestPlannerCostLogging(unittest.TestCase):
    def setUp(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        _clear_env()

    def tearDown(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        _clear_env()

    def test_jsonl_log_emits_one_record_per_decide(self):
        """SGLANG_XPOOL_COST_LOG-driven JSONL captures fire & no-fire ticks
        with c_σ regime info; per-tick breakdown plumbing varies by gate
        mode (legacy: pressure-adapter breakdown attached; NB-direction-
        aware: direction reasoning embedded in `reason`)."""
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            log_path = f.name
        try:
            os.environ["SGLANG_XPOOL_COST_LOG"] = log_path
            from sglang.srt.budgeter.cross_pool_planner import CrossPoolPlanner
            planner = CrossPoolPlanner()
            try:
                planner.decide(
                    usage_kv=0.95, usage_mamba=0.30,
                    queue_depth=4,
                    snapshot={
                        "num_evicted_tokens_recent": 1500,
                        "num_retracted_reqs": 0,
                        "num_paused_reqs": 0,
                        "num_queue_reqs": 4,
                        "mean_recovery_len_kv": 4096,
                        "mean_recovery_len_retract": 4096,
                    },
                    edge_active=False,
                )
                planner.decide(
                    usage_kv=0.40, usage_mamba=0.30,
                    queue_depth=0,
                    snapshot={"mean_recovery_len_kv": 1024},
                    edge_active=False,
                )
            finally:
                planner.close()
            with open(log_path) as fp:
                lines = [json.loads(L) for L in fp if L.strip()]
            self.assertEqual(len(lines), 2)
            self.assertIn("c_kv_ms_at_L", lines[0])
            self.assertIn("c_m_ms_at_L", lines[0])
            self.assertIn("regime", lines[0])
            self.assertIn("reason", lines[0])
        finally:
            try: os.unlink(log_path)
            except OSError: pass

    def test_nb_direction_aware_picks_correct_side(self):
        """Paper Eq nb-direction-gate: at L < L*, mamba evictions are more
        expensive → if mamba is saturated and KV has slack, NB should
        favor k2m (grow mamba). At L > L*, the inequality inverts."""
        os.environ["SGLANG_XPOOL_NB_DIRECTION_AWARE"] = "1"
        # Force calibrated curves
        os.environ["SGLANG_CSIGMA_KV_ALPHA"] = "1.19e-7"
        os.environ["SGLANG_CSIGMA_KV_GAMMA"] = "0.44"
        os.environ["SGLANG_CSIGMA_M_ALPHA"] = "2.17e-3"
        os.environ["SGLANG_CSIGMA_M_BETA"] = "6.99"
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        from sglang.srt.budgeter.cross_pool_planner import CrossPoolPlanner
        # Case A: L < L*, mamba saturated, KV has slack → fire k2m
        p1 = CrossPoolPlanner()
        d1 = p1.decide(
            usage_kv=0.30, usage_mamba=0.95, queue_depth=0,
            snapshot={"mean_recovery_len_kv": 2048,
                      "mean_recovery_len_retract": 2048},
        )
        self.assertEqual(d1.direction, "kv_to_mamba",
            f"expected k2m at L=2048 (M-exp), got {d1.direction}: {d1.reason}")

        # Case B: L > L*, KV saturated, mamba has slack → fire m2k
        p2 = CrossPoolPlanner()
        d2 = p2.decide(
            usage_kv=0.95, usage_mamba=0.30, queue_depth=0,
            snapshot={"mean_recovery_len_kv": 30000,
                      "mean_recovery_len_retract": 30000},
        )
        self.assertEqual(d2.direction, "mamba_to_kv",
            f"expected m2k at L=30000 (KV-exp), got {d2.direction}: {d2.reason}")

        # Case C: both pools tight (both ABOVE_HIGH) → NB saturation guard
        # forbids shrinking either source. Cross-pool fire is suppressed and
        # each pool self-evicts under its own intra-pool LRU. This matches
        # the unified-pool view: when a pool is pinned at saturation, the
        # cross-pool transfer cannot help — its own LRU is the cheaper
        # eviction path.
        p3 = CrossPoolPlanner()
        d3 = p3.decide(
            usage_kv=0.95, usage_mamba=0.95, queue_depth=0,
            snapshot={"mean_recovery_len_kv": 30000,
                      "mean_recovery_len_retract": 30000},
        )
        self.assertIsNone(d3.direction,
            f"expected no fire when both pools pinned ABOVE_HIGH, "
            f"got {d3.direction}: {d3.reason}")

        # Case D: both pools at slack → no fire (no NB ≥ threshold)
        p4 = CrossPoolPlanner()
        d4 = p4.decide(
            usage_kv=0.20, usage_mamba=0.20, queue_depth=0,
            snapshot={"mean_recovery_len_kv": 2048,
                      "mean_recovery_len_retract": 2048},
        )
        self.assertIsNone(d4.direction,
            f"expected no fire when both at slack, got {d4.direction}")

        for p in (p1, p2, p3, p4):
            p.close()


class TestRuntimeActuatorEWMA(unittest.TestCase):
    def setUp(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_runtime_actuator_cost()
        cost_model.reset_cost_curves()
        _clear_env()

    def tearDown(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_runtime_actuator_cost()
        cost_model.reset_cost_curves()
        _clear_env()

    def test_initial_conservative_until_calibrated(self):
        from sglang.srt.budgeter.cost_model import get_runtime_actuator_cost
        rt = get_runtime_actuator_cost()
        self.assertEqual(rt.n_observations, 0)
        self.assertFalse(rt.is_calibrated)
        # Default initial is 10ms = 10000us
        self.assertAlmostEqual(rt.current_us, 10000.0)

    def test_first_obs_seeds_ewma(self):
        from sglang.srt.budgeter.cost_model import get_runtime_actuator_cost
        rt = get_runtime_actuator_cost()
        # First observation REPLACES the conservative initial directly
        # (no dilution), since the initial is meant only for cold-start.
        rt.update(total_us=4500.0, n_chunks=1)
        self.assertEqual(rt.n_observations, 1)
        self.assertAlmostEqual(rt.current_us, 4500.0)
        self.assertFalse(rt.is_calibrated)  # need 3+

    def test_subsequent_obs_ewma(self):
        from sglang.srt.budgeter.cost_model import get_runtime_actuator_cost
        rt = get_runtime_actuator_cost()
        rt.update(total_us=4500.0, n_chunks=1)
        rt.update(total_us=5500.0, n_chunks=1)
        # alpha=0.3: 0.3 * 5500 + 0.7 * 4500 = 4800
        self.assertAlmostEqual(rt.current_us, 4800.0, delta=1.0)
        rt.update(total_us=5500.0, n_chunks=1)
        # 0.3 * 5500 + 0.7 * 4800 = 5010
        self.assertAlmostEqual(rt.current_us, 5010.0, delta=1.0)
        self.assertTrue(rt.is_calibrated)

    def test_planner_uses_runtime_cost_after_calibration(self):
        """When EWMA has 3+ observations, the planner threshold uses the
        runtime estimate instead of the static config value."""
        os.environ["SGLANG_CSIGMA_KV_ALPHA"] = "1.19e-7"
        os.environ["SGLANG_CSIGMA_KV_GAMMA"] = "0.44"
        os.environ["SGLANG_CSIGMA_M_ALPHA"] = "2.17e-3"
        os.environ["SGLANG_CSIGMA_M_BETA"] = "6.99"
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        cost_model.reset_runtime_actuator_cost()
        rt = cost_model.get_runtime_actuator_cost()
        # Seed 3 cheap observations (200us/chunk) so the EWMA is calibrated
        # and well below the legacy 5000us static default.
        for _ in range(3):
            rt.update(total_us=200.0, n_chunks=1)
        self.assertTrue(rt.is_calibrated)
        from sglang.srt.budgeter.cross_pool_planner import CrossPoolPlanner
        p = CrossPoolPlanner()
        # Scenario: M-expensive (L=2K), mamba saturated, KV slack. With
        # static cost 5000us × α=1.5 = 7500us threshold this fires (per
        # earlier test); with runtime 200us × 1.5 = 300us threshold also
        # fires but the gate is using the lower threshold. Validate via
        # the reason string mentioning the lower threshold value.
        d = p.decide(
            usage_kv=0.20, usage_mamba=0.95, queue_depth=0,
            snapshot={"mean_recovery_len_kv": 2048,
                      "mean_recovery_len_retract": 2048},
        )
        self.assertEqual(d.direction, "kv_to_mamba")
        # Reason text should reflect the runtime threshold (200 × 1.5 = 300us)
        # rather than the legacy 7500us.
        self.assertIn("threshold=300us", d.reason)
        p.close()


class TestSatGuardActiveVsTotal(unittest.TestCase):
    """When mamba pool is total-saturated by snapshots but active slots
    are far below high-water, sat-guard should NOT block m2k."""

    def setUp(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_runtime_actuator_cost()
        cost_model.reset_cost_curves()
        _clear_env()

    def tearDown(self):
        from sglang.srt.budgeter import cost_model
        cost_model.reset_runtime_actuator_cost()
        cost_model.reset_cost_curves()
        _clear_env()

    def test_m2k_allowed_when_only_snapshots_saturate_mamba(self):
        os.environ["SGLANG_CSIGMA_KV_ALPHA"] = "1.19e-7"
        os.environ["SGLANG_CSIGMA_KV_GAMMA"] = "0.44"
        os.environ["SGLANG_CSIGMA_M_ALPHA"] = "2.17e-3"
        os.environ["SGLANG_CSIGMA_M_BETA"] = "6.99"
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        cost_model.reset_runtime_actuator_cost()
        # Seed runtime actuator to a cheap value so the gate threshold
        # is low (otherwise cold-start 10ms suppresses everything).
        rt = cost_model.get_runtime_actuator_cost()
        for _ in range(3):
            rt.update(total_us=200.0, n_chunks=1)
        from sglang.srt.budgeter.cross_pool_planner import CrossPoolPlanner
        p = CrossPoolPlanner()
        # KV saturated, mamba TOTAL=0.99 but ACTIVE=0.20 (e.g., 30
        # running reqs out of 384 slots = 8% active, 91% snapshots).
        d = p.decide(
            usage_kv=0.95, usage_mamba=0.99, queue_depth=0,
            snapshot={
                # L > L* (=21780) → KV-expensive regime: m2k makes sense
                # (evict cheap mamba snapshots to relieve KV).
                "mean_recovery_len_kv": 30000,
                "mean_recovery_len_retract": 30000,
                "usage_mamba_active": 0.20,
            },
        )
        self.assertEqual(d.direction, "mamba_to_kv",
            f"expected m2k at L>L* with mamba ACTIVE=0.20 (slack), got "
            f"{d.direction}: {d.reason[:200]}")
        p.close()

    def test_m2k_blocked_when_active_slots_truly_saturate(self):
        os.environ["SGLANG_CSIGMA_KV_ALPHA"] = "1.19e-7"
        os.environ["SGLANG_CSIGMA_KV_GAMMA"] = "0.44"
        os.environ["SGLANG_CSIGMA_M_ALPHA"] = "2.17e-3"
        os.environ["SGLANG_CSIGMA_M_BETA"] = "6.99"
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        cost_model.reset_runtime_actuator_cost()
        rt = cost_model.get_runtime_actuator_cost()
        for _ in range(3):
            rt.update(total_us=200.0, n_chunks=1)
        from sglang.srt.budgeter.cross_pool_planner import CrossPoolPlanner
        p = CrossPoolPlanner()
        # Mamba ACTIVE=0.95 (real admission saturation) → m2k must be
        # blocked even at L > L*. Both KV and m saturated.
        d = p.decide(
            usage_kv=0.95, usage_mamba=0.99, queue_depth=0,
            snapshot={
                "mean_recovery_len_kv": 30000,
                "mean_recovery_len_retract": 30000,
                "usage_mamba_active": 0.95,
            },
        )
        self.assertIsNone(d.direction,
            f"expected no fire when both ACTIVE-saturated, got {d.direction}")
        p.close()

    def test_back_compat_no_active_field_falls_back(self):
        """Snapshots without `usage_mamba_active` use total usage_mamba
        for the guard (preserves pre-split behavior)."""
        os.environ["SGLANG_CSIGMA_KV_ALPHA"] = "1.19e-7"
        os.environ["SGLANG_CSIGMA_KV_GAMMA"] = "0.44"
        os.environ["SGLANG_CSIGMA_M_ALPHA"] = "2.17e-3"
        os.environ["SGLANG_CSIGMA_M_BETA"] = "6.99"
        from sglang.srt.budgeter import cost_model
        cost_model.reset_cost_curves()
        cost_model.reset_runtime_actuator_cost()
        rt = cost_model.get_runtime_actuator_cost()
        for _ in range(3):
            rt.update(total_us=200.0, n_chunks=1)
        from sglang.srt.budgeter.cross_pool_planner import CrossPoolPlanner
        p = CrossPoolPlanner()
        d = p.decide(
            usage_kv=0.95, usage_mamba=0.95, queue_depth=0,
            snapshot={"mean_recovery_len_kv": 30000,
                      "mean_recovery_len_retract": 30000},
        )
        self.assertIsNone(d.direction)  # both saturated by total → no fire
        p.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
