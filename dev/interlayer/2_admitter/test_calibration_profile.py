"""#265 — unified calibration profile: c_m env-pin + boot-probe dump.

A per-(model, GPU) profile (dev/eval/cost_model/calibrate_profile.sh)
folds three constants into one source-able file: κ_i (offline, the
SGLANG_CSIGMA_* path, covered by test_boot_probe.py), plus c^xfer and
c_m measured by the engine boot probe and dumped to JSON. This file
pins the two NEW seams that make that profile possible:

1. c_m can be pinned offline via SGLANG_CM_MAMBA_PER_SLOT_US — it is a
   fixed-hardware constant (no drift), so env-precedence applies exactly
   like κ_i's curves. (set / negative-rejected / unset cold-start.)
2. The boot probe defers to the env-pinned c_m (does NOT re-measure).
3. `_dump_probe_results` writes c^xfer (the EWMA seed) + c_m to the
   SGLANG_BUDGETER_PROBE_DUMP JSON path; un-probed c_m (+inf) → null;
   no env → no-op.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, "/scratch/yuzhou/projects/sglang/python")

_CM_ENV = "SGLANG_CM_MAMBA_PER_SLOT_US"
_DUMP_ENV = "SGLANG_BUDGETER_PROBE_DUMP"


def _clear_env():
    for k in (_CM_ENV, _DUMP_ENV):
        os.environ.pop(k, None)


def _fresh_migrate_cost():
    from sglang.srt.budgeter.cost_model import (
        get_migrate_cost,
        reset_migrate_cost,
    )
    reset_migrate_cost()
    return get_migrate_cost()


def test_1_cm_pinned_from_env():
    _clear_env()
    os.environ[_CM_ENV] = "137.5"
    try:
        mc = _fresh_migrate_cost()
        assert mc.is_env_pinned is True
        assert mc.is_calibrated is True
        assert abs(mc.mamba_per_slot_us - 137.5) < 1e-9, mc.mamba_per_slot_us
    finally:
        _clear_env()
        _fresh_migrate_cost()
    print("  PASS  1  c_m pinned from SGLANG_CM_MAMBA_PER_SLOT_US=137.5µs")


def test_2_cm_env_rejects_nonpositive():
    _clear_env()
    os.environ[_CM_ENV] = "-1"
    try:
        raised = False
        try:
            _fresh_migrate_cost()
        except ValueError:
            raised = True
        assert raised, "negative SGLANG_CM_MAMBA_PER_SLOT_US must raise"
    finally:
        _clear_env()
        _fresh_migrate_cost()
    print("  PASS  2  c_m env rejects non-positive value")


def test_3_cm_unset_cold_starts_infeasible():
    _clear_env()
    mc = _fresh_migrate_cost()
    assert mc.is_env_pinned is False
    assert mc.is_calibrated is False
    assert mc.mamba_per_slot_us == float("inf")
    print("  PASS  3  c_m unset → +inf, not calibrated, not pinned")


def test_4_boot_probe_defers_to_pinned_cm():
    """_run_migrate_probe must NOT re-measure when c_m is env-pinned."""
    _clear_env()
    os.environ[_CM_ENV] = "200.0"
    try:
        _fresh_migrate_cost()
        from sglang.srt.budgeter import agent as agent_mod
        from sglang.srt.budgeter.agent import BudgetAgent

        called = {"n": 0}

        def _boom(_pool):
            called["n"] += 1
            return 999.0  # would clobber the pinned 200 if applied

        # _run_migrate_probe imports measure_mamba_migrate from
        # migrate_probe; patch it there.
        from sglang.srt.budgeter import migrate_probe
        orig = migrate_probe.measure_mamba_migrate
        migrate_probe.measure_mamba_migrate = _boom
        try:
            # Body touches only get_migrate_cost() + logger on the pinned
            # path, so an unbound call with a bare namespace is enough.
            import types
            BudgetAgent._run_migrate_probe(types.SimpleNamespace(), object())
        finally:
            migrate_probe.measure_mamba_migrate = orig

        from sglang.srt.budgeter.cost_model import get_migrate_cost
        mc = get_migrate_cost()
        assert called["n"] == 0, "probe must skip when env-pinned"
        assert abs(mc.mamba_per_slot_us - 200.0) < 1e-9, mc.mamba_per_slot_us
    finally:
        _clear_env()
        _fresh_migrate_cost()
    print("  PASS  4  boot probe defers to env-pinned c_m (no re-measure)")


def test_5_dump_writes_xfer_and_cm():
    _clear_env()
    from sglang.srt.budgeter.cost_model import (
        get_migrate_cost,
        get_runtime_actuator_cost,
        reset_migrate_cost,
        reset_runtime_actuator_cost,
    )
    from sglang.srt.budgeter.agent import BudgetAgent
    import types

    reset_runtime_actuator_cost()
    reset_migrate_cost()
    get_runtime_actuator_cost().seed_from_boot_probe(90.7)
    get_migrate_cost().set_mamba(113.0)

    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as f:
        path = f.name
    os.environ[_DUMP_ENV] = path
    try:
        BudgetAgent._dump_probe_results(types.SimpleNamespace())
        rec = json.load(open(path))
        assert abs(rec["c_xfer_us_per_page"] - 90.7) < 1e-6, rec
        assert abs(rec["c_m_us_per_slot"] - 113.0) < 1e-6, rec
        assert rec["c_m_calibrated"] is True
    finally:
        os.remove(path)
        _clear_env()
        reset_runtime_actuator_cost()
        reset_migrate_cost()
    print("  PASS  5  dump writes c^xfer=90.7 + c_m=113.0 to JSON")


def test_6_dump_nulls_unprobed_cm():
    _clear_env()
    from sglang.srt.budgeter.cost_model import (
        get_runtime_actuator_cost,
        reset_migrate_cost,
        reset_runtime_actuator_cost,
    )
    from sglang.srt.budgeter.agent import BudgetAgent
    import types

    reset_runtime_actuator_cost()
    reset_migrate_cost()  # c_m stays +inf
    get_runtime_actuator_cost().seed_from_boot_probe(91.0)

    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as f:
        path = f.name
    os.environ[_DUMP_ENV] = path
    try:
        BudgetAgent._dump_probe_results(types.SimpleNamespace())
        rec = json.load(open(path))
        assert rec["c_m_us_per_slot"] is None, rec
        assert rec["c_m_calibrated"] is False, rec
    finally:
        os.remove(path)
        _clear_env()
        reset_runtime_actuator_cost()
        reset_migrate_cost()
    print("  PASS  6  dump nulls un-probed c_m (+inf → null)")


def test_7_dump_noop_without_env():
    _clear_env()  # no SGLANG_BUDGETER_PROBE_DUMP
    from sglang.srt.budgeter.agent import BudgetAgent
    import types
    # Must not raise and must not write anything (nothing to assert but
    # the absence of an exception / file).
    BudgetAgent._dump_probe_results(types.SimpleNamespace())
    print("  PASS  7  dump is a no-op when SGLANG_BUDGETER_PROBE_DUMP unset")


def test_8_recompute_probe_noop_when_offline():
    """κ_i is calibrated OFFLINE (calibrate_profile.sh); the in-engine
    `_run_recompute_probe` is intentionally inert because
    `_make_prefill_timer` returns None. Pin that contract: with no timer
    it is a clean no-op that never installs curves (so the env/builtin
    κ_i is preserved) and never raises."""
    from sglang.srt.budgeter import cost_model
    from sglang.srt.budgeter.agent import BudgetAgent
    import types
    called = {"n": 0}
    orig = cost_model.set_cost_curves
    cost_model.set_cost_curves = lambda c: called.__setitem__("n", called["n"] + 1)
    try:
        ns = types.SimpleNamespace(
            _make_prefill_timer=lambda: None, _boot_probe_warned=False
        )
        BudgetAgent._run_recompute_probe(ns)  # must not raise, must not install
    finally:
        cost_model.set_cost_curves = orig
    assert called["n"] == 0, (
        "in-engine κ_i probe must be a no-op when timer=None (offline-only)"
    )
    print("  PASS  8  _run_recompute_probe clean no-op when timer=None "
          "(κ_i offline-only)")


def test_9_dump_omits_cxfer_when_not_seeded():
    """c^xfer is dumped only when the boot probe SEEDED a measured wall
    (is_boot_seeded). A fresh / failed-probe actuator holds the
    conservative default in current_us but is_boot_seeded=False, so the
    dump writes c_xfer_us_per_page=None → the profile omits the seed. This
    pins the not-seeded→None half; test_5 (seeded but is_calibrated still
    False → c^xfer STILL emitted) pins the half that discriminates the
    correct is_boot_seeded gate from the wrong is_calibrated one."""
    _clear_env()
    from sglang.srt.budgeter.cost_model import (
        get_runtime_actuator_cost,
        reset_migrate_cost,
        reset_runtime_actuator_cost,
    )
    from sglang.srt.budgeter.agent import BudgetAgent
    import types
    reset_runtime_actuator_cost()
    reset_migrate_cost()
    xfer = get_runtime_actuator_cost()  # fresh: default in current_us, not seeded
    assert xfer.is_boot_seeded is False
    assert xfer.current_us > 0  # the conservative default is present...
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as f:
        path = f.name
    os.environ[_DUMP_ENV] = path
    try:
        BudgetAgent._dump_probe_results(types.SimpleNamespace())
        rec = json.load(open(path))
        assert rec["c_xfer_us_per_page"] is None, (
            f"...but an un-seeded probe must dump c_xfer=None, got {rec}"
        )
    finally:
        os.remove(path)
        _clear_env()
        reset_runtime_actuator_cost()
        reset_migrate_cost()
    print("  PASS  9  dump writes c_xfer=None when probe didn't seed "
          "(is_boot_seeded gate, not is_calibrated)")


def test_10_warn_on_model_mismatch():
    """A stale calibration profile (SGLANG_CSIGMA_MODEL set for model A)
    deployed against running model B must WARN (no longer silent), once.
    Basename comparison so a HF repo id vs a local checkpoint path of the
    SAME model does NOT false-positive."""
    from sglang.srt.budgeter import cost_model

    warned = {"n": 0}
    orig = cost_model.logger.warning
    cost_model.logger.warning = lambda *a, **k: warned.__setitem__("n", warned["n"] + 1)
    try:
        # (a) genuine mismatch → warns exactly once
        os.environ["SGLANG_CSIGMA_MODEL"] = "Qwen/Qwen3.5-9B"
        cost_model._model_mismatch_warned = False
        cost_model.warn_on_model_mismatch("meta-llama/Llama-3-8B")
        assert warned["n"] == 1, "genuine model mismatch must warn"
        cost_model.warn_on_model_mismatch("meta-llama/Llama-3-8B")
        assert warned["n"] == 1, "warn-once: the second call must be silent"

        # (b) basename match (HF id vs local path) → no warn
        warned["n"] = 0
        cost_model._model_mismatch_warned = False
        cost_model.warn_on_model_mismatch("/data/models/Qwen3.5-9B")
        assert warned["n"] == 0, "same model via local path must NOT warn"

        # (c) no profile env → no warn (nothing to compare)
        warned["n"] = 0
        cost_model._model_mismatch_warned = False
        os.environ.pop("SGLANG_CSIGMA_MODEL", None)
        cost_model.warn_on_model_mismatch("anything")
        assert warned["n"] == 0, "no SGLANG_CSIGMA_MODEL → no warn"
    finally:
        cost_model.logger.warning = orig
        os.environ.pop("SGLANG_CSIGMA_MODEL", None)
        cost_model._model_mismatch_warned = False
    print("  PASS  10 warn_on_model_mismatch: mismatch warns once; "
          "basename match (HF id vs local path) + no-profile stay silent")


if __name__ == "__main__":
    test_1_cm_pinned_from_env()
    test_2_cm_env_rejects_nonpositive()
    test_3_cm_unset_cold_starts_infeasible()
    test_4_boot_probe_defers_to_pinned_cm()
    test_5_dump_writes_xfer_and_cm()
    test_6_dump_nulls_unprobed_cm()
    test_7_dump_noop_without_env()
    test_8_recompute_probe_noop_when_offline()
    test_9_dump_omits_cxfer_when_not_seeded()
    test_10_warn_on_model_mismatch()
    print("\nAll calibration-profile tests passed.")
