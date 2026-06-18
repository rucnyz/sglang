"""Boot-time probes for `c^xfer` (cross-pool transfer wall) and `κ_i`
(recompute-curve coefficients) — design.md §"Boot-time probe".

Companion to `migrate_probe.py` (`c_m`). At engine start, before
serving begins, these pin the HW/model-dependent cost constants the
Admitter / fire-planner price against, so steady-state predictions are
deterministic:

  * `c^xfer` per chunk — wall of one cuMemUnmap+cuMemMap roundtrip on
    the actuator. These are SYNCHRONOUS host driver calls (not stream
    kernels), so wall-clock `perf_counter` timing is correct — no CUDA
    events / stream needed (contrast `migrate_probe`, whose copy IS a
    stream kernel). Seeds `RuntimeActuatorCost` (the runtime EWMA then
    drift-detects on top).

  * `κ_i` recompute coefficients — re-prefill wall at varying sequence
    lengths `L`, fitted to `c_KV = κ_α·L² + κ_β·L + κ_γ` (quadratic) and
    `c_M = κ_α·L + κ_β` (linear) by `cost_model.fit_cost_curves`. The
    GPU forward is injected as a `time_prefill_ms` callable so the
    sample-collection + fit are unit-testable without a model.

Both run once, at the moment the actuator chain is built
(`BudgetAgent._ensure_actuator_chain`), and fail-closed: a probe
exception leaves the conservative cold-start constant in place
(`c^xfer` default, `c_i` builtin/env curves) and warns once.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


def balance_restore(
    baseline_capacity: int,
    fire: Callable[[str, int], int],
    capacity: Callable[[], int],
    *,
    step_pages: int,
    max_fires: int = 16,
) -> int:
    """Drive a pool's capacity back to `baseline_capacity` after the
    c^xfer probe's self-reversing fires perturbed it. Returns the
    residual `capacity() - baseline_capacity` (0 == fully restored).

    `fire(direction, n_pages) -> moved_pages` executes one real fire;
    `capacity() -> int` reads the (kv) pool's current capacity. We close
    the loop on the OBSERVED capacity rather than pairing fires
    one-for-one, because the `kv_to_mamba` and `mamba_to_kv` directions
    round to `step_pages` (`lcm_pages`) with DIFFERENT per-subpool
    geometry — their per-fire capacity deltas need not be commensurable,
    so exact baseline is not guaranteed on every geometry. Fire whichever
    direction shrinks |capacity − baseline|; stop at baseline, on no
    progress, or after `max_fires` (bounded; the caller warns on a
    non-zero residual and the budgeter's steady-state rebalancing
    absorbs it).
    """
    for _ in range(max_fires):
        cap = capacity()
        if cap == baseline_capacity:
            return 0
        direction = (
            "mamba_to_kv" if cap < baseline_capacity else "kv_to_mamba"
        )
        moved = fire(direction, step_pages)
        if moved == 0 or capacity() == cap:
            break  # no progress — avoid spinning
    return capacity() - baseline_capacity


def measure_recompute_curves(
    time_prefill_ms: Callable[[int, str], float],
    *,
    kv_lengths: List[int],
    m_lengths: List[int],
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Collect `κ_i` prefill-probe samples: re-prefill each `L` through
    the KV / mamba stack and record `(L, wall_ms)`.

    `time_prefill_ms(L, pool)` is injected by `BudgetAgent` (wired to
    the model worker's synthetic forward) — isolating the GPU forward
    here keeps sample-collection + the downstream `fit_cost_curves` fit
    unit-testable with a synthetic timer. Returns
    `(kv_samples, m_samples)` ready for `cost_model.fit_cost_curves`.

    `kv_lengths` needs ≥3 distinct values (quadratic fit), `m_lengths`
    ≥2 (linear fit); the fitter enforces this and raises otherwise.
    """
    kv_samples = [(int(L), float(time_prefill_ms(L, "kv"))) for L in kv_lengths]
    m_samples = [(int(L), float(time_prefill_ms(L, "mamba"))) for L in m_lengths]
    logger.info(
        "RecomputeProbe: collected %d KV + %d mamba prefill samples",
        len(kv_samples), len(m_samples),
    )
    return kv_samples, m_samples
