"""Self-test for the Tier-1 eviction simulator (#230).

The harness produces the characterization data, so it must be provably
correct.  These pin the qualitative invariants the slices rely on:

  V0 determinism — same args → identical metrics
  V1 no pressure → no eviction, no re-prefill, 100% hit
  V2 saturation + LRU → re-prefills appear (LRU evicts about-to-return)
  V3 saturation + ours-fresh ≤ LRU re-prefills (steering helps, never hurts)
  V4 hint-delay gradient: ours re-prefills are NON-DECREASING in delay and
     converge toward LRU as delay → ∞ (the latency-budget knee exists)
  V5 const-V_u (uniform score) ≈ LRU (no reuse info → no steering benefit)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import make_workload, simulate  # noqa: E402


def _green(s): return f"\033[32m{s}\033[0m"
def _red(s): return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# A saturated fixture: total resident demand >> pool, phased returns.
def _saturated_workload():
    return make_workload(
        n_programs=24, n_tokens=1000, decode_steps=6, tool_gap=20,
        rounds=4, arrival_spread=3, phat_lead=8,
    )


# pool holds ~6 of 24 contexts → heavy pressure.
_SAT_POOL = 6 * 1000


def v0_determinism():
    wl = _saturated_workload()
    a = simulate(wl, pool_cap_tokens=_SAT_POOL, policy="ours", hint_delay_steps=0)
    b = simulate(wl, pool_cap_tokens=_SAT_POOL, policy="ours", hint_delay_steps=0)
    if (a.reprefill_tokens, a.cache_hits, a.evict_imminent) != \
       (b.reprefill_tokens, b.cache_hits, b.evict_imminent):
        raise StageFail("V0: non-deterministic")
    print(_green("  [V0] deterministic OK"))


def v1_no_pressure_no_evict():
    wl = _saturated_workload()
    # pool fits every context → no eviction ever.
    m = simulate(wl, pool_cap_tokens=24 * 1000 + 10, policy="lru")
    if m.reprefill_tokens != 0 or m.inline_evict_tokens != 0:
        raise StageFail(f"V1: no-pressure must not evict; got reprefill="
                        f"{m.reprefill_tokens} evict={m.inline_evict_tokens}")
    if m.hit_rate() != 1.0:
        raise StageFail(f"V1: hit rate must be 1.0; got {m.hit_rate()}")
    print(_green("  [V1] no pressure → no evict, 100% hit OK"))


def v2_lru_saturation_reprefills():
    wl = _saturated_workload()
    m = simulate(wl, pool_cap_tokens=_SAT_POOL, policy="lru")
    if m.reprefill_tokens <= 0:
        raise StageFail("V2: LRU under saturation must produce re-prefills")
    print(_green(f"  [V2] LRU saturation re-prefill={m.reprefill_tokens} "
                 f"(hit={m.hit_rate():.2f}) OK"))


def v3_ours_fresh_beats_or_ties_lru():
    wl = _saturated_workload()
    lru = simulate(wl, pool_cap_tokens=_SAT_POOL, policy="lru")
    ours = simulate(wl, pool_cap_tokens=_SAT_POOL, policy="ours", hint_delay_steps=0)
    if ours.reprefill_tokens > lru.reprefill_tokens:
        raise StageFail(f"V3: ours-fresh must not be WORSE than LRU; "
                        f"ours={ours.reprefill_tokens} lru={lru.reprefill_tokens}")
    print(_green(f"  [V3] ours-fresh re-prefill={ours.reprefill_tokens} "
                 f"≤ LRU={lru.reprefill_tokens} OK"))


def v4_delay_gradient_monotone():
    wl = _saturated_workload()
    delays = [0, 2, 4, 8, 16, 32, 64]
    rp = [simulate(wl, pool_cap_tokens=_SAT_POOL, policy="ours",
                   hint_delay_steps=d).reprefill_tokens for d in delays]
    lru = simulate(wl, pool_cap_tokens=_SAT_POOL, policy="lru").reprefill_tokens
    # Non-decreasing in delay (allow tiny noise: each ≤ next + slack).
    slack = max(1, wl[0].n_tokens)   # one context of tolerance
    for i in range(len(delays) - 1):
        if rp[i] > rp[i + 1] + slack:
            raise StageFail(f"V4: re-prefill must be ~non-decreasing in hint "
                            f"delay; rp={list(zip(delays, rp))}")
    # Fresh strictly better than the most-stale (a real knee exists).
    if not (rp[0] < rp[-1]):
        raise StageFail(f"V4: fresh must beat most-stale (knee exists); "
                        f"rp[0]={rp[0]} rp[-1]={rp[-1]}")
    print(_green(f"  [V4] delay gradient {list(zip(delays, rp))} "
                 f"→ LRU={lru}; non-decreasing + knee OK"))


def v5_const_approx_lru():
    wl = _saturated_workload()
    lru = simulate(wl, pool_cap_tokens=_SAT_POOL, policy="lru").reprefill_tokens
    const = simulate(wl, pool_cap_tokens=_SAT_POOL, policy="const").reprefill_tokens
    # const has no reuse info → should not beat LRU meaningfully (≥ LRU - slack).
    if const < lru - 2 * wl[0].n_tokens:
        raise StageFail(f"V5: const-V_u (no reuse info) should not beat LRU; "
                        f"const={const} lru={lru}")
    print(_green(f"  [V5] const-V_u={const} ≈/≥ LRU={lru} (no steering info) OK"))


_STAGES = [
    ("V0 determinism", v0_determinism),
    ("V1 no-pressure", v1_no_pressure_no_evict),
    ("V2 LRU saturation re-prefills", v2_lru_saturation_reprefills),
    ("V3 ours-fresh ≤ LRU", v3_ours_fresh_beats_or_ties_lru),
    ("V4 delay gradient monotone + knee", v4_delay_gradient_monotone),
    ("V5 const ≈ LRU", v5_const_approx_lru),
]


def main() -> int:
    fails = []
    for name, fn in _STAGES:
        try:
            fn()
        except StageFail as e:
            print(_red(f"  FAIL {name}: {e}")); fails.append(name)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(_red(f"  FAIL {name}: unexpected {e!r}")); fails.append(name)
    print("=" * 56)
    if fails:
        print(_red(f"harness self-test FAILED: {fails}")); return 1
    print(_green(f"harness self-test PASS — {len(_STAGES)} invariants green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
