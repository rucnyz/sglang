"""T16 verify — `re_use` no-double-count (DESIGN §8, round-9 part 1 B1).

The capacity_fits check (DESIGN §8) determines whether resuming a
paused program would overflow HBM.  Its first term forecasts inflight
HBM demand; the second adds ``re_use[sp]`` — bytes that would re-enter
HBM upon resume.  **Round-9 B1 fix**: ``re_use[sp]`` must NOT include
bytes already resident in HBM (kept alive by other holders), or the
two terms double-count and capacity_fits over-pessimises (the paused
program gets stuck unable to resume because the "needed" bytes are
actually already there).

The probe targets ``expected_peak_hbm_after_resume(program_unit_hashes,
units)`` — a pure function exposed by ``daemon/_admission_math.py``.
T34 (#156, multi-axis DP) wires this into the candidate generator
later; T16 lands the function and its property tests up front so
T34 doesn't have to re-test.

Property under test (Round-9 B1): for any unit ``u`` whose
``residence`` contains ``Tier.HBM``, ``re_use[*]`` MUST be 0 for that
unit — regardless of how many holders the unit has or what state
they're in.

Usage:
    python dev/aginfer/verify/t16/verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from baselines.base import ReuseUnit, Scope, Tier, UnitType  # noqa: E402
from daemon._admission_math import expected_peak_hbm_after_resume  # noqa: E402


def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


# ----------------------------------------------------------------- helpers


def _unit(
    uid: str,
    *,
    residence: List[Tier],
    n_bytes_by_tier: Dict[Tier, Dict[str, int]],
    holders: List[str],
    n_tokens: int = 256,
) -> ReuseUnit:
    return ReuseUnit(
        id=uid,
        type=UnitType.SESSION,
        scope=Scope.SESSION,
        n_tokens=n_tokens,
        n_bytes_by_tier=n_bytes_by_tier,
        residence=residence,
        holders=holders,
    )


# ----------------------------------------------------------------- stages


def stage_0_empty_program() -> None:
    """A program with no committed units → re_use is the empty dict.
    Trivial green-path baseline; defends against an impl that
    accidentally returns ``None``."""
    out = expected_peak_hbm_after_resume(
        program_unit_hashes=[],
        units={"h1": _unit("h1", residence=[Tier.HBM],
                           n_bytes_by_tier={Tier.HBM: {"attn": 4096}},
                           holders=["other"])},
    )
    if out != {}:
        raise StageFail(f"empty program should return {{}}: got {out!r}")


def stage_1_hbm_resident_unit_contributes_zero() -> None:
    """**The B1 fix in isolation.**  Paused program holds unit ``h1``
    which is currently HBM-resident (kept alive by another live
    holder).  re_use[*] MUST be 0 for that unit — no bytes need to
    re-enter HBM upon resume because they're already there."""
    units = {
        "h1": _unit(
            "h1",
            residence=[Tier.HBM, Tier.DRAM],   # write-through state
            n_bytes_by_tier={
                Tier.HBM: {"attn": 4096},
                Tier.DRAM: {"attn": 4096},
            },
            holders=["paused-prog", "live-prog"],
        ),
    }
    out = expected_peak_hbm_after_resume(
        program_unit_hashes=["h1"], units=units,
    )
    if out != {}:
        raise StageFail(
            f"HBM-resident unit should contribute 0 (round-9 B1 fix); "
            f"got re_use={out!r}"
        )


def stage_2_dram_only_unit_full_bytes() -> None:
    """Paused program holds unit ``h1`` that lives ONLY on DRAM
    (other holders evicted from HBM).  On resume, the bytes must
    re-enter HBM — re_use[sp] equals the unit's n_bytes per subpool.
    """
    units = {
        "h1": _unit(
            "h1",
            residence=[Tier.DRAM],
            n_bytes_by_tier={Tier.DRAM: {"attn": 8192}},
            holders=["paused-prog"],
        ),
    }
    out = expected_peak_hbm_after_resume(
        program_unit_hashes=["h1"], units=units,
    )
    if out != {"attn": 8192}:
        raise StageFail(
            f"DRAM-only unit should drive re_use[attn]=8192; got {out!r}"
        )


def stage_3_disk_only_unit_full_bytes() -> None:
    """Same as Stage 2 but the unit dropped all the way to DISK.
    Resume still requires the full byte count to re-enter HBM."""
    units = {
        "h1": _unit(
            "h1",
            residence=[Tier.DISK],
            n_bytes_by_tier={Tier.DISK: {"attn": 16384}},
            holders=["paused-prog"],
        ),
    }
    out = expected_peak_hbm_after_resume(
        program_unit_hashes=["h1"], units=units,
    )
    if out != {"attn": 16384}:
        raise StageFail(
            f"DISK-only unit should drive re_use[attn]=16384; got {out!r}"
        )


def stage_4_mixed_bag_only_non_hbm_counts() -> None:
    """Realistic scenario: paused program holds three units —
       h_hbm:   HBM-resident          → contributes 0
       h_dram:  DRAM-only             → contributes n_bytes
       h_disk:  DISK-only             → contributes n_bytes
    Plus one unit (h_unrelated) NOT held by the program — must not
    contribute at all (the formula only sums over program_unit_hashes).
    """
    units = {
        "h_hbm": _unit(
            "h_hbm",
            residence=[Tier.HBM, Tier.DRAM],
            n_bytes_by_tier={
                Tier.HBM: {"attn": 4096},
                Tier.DRAM: {"attn": 4096},
            },
            holders=["paused-prog"],
        ),
        "h_dram": _unit(
            "h_dram",
            residence=[Tier.DRAM],
            n_bytes_by_tier={Tier.DRAM: {"attn": 2048}},
            holders=["paused-prog"],
        ),
        "h_disk": _unit(
            "h_disk",
            residence=[Tier.DISK],
            n_bytes_by_tier={Tier.DISK: {"attn": 8192}},
            holders=["paused-prog"],
        ),
        "h_unrelated": _unit(
            "h_unrelated",
            residence=[Tier.DRAM],
            n_bytes_by_tier={Tier.DRAM: {"attn": 99999}},
            holders=["another-prog"],
        ),
    }
    out = expected_peak_hbm_after_resume(
        program_unit_hashes=["h_hbm", "h_dram", "h_disk"], units=units,
    )
    if out != {"attn": 2048 + 8192}:
        raise StageFail(
            f"mixed: HBM=0 + DRAM=2048 + DISK=8192 = 10240; "
            f"got re_use={out!r}"
        )


def stage_5_multi_subpool_aggregation() -> None:
    """Multi-stack hybrid (DESIGN §12 S2/S3): a paused program's
    non-HBM units span more than one subpool (e.g. attn + moe_expert
    + ssm).  re_use aggregates per-subpool independently."""
    units = {
        "h_attn": _unit(
            "h_attn",
            residence=[Tier.DRAM],
            n_bytes_by_tier={Tier.DRAM: {"attn": 4096}},
            holders=["paused-prog"],
        ),
        "h_moe": _unit(
            "h_moe",
            residence=[Tier.DRAM],
            n_bytes_by_tier={Tier.DRAM: {"moe_expert": 32768}},
            holders=["paused-prog"],
        ),
        "h_ssm": _unit(
            "h_ssm",
            residence=[Tier.DISK],
            n_bytes_by_tier={Tier.DISK: {"ssm_snapshot": 2048}},
            holders=["paused-prog"],
        ),
        # HBM-resident multi-subpool unit — every subpool contributes 0.
        "h_hbm_multi": _unit(
            "h_hbm_multi",
            residence=[Tier.HBM, Tier.DRAM],
            n_bytes_by_tier={
                Tier.HBM: {"attn": 1024, "moe_expert": 8192},
                Tier.DRAM: {"attn": 1024, "moe_expert": 8192},
            },
            holders=["paused-prog", "other"],
        ),
    }
    out = expected_peak_hbm_after_resume(
        program_unit_hashes=["h_attn", "h_moe", "h_ssm", "h_hbm_multi"],
        units=units,
    )
    if out != {"attn": 4096, "moe_expert": 32768, "ssm_snapshot": 2048}:
        raise StageFail(
            f"multi-subpool aggregation wrong: {out!r}"
        )


def stage_6_partial_drop_credits_reprefill() -> None:
    """#216: a hash NOT in ``units`` (DROPped while the program was gated)
    is credited a re-prefill estimate when the program has SURVIVING units
    (partial drop) — on resume it re-prefills that prefix into HBM, a burst
    the load-back model misses, so ``capacity_fits`` must see it.  The
    estimate is the program's own per-unit mean (per subpool).

    1 surviving DRAM unit (1024 in 'attn') + 2 dropped:
      re_use = 1024 (load-back) + (1024/1)*2 (dropped credit) = 3072."""
    units = {
        "h_present": _unit(
            "h_present",
            residence=[Tier.DRAM],
            n_bytes_by_tier={Tier.DRAM: {"attn": 1024}},
            holders=["paused-prog"],
        ),
    }
    out = expected_peak_hbm_after_resume(
        program_unit_hashes=["h_present", "h_gone", "h_also_gone"],
        units=units,
    )
    if out != {"attn": 3072}:
        raise StageFail(
            f"partial drop must credit dropped units the surviving per-unit "
            f"mean (1024 load-back + 2×1024 dropped = 3072); got {out!r}")


def stage_6b_full_drop_stays_zero() -> None:
    """#216 carve-out preserving #211/#213: a FULLY-dropped program (NO
    surviving unit to size from) keeps re_use = {} so it can still un-starve
    — releasing the gate is free, re-prefill is future work sglang admission-
    controls.  Two live shapes: (a) hashes listed but all gone, (b) empty
    unit_hashes (the overlay's empty residue)."""
    units = {}  # all of this program's units were DROPped
    out_a = expected_peak_hbm_after_resume(["h_gone", "h_also_gone"], units)
    if out_a != {}:
        raise StageFail(f"#211: fully-dropped (hashes listed, all gone) must "
                        f"keep re_use={{}} to un-starve; got {out_a!r}")
    out_b = expected_peak_hbm_after_resume([], units)
    if out_b != {}:
        raise StageFail(f"#211: empty unit_hashes (overlay residue) must keep "
                        f"re_use={{}}; got {out_b!r}")


def stage_6c_hbm_resident_survivor_sizes_dropped() -> None:
    """#216 + B1: a surviving HBM-resident unit contributes 0 to its own
    load-back (B1), but DOES size the dropped-unit credit (we need byte SIZE,
    not residence).  1 HBM-resident survivor (2048 in 'attn') + 1 dropped:
      re_use = 0 (B1, no load-back) + (2048/1)*1 (dropped credit) = 2048."""
    units = {
        "h_hbm": _unit(
            "h_hbm",
            residence=[Tier.HBM],
            n_bytes_by_tier={Tier.HBM: {"attn": 2048}},
            holders=["paused-prog", "other"],
        ),
    }
    out = expected_peak_hbm_after_resume(["h_hbm", "h_gone"], units)
    if out != {"attn": 2048}:
        raise StageFail(
            f"HBM-resident survivor must size the dropped credit (0 load-back "
            f"+ 1×2048 dropped = 2048); got {out!r}")


def stage_6d_no_drop_unchanged() -> None:
    """No dropped units → behaviour is exactly the load-back sum (the #216
    credit never fires).  2 DRAM survivors (1024 + 4096) → {'attn': 5120}."""
    units = {
        "h1": _unit("h1", residence=[Tier.DRAM],
                    n_bytes_by_tier={Tier.DRAM: {"attn": 1024}}, holders=["p"]),
        "h2": _unit("h2", residence=[Tier.DRAM],
                    n_bytes_by_tier={Tier.DRAM: {"attn": 4096}}, holders=["p"]),
    }
    out = expected_peak_hbm_after_resume(["h1", "h2"], units)
    if out != {"attn": 5120}:
        raise StageFail(f"no-drop must equal the load-back sum 5120; got {out!r}")


def stage_6e_multi_subpool_credit() -> None:
    """#216 credit is distributed per subpool at each subpool's surviving
    mean.  2 survivors: one DRAM 'full' 600, one DRAM 'swa' 200; 2 dropped.
      'full': 600 load-back + (600/2)*2 = 600+600 = 1200
      'swa' : 200 load-back + (200/2)*2 = 200+200 = 400
    (mean per subpool divides by the TOTAL survivor count, n=2.)"""
    units = {
        "h_full": _unit("h_full", residence=[Tier.DRAM],
                        n_bytes_by_tier={Tier.DRAM: {"full": 600}}, holders=["p"]),
        "h_swa": _unit("h_swa", residence=[Tier.DRAM],
                       n_bytes_by_tier={Tier.DRAM: {"swa": 200}}, holders=["p"]),
    }
    out = expected_peak_hbm_after_resume(
        ["h_full", "h_swa", "h_d1", "h_d2"], units)
    if out != {"full": 1200, "swa": 400}:
        raise StageFail(f"multi-subpool credit wrong; expected "
                        f"{{'full':1200,'swa':400}}, got {out!r}")


def stage_7_idempotent_pure_function() -> None:
    """The helper is pure — calling it twice with the same args
    returns the same result, and does NOT mutate either input.
    Defends against an impl that builds the return dict in-place
    on the units dict (would corrupt SchedulerState)."""
    units = {
        "h1": _unit(
            "h1",
            residence=[Tier.DRAM],
            n_bytes_by_tier={Tier.DRAM: {"attn": 4096}},
            holders=["paused-prog"],
        ),
    }
    snapshot_residence = list(units["h1"].residence)
    snapshot_n_bytes = {
        t: dict(sp) for t, sp in units["h1"].n_bytes_by_tier.items()
    }
    out_a = expected_peak_hbm_after_resume(["h1"], units)
    out_b = expected_peak_hbm_after_resume(["h1"], units)
    if out_a != out_b:
        raise StageFail(f"non-deterministic: {out_a!r} vs {out_b!r}")
    if units["h1"].residence != snapshot_residence:
        raise StageFail(
            f"input residence mutated: was {snapshot_residence}, "
            f"now {units['h1'].residence}"
        )
    if units["h1"].n_bytes_by_tier != snapshot_n_bytes:
        raise StageFail(
            f"input n_bytes_by_tier mutated: was {snapshot_n_bytes}, "
            f"now {units['h1'].n_bytes_by_tier}"
        )


def stage_8_capacity_fits_no_double_count_scenario() -> None:
    """End-to-end B1 scenario.  Build a state where:
       - HBM cap = 10 KB, currently 4 KB used (unit h1 on HBM held by
         the LIVE program live-prog)
       - Paused program paused-prog ALSO holds h1 (multi-holder)
       - Free HBM = 6 KB
       - re_use(paused-prog) MUST be 0 (B1) → capacity_fits trivially
         passes for paused-prog resume.
       Pre-B1-fix re_use would have been 4096 → capacity_fits would
       check `free (6 KB) >= forecast (0) + re_use (4 KB)` which
       still passes BUT in a tighter HBM (e.g. free=3 KB) it would
       have over-pessimised.  We re-run with free=3 KB to surface
       the bug shape.
    """
    units = {
        "h1": _unit(
            "h1",
            residence=[Tier.HBM, Tier.DRAM],
            n_bytes_by_tier={
                Tier.HBM: {"attn": 4096},
                Tier.DRAM: {"attn": 4096},
            },
            holders=["paused-prog", "live-prog"],
        ),
    }
    re_use = expected_peak_hbm_after_resume(["h1"], units)
    # The B1 property: HBM-resident contributes 0.
    if re_use:
        raise StageFail(
            f"B1 scenario: re_use should be empty for HBM-resident "
            f"shared unit; got {re_use!r}"
        )
    # capacity_fits proxy: with re_use=0, paused-prog resume needs
    # 0 extra HBM bytes — fits even at 100 % HBM occupancy.  Pre-B1
    # re_use would have been 4096 → would refuse resume at 6 KB
    # free / 10 KB cap (since the forecast term might already use
    # part of that).
    hypothetical_free_hbm = 3 * 1024  # tighter than 4 KB
    paused_forecast = 0  # no inflight, just resume capacity
    if not (hypothetical_free_hbm >= paused_forecast + sum(re_use.values())):
        raise StageFail(
            "capacity_fits should pass under tight HBM since re_use=0"
        )


# ----------------------------------------------------------------- run


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("0  empty program → {}",                          stage_0_empty_program),
    ("1  HBM-resident unit contributes 0 (B1 isolated)",
                                                       stage_1_hbm_resident_unit_contributes_zero),
    ("2  DRAM-only unit full bytes",                   stage_2_dram_only_unit_full_bytes),
    ("3  DISK-only unit full bytes",                   stage_3_disk_only_unit_full_bytes),
    ("4  mixed bag — only non-HBM counts",             stage_4_mixed_bag_only_non_hbm_counts),
    ("5  multi-subpool aggregation",                   stage_5_multi_subpool_aggregation),
    ("6  partial-drop credits re-prefill estimate (#216)",
                                                       stage_6_partial_drop_credits_reprefill),
    ("6b full-drop stays zero — un-starve preserved (#211/#213)",
                                                       stage_6b_full_drop_stays_zero),
    ("6c HBM-resident survivor sizes dropped credit (#216+B1)",
                                                       stage_6c_hbm_resident_survivor_sizes_dropped),
    ("6d no-drop unchanged (load-back sum)",           stage_6d_no_drop_unchanged),
    ("6e multi-subpool credit distribution (#216)",    stage_6e_multi_subpool_credit),
    ("7  pure / idempotent / non-mutating",            stage_7_idempotent_pure_function),
    ("8  capacity_fits no-double-count scenario (B1 E2E)",
                                                       stage_8_capacity_fits_no_double_count_scenario),
]


def main() -> int:
    failures: List[str] = []
    for label, fn in _STAGES:
        try:
            fn()
            print(f"  {_green('PASS')}  Stage {label}")
        except StageFail as exc:
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures.append(label)
            print(f"  {_red('FAIL')}  Stage {label}: unexpected {type(exc).__name__}: {exc}")
    if failures:
        print(_red(f"\nT16 FAILED ({len(failures)} stage(s)): {failures}"))
        return 1
    print(_green(f"\nT16 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
