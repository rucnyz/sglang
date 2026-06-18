"""T35 — authoritative_tier(residence) (DESIGN §7).

The rule (DESIGN §7):
  authoritative_tier(residence) = HBM if HBM ∈ residence
                                  else DRAM if DRAM ∈ residence
                                  else DISK if DISK ∈ residence
                                  else raise (empty residence ≡ unit
                                              shouldn't appear in
                                              units[] per DESIGN §5)

This rule drives:
  * V_u's holding-cost denominator (the tier whose h_(τ, sp) governs
    the unit's storage tax).
  * Migration source for `bytes_at(u, σ)` in DESIGN §7 transitions.
  * admission_controller's tier classification.

Implementation: `baselines.base.ReuseUnit.authoritative_tier`
@property.

Stages (8):
  A0 HBM in residence → HBM (regardless of co-residence in DRAM/DISK)
  A1 only DRAM → DRAM
  A2 only DISK → DISK
  A3 HBM+DRAM (post-write_through) → HBM
  A4 HBM+DISK (rare but legal) → HBM
  A5 DRAM+DISK (mid-migrate) → DRAM
  A6 HBM+DRAM+DISK (post-write_through with disk backup) → HBM
  A7 empty residence → ValueError (deployment-bug class — unit shouldn't
     appear in units[] per DESIGN §5)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Tuple


_HERE = Path(__file__).resolve().parent
_AGINFER_ROOT = _HERE.parent.parent
if str(_AGINFER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGINFER_ROOT))

from baselines.base import ReuseUnit, Scope, Tier, UnitType  # noqa: E402


def _green(s: str) -> str: return f"\033[32m{s}\033[0m"
def _red(s: str) -> str:   return f"\033[31m{s}\033[0m"


class StageFail(AssertionError):
    pass


def _u(residence: List[Tier]) -> ReuseUnit:
    """Minimal ReuseUnit with the requested residence."""
    return ReuseUnit(
        id="u", type=UnitType.SESSION, scope=Scope.SESSION,
        n_tokens=100,
        n_bytes_by_tier={t: {"kv": 100 * 2048} for t in residence},
        residence=list(residence),
        age_seconds=1.0, p_hat=0.5, lambda_rate=0.1, holders=["p"],
    )


def stage_a0_hbm_dominates() -> None:
    u = _u([Tier.HBM])
    if u.authoritative_tier is not Tier.HBM:
        raise StageFail(f"HBM-only → {u.authoritative_tier}")


def stage_a1_dram_only() -> None:
    u = _u([Tier.DRAM])
    if u.authoritative_tier is not Tier.DRAM:
        raise StageFail(f"DRAM-only → {u.authoritative_tier}")


def stage_a2_disk_only() -> None:
    u = _u([Tier.DISK])
    if u.authoritative_tier is not Tier.DISK:
        raise StageFail(f"DISK-only → {u.authoritative_tier}")


def stage_a3_hbm_dram() -> None:
    """Post-write_through state: unit in BOTH HBM and DRAM.
    Authoritative is HBM (the fastest)."""
    u = _u([Tier.HBM, Tier.DRAM])
    if u.authoritative_tier is not Tier.HBM:
        raise StageFail(f"HBM+DRAM → {u.authoritative_tier}")


def stage_a4_hbm_disk() -> None:
    """Rare but legal: HBM-resident + Mooncake L3 backup."""
    u = _u([Tier.HBM, Tier.DISK])
    if u.authoritative_tier is not Tier.HBM:
        raise StageFail(f"HBM+DISK → {u.authoritative_tier}")


def stage_a5_dram_disk() -> None:
    """Mid-migrate state: evicted from HBM, still in DRAM, with
    backup in DISK.  Authoritative is DRAM (fastest available)."""
    u = _u([Tier.DRAM, Tier.DISK])
    if u.authoritative_tier is not Tier.DRAM:
        raise StageFail(f"DRAM+DISK → {u.authoritative_tier}")


def stage_a6_all_three() -> None:
    """Post-write_through + disk backup: residence = {HBM, DRAM, DISK}.
    Authoritative is HBM."""
    u = _u([Tier.HBM, Tier.DRAM, Tier.DISK])
    if u.authoritative_tier is not Tier.HBM:
        raise StageFail(f"HBM+DRAM+DISK → {u.authoritative_tier}")


def stage_a7_empty_raises() -> None:
    """DESIGN §5: empty-residence units don't appear in units[].
    Constructing one + reading authoritative_tier raises — surfaces
    a contract violation loud rather than picking an arbitrary tier."""
    # Build directly (bypassing _u so the residence is genuinely
    # empty AND n_bytes_by_tier is empty).
    u = ReuseUnit(
        id="bad", type=UnitType.SESSION, scope=Scope.SESSION,
        n_tokens=100, n_bytes_by_tier={}, residence=[],
        age_seconds=1.0, p_hat=0.5, lambda_rate=0.1, holders=["p"],
    )
    try:
        _ = u.authoritative_tier
    except ValueError:
        return
    raise StageFail("empty residence should raise ValueError")


_STAGES: List[Tuple[str, Callable[[], None]]] = [
    ("A0 HBM-only → HBM",              stage_a0_hbm_dominates),
    ("A1 DRAM-only → DRAM",            stage_a1_dram_only),
    ("A2 DISK-only → DISK",            stage_a2_disk_only),
    ("A3 HBM+DRAM → HBM",              stage_a3_hbm_dram),
    ("A4 HBM+DISK → HBM",              stage_a4_hbm_disk),
    ("A5 DRAM+DISK → DRAM",            stage_a5_dram_disk),
    ("A6 HBM+DRAM+DISK → HBM",         stage_a6_all_three),
    ("A7 empty residence → ValueError", stage_a7_empty_raises),
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
            print(
                f"  {_red('FAIL')}  Stage {label}: "
                f"unexpected {type(exc).__name__}: {exc}"
            )
    if failures:
        print(_red(f"\nT35 FAILED ({len(failures)}): {failures}"))
        return 1
    print(_green(f"\nT35 PASS — all {len(_STAGES)} stages green"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
