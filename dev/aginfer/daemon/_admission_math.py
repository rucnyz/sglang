"""Pure helpers for DESIGN §8 admission math (capacity_fits / re_use).

T16 (PLAN §2) lands the Round-9 part 1 B1 fix as an isolated pure
function so the property is testable independently of T34's
multi-axis DP candidate generator (#156 will wire this in).

DESIGN §8 ``Resume(p, gain, re_use)`` candidates:

  re_use[sp] = bytes that would re-enter HBM upon resume of p.

Round-9 B1 fix: a paused program's unit that is **still HBM-resident**
(kept alive by other live holders) MUST contribute 0 to re_use[sp].
Reason: capacity_fits' first ``forecast`` term already accounts for
those bytes — adding them again via ``re_use`` double-counts, and
the paused program gets stuck unable to resume because the "needed"
bytes are reported as needed twice.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping

from baselines.base import ReuseUnit, Tier


def expected_peak_hbm_after_resume(
    program_unit_hashes: Iterable[str],
    units: Mapping[str, ReuseUnit],
) -> Dict[str, int]:
    """Return ``re_use[sp]`` = per-HBM-subpool bytes that would re-
    enter HBM upon resume of the program identified by
    ``program_unit_hashes``.

    Pure function; does NOT mutate either input.  T34 (multi-axis DP,
    #156) calls this inside the Resume-candidate builder.

    Round-9 part 1 B1 (see module docstring): ``Tier.HBM in unit
    .residence`` ⇒ unit contributes 0 to every subpool.  Hashes that
    aren't in ``units`` (legitimate: DROPped post-pause) contribute
    0 silently — pre-filtering is the caller's option, not a
    requirement of this function.

    For units whose HBM bytes would need to re-enter (residence does
    not include HBM), we size the would-be HBM byte count by reading
    the unit's ``n_bytes_by_tier`` for any tier the unit IS on (per-
    token byte size is identical across tiers by the sglang schema's
    invariant — see DESIGN §6 L736).
    """
    re_use: Dict[str, int] = {}
    for h in program_unit_hashes:
        unit = units.get(h)
        if unit is None:
            # DROPped post-pause; no bytes to bring back.
            continue
        if Tier.HBM in unit.residence:
            # Round-9 B1: already in HBM, no re-entry needed.
            continue
        # Size from whichever non-HBM tier the unit lives on.  Prefer
        # DRAM (cheapest size to look up; identical to HBM byte count
        # post-write-through), fall back to DISK for cold units.
        for tier in (Tier.DRAM, Tier.DISK):
            sp_dict = unit.n_bytes_by_tier.get(tier, {})
            if not sp_dict:
                continue
            for sp, n_bytes in sp_dict.items():
                re_use[sp] = re_use.get(sp, 0) + int(n_bytes)
            break  # one tier is enough — per-token size is invariant
    return re_use
