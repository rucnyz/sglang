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

from .base import ReuseUnit, Tier


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

    #216 (partial-drop re-prefill credit): a unit DROPped while the
    program was gated (``units.get(h) is None``) is gone from every tier,
    so the load-back terms above credit it 0 — yet on resume the program
    re-prefills that prefix straight back into HBM (its conversation
    context needs it), a near-instantaneous burst the load-back model
    misses, so ``capacity_fits`` could admit a resume that immediately
    re-pressures HBM.  We credit each dropped hash a re-prefill estimate
    sized from the program's OWN surviving units (mean HBM-equivalent
    bytes per unit, per subpool).

    The credit is applied **only when the program has surviving units**.
    A FULLY-dropped program (no survivor to size from — its units were all
    DROPped, or ``unit_hashes`` is the overlay's empty residue) keeps
    ``re_use = {}`` so it can still un-starve: releasing its proxy gate is
    free, and its re-prefill is future work sglang's allocator admission-
    controls on the actual prefill.  This preserves the #211 / #213
    zero-re_use un-starve exactly (only partial drops, which carry real
    resident context, pay the re-prefill reserve)."""
    re_use: Dict[str, int] = {}
    # For the #216 dropped-unit credit: sum surviving units' HBM-equivalent
    # bytes per subpool (regardless of residence — we need SIZE, not where
    # it lives) and count survivors, to size each dropped hash at the
    # program's own per-unit mean.
    surviving_by_sp: Dict[str, int] = {}
    n_surviving = 0
    n_dropped = 0
    for h in program_unit_hashes:
        unit = units.get(h)
        if unit is None:
            n_dropped += 1
            continue
        n_surviving += 1
        # Size this surviving unit from whichever tier it lives on (size is
        # tier-invariant), for the dropped-unit mean.  HBM first so a
        # fully-HBM-resident program still has a size to credit dropped peers.
        for tier in (Tier.HBM, Tier.DRAM, Tier.DISK):
            sp_dict = unit.n_bytes_by_tier.get(tier, {})
            if sp_dict:
                for sp, n_bytes in sp_dict.items():
                    surviving_by_sp[sp] = surviving_by_sp.get(sp, 0) + int(n_bytes)
                break
        if Tier.HBM in unit.residence:
            # Round-9 B1: already in HBM, no re-entry needed (but it DID
            # contribute its size to surviving_by_sp above).
            continue
        # Load-back: size from whichever non-HBM tier the unit lives on.
        for tier in (Tier.DRAM, Tier.DISK):
            sp_dict = unit.n_bytes_by_tier.get(tier, {})
            if not sp_dict:
                continue
            for sp, n_bytes in sp_dict.items():
                re_use[sp] = re_use.get(sp, 0) + int(n_bytes)
            break  # one tier is enough — per-token size is invariant
    # #216 dropped-unit re-prefill credit (partial drop only).
    if n_dropped and n_surviving:
        for sp, tot in surviving_by_sp.items():
            re_use[sp] = re_use.get(sp, 0) + (tot // n_surviving) * n_dropped
    return re_use
