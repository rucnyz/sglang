"""
Calibration knobs for the cost functions. Plug in measurements from V4-Flash
benchmarks; defaults here are reasonable order-of-magnitude placeholders.

Paper Section 2.2 / Section 7. All units explicit so you can sanity-check.
"""
from __future__ import annotations
from .base import Tier
from .ours_greedy import TierCosts


# Conservative defaults: tier 3 (HBM) ~ free, tier 2 (DRAM) ~ PCIe Gen5 x16 read,
# tier 1 (NVMe) ~ enterprise NVMe. Calibrate from real measurements once smoke is up.
DEFAULT_BW = {
    (Tier.HBM, Tier.DRAM): 64e9,    # 64 GB/s PCIe G5 x16 read (B300 effective)
    (Tier.DRAM, Tier.HBM): 64e9,
    (Tier.DRAM, Tier.DISK): 7e9,    # NVMe seq write ~7 GB/s
    (Tier.DISK, Tier.DRAM): 7e9,
    (Tier.HBM, Tier.DISK): 7e9,     # bottleneck is the disk
    (Tier.DISK, Tier.HBM): 7e9,
}

# Reload cost per token (sec/tok). DeepSeek-V4-Flash MLA + NSA on B300:
# - direct GPU use (tier 3): 0
# - reload from DRAM (tier 2): ~PCIe copy of ~512 B KV/token => ~8 ns/token (negligible)
# - reload from NVMe (tier 1): ~70 ns/token at 7 GB/s
# - re-prefill (tier 0): ~50 us/token at ~20K tok/s prefill speed (measure later)
DEFAULT_RHO = {
    Tier.HBM: 0.0,
    Tier.DRAM: 8e-9,
    Tier.DISK: 70e-9,
    # Tier.DROP is per-unit (pi_u), not a tier-wide rho.
}

# Holding cost coefficient (per byte per second). The point of h is to encode
# opportunity cost (memory could host something more valuable). Calibrated so
# h * b_u * (1 sec) is on the same order as the prefill cost it displaces.
DEFAULT_H_BASE = {
    Tier.HBM: 1e-9,    # HBM is precious
    Tier.DRAM: 1e-11,
    Tier.DISK: 1e-13,
}


def default_costs() -> TierCosts:
    return TierCosts(rho=DEFAULT_RHO, h_base=DEFAULT_H_BASE, bw=DEFAULT_BW)
