"""Tier-1 characterization sweep (#230) — produces the headline tables.

Runs the deterministic simulator across the slice matrix and prints
markdown tables for RESULTS.md.  CPU, seconds.  Metric is re-prefill
tokens (lower = better eviction) + cache-hit rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim import make_workload, simulate  # noqa: E402

N_CTX = 24
TOK = 1000


def _wl(tool_gap=20, decode=6, rounds=4, phat_lead=8, spread=3):
    return make_workload(n_programs=N_CTX, n_tokens=TOK, decode_steps=decode,
                         tool_gap=tool_gap, rounds=rounds,
                         arrival_spread=spread, phat_lead=phat_lead)


# Pressure regimes = pool capacity as a fraction of total resident demand.
REGIMES = [
    ("under (75%)", int(0.75 * N_CTX) * TOK),
    ("critical (50%)", int(0.50 * N_CTX) * TOK),
    ("tight (33%)", int(0.33 * N_CTX) * TOK),
    ("saturated (20%)", int(0.20 * N_CTX) * TOK),
]


def _rp(wl, cap, policy, delay=0, migrate=False):
    m = simulate(wl, pool_cap_tokens=cap, policy=policy,
                 hint_delay_steps=delay, migrate=migrate)
    return m.reprefill_tokens, m.hit_rate(), m


def s1_s4_policy_x_regime():
    print("\n## S1+S4 — policy × pressure (UQ1 value of steering, UQ4 where it binds)\n")
    print("re-prefill tokens (hit-rate); lower is better\n")
    print("| regime | LRU | ours-fresh | const-V_u | ours gain vs LRU |")
    print("|---|---|---|---|---|")
    wl = _wl()
    for name, cap in REGIMES:
        lru, lh, _ = _rp(wl, cap, "lru")
        ours, oh, _ = _rp(wl, cap, "ours", delay=0)
        const, ch, _ = _rp(wl, cap, "const")
        gain = (lru - ours) / lru * 100 if lru else 0.0
        print(f"| {name} | {lru} ({lh:.2f}) | {ours} ({oh:.2f}) | "
              f"{const} ({ch:.2f}) | {gain:+.0f}% |")


def s2_delay_gradient():
    print("\n## S2 — hint-freshness latency budget (UQ2, the headline)\n")
    print("ours re-prefill tokens vs hint delay (steps); LRU shown as the ceiling\n")
    delays = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    # critical + tight regimes (where steering discriminates).
    for name, cap in [r for r in REGIMES if "critical" in r[0] or "tight" in r[0]]:
        wl = _wl()
        lru, _, _ = _rp(wl, cap, "lru")
        row = [(_rp(wl, cap, "ours", delay=d)[0]) for d in delays]
        print(f"\n**{name}** (LRU ceiling = {lru}):\n")
        print("| delay (steps) | " + " | ".join(str(d) for d in delays) + " |")
        print("|" + "---|" * (len(delays) + 1))
        print("| re-prefill | " + " | ".join(str(r) for r in row) + " |")
        # locate the knee: first delay where re-prefill ≥ 50% of the way to LRU.
        if lru > row[0]:
            half = row[0] + 0.5 * (lru - row[0])
            knee = next((delays[i] for i, r in enumerate(row) if r >= half), delays[-1])
            print(f"\n→ knee ≈ **{knee} steps** (delay where benefit half-decays "
                  f"= the freshness budget; physically the predictable-reuse lead time)")


def s5_reuse_sensitivity():
    print("\n## S5 — reuse-structure sensitivity (UQ5)\n")
    print("critical regime, ours-fresh vs LRU, across reuse patterns\n")
    cap = int(0.50 * N_CTX) * TOK
    patterns = [
        ("imminent (short tool gap=8)", _wl(tool_gap=8, phat_lead=6)),
        ("delayed (long tool gap=40)", _wl(tool_gap=40, phat_lead=8)),
        ("high-churn (rounds=8)", _wl(rounds=8, tool_gap=16)),
        ("one-shot (rounds=1)", _wl(rounds=1, tool_gap=20)),
    ]
    print("| reuse pattern | LRU | ours-fresh | gain |")
    print("|---|---|---|---|")
    for name, wl in patterns:
        lru, _, _ = _rp(wl, cap, "lru")
        ours, _, _ = _rp(wl, cap, "ours", delay=0)
        gain = (lru - ours) / lru * 100 if lru else 0.0
        print(f"| {name} | {lru} | {ours} | {gain:+.0f}% |")


def s3_imperative():
    print("\n## S3 — imperative migrate contribution (UQ3)\n")
    print("ours-fresh inline-steering, migrate OFF vs ON, by regime\n")
    print("| regime | migrate OFF | migrate ON | Δ | demoted-tokens(ON) |")
    print("|---|---|---|---|---|")
    wl = _wl()
    for name, cap in REGIMES:
        off = simulate(wl, pool_cap_tokens=cap, policy="ours", migrate=False)
        on = simulate(wl, pool_cap_tokens=cap, policy="ours", migrate=True)
        d = off.reprefill_tokens - on.reprefill_tokens
        print(f"| {name} | {off.reprefill_tokens} | {on.reprefill_tokens} | "
              f"{d:+d} | {on.migrate_demote_tokens} |")


def main():
    print("# Tier-1 eviction characterization sweep (#230)\n")
    print(f"workload: {N_CTX} agents × {TOK} tok ctx, phased tool round-trips; "
          f"deterministic.\n")
    s1_s4_policy_x_regime()
    s2_delay_gradient()
    s5_reuse_sensitivity()
    s3_imperative()
    print("\n_Tier-1 models scheduling logic + timing, not GPU kernels; "
          "Tier-2 e2e grounds absolute latencies._")


if __name__ == "__main__":
    main()
