"""calibration_sanity (§calibration_sanity) — cost-model calibration ratio sanity check.

The Admitter compares 5 cost candidates per arrival:
  own-free, own-evict, cross-free, cross-evict, defer

For that comparison to be meaningful (not theater), the cost
quantities loaded from Stage-0 calibration must produce ratios in a
reasonable range. Specifically:

  - `c^xfer` (transfer wall) ~ 70 µs/chunk × n_0 chunks ≈ 280 µs at n_0=4
  - `c^evict_i(s)` (re-prefill cost on evict) ~ tokens × k_evict
                                              ≈ 12.5 µs/token × s tokens
  - `w_q` (queue penalty) per ms wait

For typical request sizes, `c^evict / c^xfer` should fall in
[0.1, 1000]:
  - < 0.1  : evict always cheaper than transfer → cross-free / cross-evict
             never picked. Cost model degenerates to "always own-evict".
  - > 1000 : transfer always cheaper than evict → own-evict / cross-evict
             never picked. Cost model degenerates to "always cross-free
             (or own-free)".

Spec implementation of cost functions is inline (matches
design.md §"Shared cost model"); replace with import from
sglang/srt/budgeter/cost_model.py once that's wired up.
"""
import sys


# ---------- spec cost functions ----------

def c_xfer(n_chunks: int, per_chunk_floor_us: float = 70.0) -> float:
    """Transfer wall in µs (linear in chunks)."""
    return n_chunks * per_chunk_floor_us


def c_evict_per_token(k_evict_us_per_token: float = 12.5) -> float:
    """Per-token re-prefill cost on eviction (Stage-0 calib)."""
    return k_evict_us_per_token


def c_evict(s_tokens: int, k_evict_us_per_token: float = 12.5,
            hit_prob: float = 1.0) -> float:
    """Expected eviction loss: tokens × per-token cost × hit_prob."""
    return s_tokens * k_evict_us_per_token * hit_prob


def w_q_us_per_ms(w_q: float = 1000.0) -> float:
    """SLO penalty per ms of queue wait."""
    return w_q


# ---------- typical operation profiles ----------

# (label, n_chunks_xfer, s_tokens_evict, hit_prob)
PROFILES = [
    # Small KV: 4-chunk fire (~16 MiB ≈ 800 tokens), evict a small prefix
    ("small-KV: n=4, s=800",       4,   800,  0.5),
    # Mid: 8-chunk fire, mid prefix
    ("mid-KV: n=8, s=4K",          8,   4000, 0.5),
    # Mamba per-slot: 30-chunk fire, evicting a large prefix
    ("mamba-slot: n=30, s=8K",     30,  8000, 0.5),
    # Cold cache: low hit prob → cheap evict
    ("cold-cache: n=4, s=8K @0.05", 4,   8000, 0.05),
    # Hot cache: high hit prob → expensive evict
    ("hot-cache: n=4, s=16K @0.95", 4,   16000, 0.95),
]


# ---------- tests ----------

def test_calibration_ratios_in_range():
    """For each typical operation, compute c^evict / c^xfer and assert
    the ratio is in [0.1, 1000]."""
    failures = []
    print("    profile                          c^xfer   c^evict   ratio")
    print("    " + "-" * 65)
    for label, n_chunks, s_tokens, hit_prob in PROFILES:
        cx = c_xfer(n_chunks)
        ce = c_evict(s_tokens, hit_prob=hit_prob)
        ratio = ce / cx
        ok = 0.1 <= ratio <= 1000
        mark = "✓" if ok else "✗"
        print(f"    {label:32s} {cx:>7.0f} µs  {ce:>7.0f} µs  "
              f"{ratio:>6.2f}× {mark}")
        if not ok:
            failures.append(
                f"{label}: c^evict/c^xfer = {ratio:.3f} (out of [0.1, 1000])")
    assert not failures, (
        "Calibration ratio out of meaningful range — cost model would "
        "degenerate. Failures:\n  " + "\n  ".join(failures))


def test_defer_cost_meaningful_vs_xfer():
    """Defer cost (Q × w_q) at typical queue depths should be in the
    same order of magnitude as evict / xfer costs. Otherwise defer is
    either always chosen (Q × w_q dominates) or never chosen (~0)."""
    # Per design.md: w_q = SLO penalty per ms of queue wait
    # Queue depth Q is per-tick observation; typical Q ∈ [0, 100] under
    # moderate load.
    w_q = w_q_us_per_ms()
    failures = []
    print("    Q (queue)   defer cost   vs xfer(n=8)   vs evict(s=4K)")
    print("    " + "-" * 60)
    cx = c_xfer(8)
    ce = c_evict(4000, hit_prob=0.5)
    for Q in [1, 5, 25, 100]:
        defer_cost = Q * w_q  # µs
        r_xfer = defer_cost / cx
        r_evict = defer_cost / ce
        print(f"    {Q:>5d}      {defer_cost:>7.0f} µs   "
              f"{r_xfer:>7.2f}×       {r_evict:>7.2f}×")
        # At some Q in this range, defer should be competitive
        # (i.e., within 10× of either xfer or evict)
    # Soft check: at Q=25, defer should be reachable
    Q = 25
    defer_at_25 = Q * w_q
    if defer_at_25 > 1000 * cx and defer_at_25 > 1000 * ce:
        failures.append(
            f"At Q={Q}, defer cost {defer_at_25} µs is > 1000× both "
            f"xfer ({cx}) and evict ({ce}) — defer would never be chosen "
            f"in the operating range Q ∈ [0, 100].")
    assert not failures, "\n  ".join(failures)


def test_admitter_5_candidates_separable():
    """With our default calibration, check that for some
    realistic state, NO single candidate's cost is 100× ALL others.
    (If one always dominates, no real cost comparison is happening.)"""
    # Scenario: dst full, src has FREE, evict cost moderate, queue light
    # Candidates:
    #   own-free   : ∞ (dst full)
    #   own-evict  : c_evict(dst, s=4K, hit=0.5)
    #   cross-free : c_xfer(n=8)
    #   cross-evict: c_xfer(n=8) + c_evict(src, s=4K, hit=0.5)
    #   defer      : Q × w_q with Q=10
    cands = {
        "own-free":    float("inf"),
        "own-evict":   c_evict(4000, hit_prob=0.5),
        "cross-free":  c_xfer(8),
        "cross-evict": c_xfer(8) + c_evict(4000, hit_prob=0.5),
        "defer":       10 * w_q_us_per_ms(),
    }
    finite_costs = {k: v for k, v in cands.items() if v != float("inf")}
    print("    candidate     cost (µs)")
    for k, v in sorted(finite_costs.items(), key=lambda kv: kv[1]):
        print(f"    {k:14s} {v:>10.0f}")

    cheapest = min(finite_costs.values())
    second   = sorted(finite_costs.values())[1]
    ratio    = second / cheapest
    print(f"    cheapest = {cheapest:.0f} µs, 2nd = {second:.0f} µs, "
          f"ratio = {ratio:.1f}×")
    assert ratio < 100, (
        f"2nd-cheapest is {ratio:.1f}× the cheapest at typical state — "
        f"effectively one candidate always wins. Cost model not "
        f"discriminating; check calibration.")


def main():
    tests = [
        ("calibration ratios in [0.1, 1000] range", test_calibration_ratios_in_range),
        ("defer cost meaningful vs xfer/evict",     test_defer_cost_meaningful_vs_xfer),
        ("5 candidates separable at typical state", test_admitter_5_candidates_separable),
    ]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}")
            print(f"        {e}")
    print(f"\nD6m-cal: {passed}/{len(tests)} passed")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())
