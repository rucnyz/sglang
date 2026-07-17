"""byte_transfer — end-to-end byte transfer + working-set invariant validator.

Parses budgeter.jsonl + server.log from a run_byte_transfer.sh run and
asserts the four conjectures from design.md §byte_transfer (plus (d) added
2026-05-26 after active-fix v2 exposed that prior byte_transfer was passing on
phantom fires triggered by radix-cache LRU saturation, not real
admission pressure):

  (a) ≥ 1 non-aborted budgeter fire emitted
  (b) per fire: dst pool grew, src pool shrank, deltas match in
      magnitude (within a per-fire chunk_unit slack of 1, since
      kv_tokens_per_fire and mamba_slots_per_fire are coarse units)
  (c) per fire: no engine error in server.log between fire start
      and the next snapshot — proves the set_capacity didn't OOM
      or cause an unmap-while-mapped failure
  (d) policy-correct: for each fire, the DST pool's `usage_*_active`
      at fire time is ≥ 0.50 — proving the fire happened in response
      to real admission pressure (running reqs holding slots), not
      to phantom pressure from LRU-evictable cache. Required since
      active-fix v2 (xpool_planner reads usage_*_active); a fire
      with usage_*_active near 0 would mean the planner is still
      classifying on phantom signal, which we don't want.

The working-set invariant (`m_src_mapped_after >= working_set +
safety_margin`) is implicit in V1 logical actuator: it pre-evicts
to make room BEFORE shrinking, then calls set_capacity to the new
cap. If working-set were violated the engine would raise on the
next admit attempt — we check server.log for those errors.

Fail-closed: 0 fires → FAIL (workload didn't saturate enough);
any fire fails an invariant → FAIL.
"""
import json
import os
import re
import sys


def parse_budgeter_jsonl(path):
    """Return all decision records, with `fire_*` fields normalized to
    None when absent."""
    out = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            out.append(rec)
    return out


def parse_server_errors(path):
    """Return (line_number, text) for engine errors in server.log,
    excluding known-benign module-load warnings (cutlass `Unexpected
    error during package walk` is sglang's own boot chatter, fires
    regardless of interlayer state)."""
    error_pats = [
        re.compile(r"Scheduler hit an exception"),
        re.compile(r"Traceback \(most recent call last\)"),
        re.compile(r"CUDA_ERROR|cuda.*error|IllegalAddress", re.I),
        re.compile(r"\bOutOfMemory|\bOOM\b", re.I),
        re.compile(r"set_capacity.*fail|capacity.*overflow", re.I),
        re.compile(r"pool memory leak detected"),
        re.compile(r"SIGQUIT received"),
    ]
    benign_pats = [
        re.compile(r"CUTE_DSL.*Unexpected error during package walk"),
        re.compile(r"^\[\d{4}-\d{2}-\d{2} [\d:]+\] "
                   r"Unexpected error during package walk: cutlass"),
    ]
    errors = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if any(b.search(line) for b in benign_pats):
                continue
            for p in error_pats:
                if p.search(line):
                    errors.append((i, line.rstrip()))
                    break
    return errors


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    budg_path = os.path.join(args.out_dir, "budgeter.jsonl")
    log_path  = os.path.join(args.out_dir, "server.log")

    if not os.path.exists(budg_path):
        print(f"FAIL: {budg_path} missing (boot crashed?)")
        return 1
    if not os.path.exists(log_path):
        print(f"FAIL: {log_path} missing")
        return 1

    records = parse_budgeter_jsonl(budg_path)
    fires = [r for r in records
             if r.get("fire_direction") and r.get("fire_direction") != "none"
             and not r.get("fire_aborted", False)]
    aborted = [r for r in records if r.get("fire_aborted")]
    errors = parse_server_errors(log_path)

    print(f"\n{'='*78}")
    print(f"byte_transfer — end-to-end byte transfer + working-set invariant")
    print(f"{'='*78}")
    print(f"budgeter records: {len(records)}")
    print(f"non-aborted fires: {len(fires)}")
    print(f"aborted fires: {len(aborted)}")
    print(f"server-log error lines: {len(errors)}")

    all_ok = True

    # (a) ≥1 fire
    if len(fires) == 0:
        print("\nFAIL (a): no non-aborted fires emitted — workload didn't "
              "trigger budgeter. Possible: workload too short, mem-fraction "
              "too high, or budgeter disabled (check SGLANG_HIMA=1).")
        if aborted:
            print(f"     ({len(aborted)} aborted fires logged — see "
                  f"fire_abort_reason)")
        return 1
    print(f"\n(a) ≥1 fire — PASS ({len(fires)} non-aborted)")

    # (b) per-fire byte-transfer invariant (FirePlanResult schema):
    #   - unmapped_pages > 0 (src actually released pages)
    #   - granted_pages > 0 (dst actually received pages)
    #   - granted_pages == unmapped_pages (no handles lost in transit)
    # AND shared_pool free_count invariant (handles re-bound, not lost):
    #   - shared_free_after == shared_free_before (across a successful fire)
    print(f"\n(b) per-fire byte-transfer invariant")
    print(f"    {'tick':>6s} {'dir':>14s} {'unmap':>7s} {'granted':>9s} "
          f"{'shared_free Δ':>15s}  {'OK?':>6s}")
    b_ok = True
    for r in fires:
        dir_ = r["fire_direction"]
        unmap = r.get("fire_unmapped_pages", 0)
        grant = r.get("fire_granted_pages", 0)
        sf_before = r.get("fire_shared_free_before", -1)
        sf_after = r.get("fire_shared_free_after", -1)
        sf_delta = sf_after - sf_before if (sf_before >= 0 and sf_after >= 0) else None
        # All three must hold for a clean fire
        ok = (unmap > 0 and grant > 0 and unmap == grant
              and sf_delta == 0)
        mark = "✓" if ok else "✗"
        sf_str = f"{sf_before}→{sf_after}" if sf_delta is not None else "?"
        print(f"    {r.get('tick','?'):>6} {dir_:>14s} {unmap:>7d} {grant:>9d} "
              f"{sf_str:>15s}  {mark:>6s}")
        if not ok:
            b_ok = False
    if not b_ok:
        all_ok = False
        print("\nFAIL (b): one or more fires violated invariants "
              "(unmapped≠granted, zero pages, or shared free_count changed).")
    else:
        print(f"    PASS — all {len(fires)} fires moved bytes cleanly")

    # (c) no engine errors during the run
    if errors:
        print(f"\n(c) server-log error scan — FAIL")
        for ln_no, txt in errors[:10]:
            print(f"    line {ln_no}: {txt[:120]}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more")
        all_ok = False
    else:
        print(f"\n(c) server-log error scan — PASS (no ERROR/OOM/CUDA/Traceback "
              f"lines)")

    # (d) policy-correctness: each fire happened in response to REAL
    # admission pressure, not phantom radix-cache LRU saturation. The
    # DST pool (the one being grown) is the side the planner believed
    # was pressured. Its usage_*_active (running-req slots only,
    # evictable cache excluded) at fire time must be ≥ 0.50.
    #
    # Fire records (fire_completion=True) don't carry snapshot fields
    # — we JOIN with the most recent snapshot whose ts ≤ fire ts. The
    # snapshot tick interval is ~1s and fire latency is <100ms, so
    # the join window is short.
    snapshots = [r for r in records if not r.get("fire_completion")]
    snapshots.sort(key=lambda r: r.get("ts", 0))
    def _snap_at(fire_ts):
        prev = None
        for s in snapshots:
            if s.get("ts", 0) > fire_ts:
                break
            prev = s
        return prev or {}
    print(f"\n(d) policy-correctness: fires responded to real "
          f"admission pressure (usage_dst_active ≥ 0.50)")
    print(f"    {'snap_tick':>10s} {'dir':>14s} {'kv_active':>10s} "
          f"{'m_active':>10s} {'running':>8s} {'queue':>6s}  {'OK?':>4s}")
    d_ok = True
    for r in fires:
        dir_ = r["fire_direction"]
        snap = _snap_at(r.get("ts", 0))
        u_kv_act = (snap.get("usage_kv_active") or 0.0)
        u_m_act  = (snap.get("usage_mamba_active") or 0.0)
        running = snap.get("num_running_reqs", 0) or 0
        queue   = snap.get("num_queue_reqs", 0) or 0
        snap_tick = snap.get("tick", "?")
        # k2m grows mamba → mamba is the pressured DST
        # m2k grows kv → kv is the pressured DST
        dst_active = u_m_act if dir_ == "kv_to_mamba" else u_kv_act
        ok = dst_active >= 0.50
        mark = "✓" if ok else "✗"
        print(f"    {snap_tick:>10} {dir_:>14s} {u_kv_act:>10.3f} "
              f"{u_m_act:>10.3f} {running:>8} {queue:>6}  {mark:>4s}")
        if not ok:
            d_ok = False
    if not d_ok:
        all_ok = False
        print("\nFAIL (d): at least one fire was triggered when the dst "
              "pool's usage_*_active was < 0.50 — meaning the planner "
              "acted on phantom pressure (likely radix-cache LRU fill, "
              "not real running-req demand). Either workload didn't "
              "actually saturate the dst pool, or planner is misreading "
              "the active-usage field.")
    else:
        print(f"    PASS — all {len(fires)} fires acted on real pressure")

    print(f"\nD7: {'ALL PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
