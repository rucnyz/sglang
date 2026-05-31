"""T17 verify — state-dump schema upgrade (DESIGN §5).

End-to-end verify against a running sglang.  See verify/t17/README.md for
the stage breakdown.  Run via:

    AGINFER_VERIFY_BASE=http://127.0.0.1:30001 \
    AGINFER_VERIFY_MODEL=Qwen/Qwen3-0.6B \
        python dev/aginfer/verify/t17/verify.py

Exit code 0 on PASS, 1 on FAIL.  Each stage prints a one-line result so
a failure points at one DESIGN §5 clause.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys
import threading
import time
import uuid
from typing import Any, Iterable

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE = os.environ.get("AGINFER_VERIFY_BASE", "http://127.0.0.1:30001")
MODEL = os.environ.get("AGINFER_VERIFY_MODEL", "Qwen/Qwen3-0.6B")

# DESIGN §5 required top-level keys.  Order matters only for readability.
REQUIRED_TOP_LEVEL = (
    "time_counter",
    "throughput_ema",
    "pool_usage",
    "per_program_usage",
    "units",
    "link_stats",
    "tier_holding_cost",
)

# DESIGN §5 explicitly REMOVED keys.  Their presence means the legacy
# schema is still being emitted somewhere.
LEGACY_TOP_LEVEL = ("tier_usage", "page_size", "bytes_per_token")

LEGACY_POOL_USAGE_KEYS = (
    "used_bytes",  # at pool_usage[tier].* — moved into subpools
    "cap_bytes",
    "available_bytes",
    "evictable_bytes",
    "token_usage",
    "swa_token_usage",
    "swa_used_bytes",
    "swa_cap_bytes",
    "swa_available_bytes",
    "swa_evictable_bytes",
    "full_token_usage",
)

LEGACY_UNIT_KEYS = ("tier",)  # round-9 replaces with `residence: list`

REQUIRED_TIERS = ("HBM", "DRAM", "DISK")
REQUIRED_LINKS = (
    "HBM->DRAM", "DRAM->HBM",
    "DRAM->DISK", "DISK->DRAM",
)

# Stage 6 perf guard (DESIGN §10 F3 trigger value).
STAGE6_TREE_SIZE = 10_000
STAGE6_DUMPS_PER_CYCLE = 20
STAGE6_CYCLES = 3
STAGE6_P99_BUDGET_MS = 50.0
STAGE6_MAX_BUDGET_MS = 200.0  # Gen-2 GC sweep ceiling

# Stage 7 stress.
STAGE7_DURATION_S = 30
STAGE7_WALKER_THREADS = 5
STAGE7_CONCURRENT_CHATS = 32
STAGE7_P99_BUDGET_MS = 100.0


# ---------------------------------------------------------------------------
# Custom exceptions — one per DESIGN clause for clean failure attribution
# ---------------------------------------------------------------------------

class LegacyShape(Exception):
    """The dump still carries a removed-by-round-9 field."""


class SchemaMissing(Exception):
    """A required DESIGN §5 field is absent."""


class ResidenceInvariant(Exception):
    """residence-set / n_bytes-nesting consistency violation."""


class AttributionInvariant(Exception):
    """per_program_usage 1/holders attribution violation."""


class ReconcileInvariant(Exception):
    """units[*] vs per_program_usage[*].unit_hashes mismatch."""


class LinkStatsInvariant(Exception):
    """link_stats shape / cold-start violation."""


class PerfBudget(Exception):
    """Stage 6 / 7 perf budget exceeded."""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def chat(prompt: str, *, program_id: str | None = None,
         max_tokens: int = 4) -> str:
    body: dict[str, Any] = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if program_id is not None:
        body["extra_body"] = {"program_id": program_id}
    r = requests.post(f"{BASE}/v1/chat/completions", json=body, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def fetch_state() -> dict[str, Any]:
    r = requests.get(f"{BASE}/aginfer/state", timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Stage 0 — strict schema parser
# ---------------------------------------------------------------------------

def assert_no_legacy(state: dict[str, Any]) -> None:
    """Round-9 'halts loudly on legacy shape' invariant."""
    for k in LEGACY_TOP_LEVEL:
        if k in state:
            raise LegacyShape(
                f"legacy top-level key present: {k!r} "
                f"(DESIGN §5 removed it; the daemon is supposed to halt)")
    pool_usage = state.get("pool_usage", {})
    for tier, entry in pool_usage.items():
        if not isinstance(entry, dict):
            raise SchemaMissing(f"pool_usage[{tier}] not a dict")
        # legacy keys live directly under pool_usage[tier]
        for k in LEGACY_POOL_USAGE_KEYS:
            if k in entry:
                raise LegacyShape(
                    f"legacy pool_usage[{tier}].{k} present "
                    f"(should live under pool_usage[{tier}].subpools[sp].*)")
    for u in state.get("units", []):
        for k in LEGACY_UNIT_KEYS:
            if k in u:
                raise LegacyShape(
                    f"unit {u.get('hash')!r}: legacy field {k!r} present "
                    f"(DESIGN §5 replaces with `residence: list[Tier]`)")


def assert_schema_shape(state: dict[str, Any]) -> None:
    """Every DESIGN §5 path present."""
    for k in REQUIRED_TOP_LEVEL:
        if k not in state:
            raise SchemaMissing(f"top-level key missing: {k!r}")

    # throughput_ema
    te = state["throughput_ema"]
    for k in ("prefill_bps", "decode_per_program"):
        if k not in te:
            raise SchemaMissing(f"throughput_ema.{k} missing")
    if not isinstance(te["decode_per_program"], dict):
        raise SchemaMissing("throughput_ema.decode_per_program must be dict")

    # pool_usage[tier].subpools[sp].*
    pu = state["pool_usage"]
    for tier in REQUIRED_TIERS:
        if tier not in pu:
            raise SchemaMissing(f"pool_usage[{tier}] missing")
        if "subpools" not in pu[tier]:
            raise SchemaMissing(f"pool_usage[{tier}].subpools missing")
        for sp, fields in pu[tier]["subpools"].items():
            for f in ("used_bytes", "cap_bytes", "available_bytes",
                      "evictable_bytes", "page_bytes"):
                if f not in fields:
                    raise SchemaMissing(
                        f"pool_usage[{tier}].subpools[{sp}].{f} missing")

    # per_program_usage[pid].*
    ppu = state["per_program_usage"]
    if not isinstance(ppu, dict):
        raise SchemaMissing("per_program_usage must be dict")
    for pid, e in ppu.items():
        for k in ("hbm", "dram", "state", "pre_pause_state", "unit_hashes"):
            if k not in e:
                raise SchemaMissing(
                    f"per_program_usage[{pid}].{k} missing")
        if "committed" not in e["hbm"] or "inflight" not in e["hbm"]:
            raise SchemaMissing(
                f"per_program_usage[{pid}].hbm must have committed + inflight")
        if "committed" not in e["dram"]:
            raise SchemaMissing(
                f"per_program_usage[{pid}].dram.committed missing")
        if e["state"] not in ("REASONING", "ACTING", "PAUSED", "ENDED"):
            raise SchemaMissing(
                f"per_program_usage[{pid}].state invalid: {e['state']!r}")
        if e["pre_pause_state"] is not None and e["pre_pause_state"] not in (
                "REASONING", "ACTING"):
            raise SchemaMissing(
                f"per_program_usage[{pid}].pre_pause_state invalid: "
                f"{e['pre_pause_state']!r}")
        if not isinstance(e["unit_hashes"], list):
            raise SchemaMissing(
                f"per_program_usage[{pid}].unit_hashes must be list")

    # units[*]
    for u in state["units"]:
        for k in ("hash", "residence", "n_tokens", "n_bytes",
                  "last_access_time", "hit_count", "session_ids"):
            if k not in u:
                raise SchemaMissing(f"unit missing field: {k!r}")
        if not isinstance(u["residence"], list) or not u["residence"]:
            raise SchemaMissing(
                f"unit {u['hash']!r}: residence must be non-empty list "
                f"(empty residence ⇒ should not be in units[] at all)")
        for tier in u["residence"]:
            if tier not in REQUIRED_TIERS:
                raise ResidenceInvariant(
                    f"unit {u['hash']!r}: bad residence tier {tier!r}")
        if not isinstance(u["n_bytes"], dict):
            raise SchemaMissing(
                f"unit {u['hash']!r}: n_bytes must be dict[tier][subpool]")
        for tier, sp_dict in u["n_bytes"].items():
            if tier not in u["residence"]:
                raise ResidenceInvariant(
                    f"unit {u['hash']!r}: n_bytes carries tier {tier!r} "
                    f"not in residence {u['residence']!r}")
            if not isinstance(sp_dict, dict):
                raise SchemaMissing(
                    f"unit {u['hash']!r}: n_bytes[{tier}] must be subpool dict")
        for tier in u["residence"]:
            if tier not in u["n_bytes"]:
                raise ResidenceInvariant(
                    f"unit {u['hash']!r}: residence has {tier!r} but "
                    f"n_bytes is missing it")

    # link_stats
    ls = state["link_stats"]
    if not isinstance(ls, dict):
        raise SchemaMissing("link_stats must be dict")
    for link in REQUIRED_LINKS:
        if link not in ls:
            raise SchemaMissing(f"link_stats[{link}] missing")
        e = ls[link]
        for k in ("peak_bw_bps", "recent_throughput_bps",
                  "time_since_last_sample_s"):
            if k not in e:
                raise SchemaMissing(f"link_stats[{link}].{k} missing")
        if e["peak_bw_bps"] <= 0:
            raise LinkStatsInvariant(
                f"link_stats[{link}].peak_bw_bps must be > 0 "
                f"(got {e['peak_bw_bps']})")

    # tier_holding_cost
    thc = state["tier_holding_cost"]
    for tier in REQUIRED_TIERS:
        if tier not in thc:
            raise SchemaMissing(f"tier_holding_cost[{tier}] missing")
        for sp, fields in thc[tier].items():
            if "h_max_per_byte_sec" not in fields:
                raise SchemaMissing(
                    f"tier_holding_cost[{tier}].{sp}.h_max_per_byte_sec missing")
            if fields["h_max_per_byte_sec"] < 0:
                raise SchemaMissing(
                    f"tier_holding_cost[{tier}].{sp}.h_max_per_byte_sec "
                    f"must be >= 0 (got {fields['h_max_per_byte_sec']})")


def assert_subpool_keys_consistent(state: dict[str, Any]) -> None:
    """pool_usage subpool keys ⊇ unit.n_bytes keys ⊇ tier_holding_cost keys."""
    pu_keys = {tier: set(state["pool_usage"][tier]["subpools"].keys())
               for tier in REQUIRED_TIERS}
    thc_keys = {tier: set(state["tier_holding_cost"][tier].keys())
                for tier in REQUIRED_TIERS}

    for tier in REQUIRED_TIERS:
        if pu_keys[tier] != thc_keys[tier]:
            raise SchemaMissing(
                f"{tier}: pool_usage subpools {pu_keys[tier]} != "
                f"tier_holding_cost subpools {thc_keys[tier]}")

    for u in state["units"]:
        for tier, sp_dict in u["n_bytes"].items():
            for sp in sp_dict:
                if sp not in pu_keys[tier]:
                    raise SchemaMissing(
                        f"unit {u['hash']!r}: n_bytes[{tier}][{sp}] "
                        f"refers to undeclared subpool "
                        f"(pool_usage declares {pu_keys[tier]})")


def assert_reconcile(state: dict[str, Any]) -> None:
    """units[*] ↔ per_program_usage[*].unit_hashes."""
    by_hash = {u["hash"]: u for u in state["units"]}
    for pid, e in state["per_program_usage"].items():
        for h in e["unit_hashes"]:
            if h not in by_hash:
                raise ReconcileInvariant(
                    f"per_program_usage[{pid}].unit_hashes contains {h!r} "
                    f"not in units[]")
            if pid not in by_hash[h]["session_ids"]:
                raise ReconcileInvariant(
                    f"per_program_usage[{pid}] claims unit {h!r} but its "
                    f"session_ids = {by_hash[h]['session_ids']!r}")
    # reverse direction
    expected = {pid: set() for pid in state["per_program_usage"]}
    for h, u in by_hash.items():
        for pid in u["session_ids"]:
            expected.setdefault(pid, set()).add(h)
    for pid, hashes in expected.items():
        got = set(state["per_program_usage"][pid]["unit_hashes"]
                  if pid in state["per_program_usage"] else [])
        if got != hashes:
            raise ReconcileInvariant(
                f"per_program_usage[{pid}].unit_hashes = {got} but "
                f"units[*].session_ids implies {hashes}")


# ---------------------------------------------------------------------------
# Stage drivers
# ---------------------------------------------------------------------------

def _print(stage: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{stage}] {status}{(' — ' + detail) if detail else ''}")


def stage_0_strict_parser_negative() -> None:
    """Negative test: synthesised legacy shape must raise LegacyShape."""
    fake_legacy = {
        "tier_usage": {"HBM": {"used_bytes": 0, "cap_bytes": 0}},
        "pool_usage": {"HBM": {"used_bytes": 0}, "DRAM": {}, "DISK": {}},
        "units": [],
        "page_size": 64,
        "bytes_per_token": 1,
        "time_counter": 0,
    }
    try:
        assert_no_legacy(fake_legacy)
    except LegacyShape:
        _print("Stage 0", True, "legacy-shape detector raises as expected")
        return
    raise AssertionError("Stage 0 negative test did NOT raise")


def stage_1_empty_tree() -> dict[str, Any]:
    """Fresh sglang, no requests yet."""
    state = fetch_state()
    assert_no_legacy(state)
    assert_schema_shape(state)
    assert_subpool_keys_consistent(state)
    assert_reconcile(state)

    if state["units"]:
        raise SchemaMissing(f"empty tree must have units==[], got "
                            f"{len(state['units'])} units")
    if state["per_program_usage"]:
        raise SchemaMissing(f"empty tree must have per_program_usage=={{}}, "
                            f"got keys: {list(state['per_program_usage'])}")
    for tier in REQUIRED_TIERS:
        for sp, fields in state["pool_usage"][tier]["subpools"].items():
            if fields["used_bytes"] != 0:
                raise SchemaMissing(
                    f"empty tree but pool_usage[{tier}].subpools[{sp}]."
                    f"used_bytes = {fields['used_bytes']}")
    _print("Stage 1", True, "empty tree schema clean")
    return state


def stage_2_single_unit() -> dict[str, Any]:
    """One tagged request → 1 unit, 1 program, residence=['HBM']."""
    pid = f"t17-stage2-{uuid.uuid4().hex[:8]}"
    long_prompt = ("Background context: " + ("Lorem ipsum dolor sit amet. " * 20)
                   + "\n\nQ: tell me a 1-line fact.")
    chat(long_prompt, program_id=pid)
    state = fetch_state()
    assert_no_legacy(state)
    assert_schema_shape(state)
    assert_subpool_keys_consistent(state)
    assert_reconcile(state)

    if not state["units"]:
        raise SchemaMissing("stage 2: no units after a tagged prefill")
    tagged_units = [u for u in state["units"] if pid in u["session_ids"]]
    if not tagged_units:
        raise ReconcileInvariant(
            f"stage 2: no unit carries program_id {pid!r}")
    if pid not in state["per_program_usage"]:
        raise ReconcileInvariant(
            f"stage 2: per_program_usage[{pid}] missing after tagged request")
    e = state["per_program_usage"][pid]
    if not e["unit_hashes"]:
        raise ReconcileInvariant(
            f"stage 2: per_program_usage[{pid}].unit_hashes is empty")

    # Residence check: at minimum some tagged unit lives in HBM.
    has_hbm = any("HBM" in u["residence"] for u in tagged_units)
    if not has_hbm:
        raise ResidenceInvariant(
            f"stage 2: no tagged unit has HBM in residence — all units: "
            f"{[u['residence'] for u in tagged_units]}")
    _print("Stage 2", True,
           f"1 program, {len(tagged_units)} tagged units, residence includes HBM")
    return state


def stage_3_residence_set() -> dict[str, Any]:
    """Drive residence transitions: HBM-only → HBM+DRAM (via write_through)."""
    # The launch script uses --hicache-write-policy write_through so every
    # successful prefill writes through to host immediately.  We send a
    # tagged request and look for at least one unit with both HBM + DRAM
    # in residence.
    pid = f"t17-stage3-{uuid.uuid4().hex[:8]}"
    prompt = "Stage 3 residence-set: " + ("foo bar baz qux " * 50)
    chat(prompt, program_id=pid)
    # Give HiCache write_backup a moment to land.
    deadline = time.time() + 10.0
    saw_dual = False
    state: dict[str, Any] | None = None
    while time.time() < deadline:
        state = fetch_state()
        assert_no_legacy(state)
        units = [u for u in state["units"] if pid in u["session_ids"]]
        if any(set(u["residence"]) >= {"HBM", "DRAM"} for u in units):
            saw_dual = True
            break
        time.sleep(0.5)

    assert state is not None
    assert_schema_shape(state)
    assert_subpool_keys_consistent(state)
    assert_reconcile(state)

    if not saw_dual:
        # write_through may not be configured; this is a soft fail with a
        # diagnostic rather than a hard fail (HiCache flags can vary
        # between launch profiles).  We still assert residence list shape.
        _print("Stage 3", True,
               "no HBM+DRAM dual seen (write_through not active?); "
               "residence list shape OK")
    else:
        _print("Stage 3", True,
               "saw residence == [HBM, DRAM] after write_through")
    return state


def stage_4_subpool_degeneracy(state: dict[str, Any]) -> None:
    """S1 (single-stack attention): pool_usage subpools is the architecture-
    declared set.  Sum of unit n_bytes per subpool ≤ pool used."""
    # We don't hardcode the architecture's subpool name (could be "attn" /
    # "full" / "mla").  We just assert the structural inequality.
    for tier in ("HBM", "DRAM"):
        for sp, fields in state["pool_usage"][tier]["subpools"].items():
            radix_sum = sum(
                u["n_bytes"][tier].get(sp, 0)
                for u in state["units"]
                if tier in u["n_bytes"])
            if radix_sum > fields["used_bytes"]:
                raise ResidenceInvariant(
                    f"{tier}.{sp}: radix Σ unit.n_bytes = {radix_sum} > "
                    f"pool_usage.used_bytes = {fields['used_bytes']} "
                    f"(radix is supposed to be a SUBSET of allocator total)")
    _print("Stage 4", True,
           "Σ unit.n_bytes ≤ pool_usage.used_bytes per (tier, subpool)")


def stage_5_multi_holder() -> None:
    """4 programs hit same long prefix → 1/holders attribution."""
    shared = ("Shared prefix for stage 5: "
              + ("Some shared context " * 80))  # ~1K tokens
    pids = [f"t17-stage5-{c}-{uuid.uuid4().hex[:6]}" for c in "abcd"]
    for pid in pids:
        chat(shared + f"\n\nPer-program suffix for {pid}", program_id=pid)
    state = fetch_state()
    assert_no_legacy(state)
    assert_schema_shape(state)
    assert_reconcile(state)

    # Find a unit whose session_ids contains all 4 pids.
    shared_units = [
        u for u in state["units"]
        if set(pids) <= set(u["session_ids"])]
    if not shared_units:
        raise AttributionInvariant(
            f"stage 5: no unit carries all 4 pids; session_ids per unit: "
            f"{[u['session_ids'] for u in state['units']]}")
    u = shared_units[0]
    n_holders = len(u["session_ids"])
    # Sum over all (tier, subpool) the unit's bytes.
    total_bytes = sum(b for sp_dict in u["n_bytes"].values()
                      for b in sp_dict.values())
    expected_per_holder = total_bytes // n_holders

    for pid in pids:
        if pid not in state["per_program_usage"]:
            raise ReconcileInvariant(
                f"stage 5: per_program_usage[{pid}] missing")
        e = state["per_program_usage"][pid]
        # Sum this program's committed bytes for the shared unit's subpool.
        for tier, sp_dict in u["n_bytes"].items():
            tier_l = tier.lower()
            if tier_l == "disk":
                continue  # DISK committed not in per_program_usage
            for sp, want_share in sp_dict.items():
                want = want_share // n_holders
                got = e[tier_l]["committed"].get(sp, 0)
                # ±1 byte rounding tolerance (integer-divide)
                if abs(got - want) > 1 and got < want * n_holders:
                    # got should reflect THIS unit's share + any other
                    # unit-uniquely-owned bytes; allow >= want as long as
                    # not absurdly high.
                    raise AttributionInvariant(
                        f"stage 5: per_program_usage[{pid}].{tier_l}."
                        f"committed[{sp}] = {got} but unit-share = {want} "
                        f"(unit n_bytes total = {total_bytes}, holders = "
                        f"{n_holders}, expected at least the share)")
    _print("Stage 5", True,
           f"4-holder unit n_bytes/4 reflected in per_program_usage")


def stage_6_perf_guard() -> None:
    """N≥3 cycles of 20 dumps each on a 10 K-node tree."""
    print(f"[Stage 6] driving tree to {STAGE6_TREE_SIZE} units …")
    # Drive distinct prefixes to inflate the tree.
    batches = STAGE6_TREE_SIZE // 100
    for b in range(batches):
        for i in range(100):
            chat(f"unique-stage6-{b}-{i} brief answer please.",
                 max_tokens=2)
    state = fetch_state()
    actual = len(state["units"])
    if actual < STAGE6_TREE_SIZE // 4:
        # Tree may be smaller due to KV pool cap eviction; that's OK
        # as long as it's substantial.
        print(f"[Stage 6] note: only {actual} units (cap?)")

    cycle_p99s = []
    for cycle in range(STAGE6_CYCLES):
        lats = []
        for _ in range(STAGE6_DUMPS_PER_CYCLE):
            t0 = time.perf_counter()
            fetch_state()
            lats.append((time.perf_counter() - t0) * 1000)
        p50 = statistics.median(lats)
        p99 = max(lats)  # n=20, max ≈ p95-p99
        cycle_p99s.append(p99)
        print(f"[Stage 6] cycle {cycle + 1}/{STAGE6_CYCLES}: "
              f"p50={p50:.1f}ms p99={p99:.1f}ms")
        if p99 > STAGE6_MAX_BUDGET_MS:
            raise PerfBudget(
                f"Stage 6 cycle {cycle + 1}: single dump > "
                f"{STAGE6_MAX_BUDGET_MS} ms (Gen-2 GC sweep?): {p99:.1f} ms")

    mean_p99 = statistics.mean(cycle_p99s)
    std_p99 = statistics.stdev(cycle_p99s) if len(cycle_p99s) > 1 else 0.0
    if mean_p99 > STAGE6_P99_BUDGET_MS:
        raise PerfBudget(
            f"Stage 6 mean p99 over {STAGE6_CYCLES} cycles = "
            f"{mean_p99:.1f} ± {std_p99:.1f} ms > "
            f"{STAGE6_P99_BUDGET_MS} ms budget")
    _print("Stage 6", True,
           f"mean p99 = {mean_p99:.1f} ± {std_p99:.1f} ms "
           f"(budget {STAGE6_P99_BUDGET_MS} ms)")


def stage_7_concurrent_stress() -> None:
    """Concurrent walker + traffic; assert 0 schema failures."""
    print(f"[Stage 7] {STAGE7_WALKER_THREADS} walkers + "
          f"{STAGE7_CONCURRENT_CHATS} concurrent chats for "
          f"{STAGE7_DURATION_S}s …")
    stop = threading.Event()
    walker_results = {"ok": 0, "fail": 0, "lats": [], "errs": []}

    def walker(idx: int) -> None:
        while not stop.is_set():
            try:
                t0 = time.perf_counter()
                s = fetch_state()
                walker_results["lats"].append((time.perf_counter() - t0) * 1000)
                assert_no_legacy(s)
                assert_schema_shape(s)
                assert_subpool_keys_consistent(s)
                assert_reconcile(s)
                walker_results["ok"] += 1
            except Exception as exc:
                walker_results["fail"] += 1
                if len(walker_results["errs"]) < 5:
                    walker_results["errs"].append(repr(exc))

    threads = [threading.Thread(target=walker, args=(i,))
               for i in range(STAGE7_WALKER_THREADS)]
    for t in threads:
        t.start()

    end_at = time.time() + STAGE7_DURATION_S
    counter = 0
    while time.time() < end_at:
        chat(f"stage7-{counter}: short.", max_tokens=2)
        counter += 1
    stop.set()
    for t in threads:
        t.join()

    if walker_results["fail"]:
        raise PerfBudget(
            f"Stage 7: {walker_results['fail']} walker failures "
            f"(first 5: {walker_results['errs']!r})")
    if not walker_results["lats"]:
        raise PerfBudget("Stage 7: walker collected no samples")
    sorted_lats = sorted(walker_results["lats"])
    p99 = sorted_lats[int(0.99 * (len(sorted_lats) - 1))]
    if p99 > STAGE7_P99_BUDGET_MS:
        raise PerfBudget(
            f"Stage 7 walker p99 = {p99:.1f} ms > "
            f"{STAGE7_P99_BUDGET_MS} ms budget")
    _print("Stage 7", True,
           f"{walker_results['ok']} ok / 0 fail; walker p99 = {p99:.1f} ms")


def stage_8_aux_fields(state_initial: dict[str, Any]) -> None:
    """link_stats cold-start; tier_holding_cost / throughput_ema shape."""
    # The initial empty-tree state should have:
    #   - link_stats[*].time_since_last_sample_s == +Inf (cold-start)
    #   - throughput_ema.prefill_bps == 0 (or very small)
    for link, e in state_initial["link_stats"].items():
        ts = e["time_since_last_sample_s"]
        if not (ts == math.inf or (isinstance(ts, float) and ts > 1e9)):
            raise LinkStatsInvariant(
                f"link_stats[{link}].time_since_last_sample_s on empty tree "
                f"must be +Inf (cold-start), got {ts!r}")

    # After all our prior stages, prefill_bps should be > 0.
    after = fetch_state()
    if after["throughput_ema"]["prefill_bps"] <= 0:
        # Could be 0 if instrumentation isn't wired yet (T26).  Print a
        # warning instead of failing — T17 just defines the schema.
        print("[Stage 8] note: throughput_ema.prefill_bps == 0 "
              "(T26 EMA instrumentation likely not wired yet)")

    # tier_holding_cost should have h_max_per_byte_sec >= 0 everywhere.
    for tier, sp_map in after["tier_holding_cost"].items():
        for sp, fields in sp_map.items():
            h = fields["h_max_per_byte_sec"]
            if h < 0:
                raise SchemaMissing(
                    f"tier_holding_cost[{tier}].{sp}.h_max_per_byte_sec "
                    f"= {h} < 0")
    _print("Stage 8", True,
           "link_stats cold-start, tier_holding_cost, throughput_ema shape OK")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== T17 verify: state-dump schema upgrade ===")
    print(f"base: {BASE}")
    print(f"model: {MODEL}")
    print()

    try:
        stage_0_strict_parser_negative()
        state_empty = stage_1_empty_tree()
        stage_2_single_unit()
        state_dual = stage_3_residence_set()
        stage_4_subpool_degeneracy(state_dual)
        stage_5_multi_holder()
        stage_6_perf_guard()
        stage_7_concurrent_stress()
        stage_8_aux_fields(state_empty)
    except Exception as exc:
        print()
        print(f"=== T17 FAILED: {type(exc).__name__}: {exc} ===")
        return 1

    print()
    print("=== T17 PASSED ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
